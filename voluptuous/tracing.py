"""Opt-in structured validation tracing and value provenance.

This module implements the optional validation trace surfaced through
:meth:`voluptuous.Schema.validate_with_trace`.  It records, for a single
validation pass, which schema node each input value travelled through,
where defaults were generated, where values were coerced, which
combinator branches failed or were selected, and where errors were
aggregated.

Design invariants:

* The default ``Schema.__call__`` path is untouched: tracing uses a
  separately compiled validator tree, so validators are never executed
  twice and the untraced path keeps its original performance profile.
* Trace state is carried in :class:`contextvars.ContextVar` objects, so
  concurrent validations (threads or asyncio tasks) are isolated from
  each other without relying on thread scheduling.
* Sensitive input values are never retained by the trace when the
  caller supplies a ``redact`` callable; redaction is applied at record
  time, before anything is stored.
* Schema node identities are structural (a tuple of child positions
  assigned deterministically at compile time), so identities such as
  the failed branches and the final choice of ``Any`` are stable across
  repeated runs of the same schema.
"""

# fmt: off
from __future__ import annotations

import collections.abc
import contextvars
import inspect
import re
import threading
import typing
from contextlib import contextmanager

from voluptuous import error as er

# fmt: on


class _NoValue:
    """Sentinel for "no value attached to this event/report"."""

    def __repr__(self) -> str:
        return 'NO_VALUE'

    def __bool__(self) -> bool:
        return False


NO_VALUE = _NoValue()

# Event kinds recorded in a ValidationTrace.
ENTER = 'enter'
EXIT = 'exit'
ERROR = 'error'
DEFAULT = 'default'
REMOVED = 'removed'

_EVENT_KINDS = frozenset({ENTER, EXIT, ERROR, DEFAULT, REMOVED})

# Compile-time registry: active only while a traced validator tree is
# being compiled.  Never set during normal validation.
_compile_registry: contextvars.ContextVar = contextvars.ContextVar(
    'voluptuous_trace_compile_registry', default=None
)

# Run-time tracer: active only inside Schema.validate_with_trace.
_active_tracer: contextvars.ContextVar = contextvars.ContextVar(
    'voluptuous_trace_active_tracer', default=None
)


class TraceEvent:
    """A single recorded step of a traced validation.

    :attr seq: Monotonic sequence number within its trace, starting at 0.
    :attr event: One of ``enter``, ``exit``, ``error``, ``default`` or
        ``removed``.
    :attr node_id: Stable structural identity of the schema node, a
        tuple of child positions from the schema root.
    :attr kind: Classification of the schema node (``mapping``,
        ``sequence``, ``coerce``, ``any``, ``all``, ``callable``, ...).
    :attr path: Path of the data being validated when the event was
        recorded, as a tuple.
    :attr value: The (possibly redacted) value relevant to the event:
        the node input for ``enter``, the node output for ``exit``, the
        generated value for ``default`` and the dropped value for
        ``removed``.  ``NO_VALUE`` when not applicable.
    :attr error: For ``error`` events, ``'<ExceptionClass>: <message>'``;
        ``None`` otherwise.
    :attr desc: Human readable, address-independent description of the
        schema node.
    """

    __slots__ = ('seq', 'event', 'node_id', 'kind', 'path', 'value', 'error', 'desc')

    def __init__(
        self,
        seq: int,
        event: str,
        node_id: typing.Tuple[int, ...],
        kind: str,
        path: typing.Tuple[typing.Hashable, ...],
        value: typing.Any = NO_VALUE,
        error: typing.Optional[str] = None,
        desc: typing.Optional[str] = None,
    ) -> None:
        assert event in _EVENT_KINDS, 'unknown trace event %r' % (event,)
        self.seq = seq
        self.event = event
        self.node_id = node_id
        self.kind = kind
        self.path = path
        self.value = value
        self.error = error
        self.desc = desc

    def __repr__(self) -> str:
        parts = [
            'seq=%r' % (self.seq,),
            'event=%r' % (self.event,),
            'node_id=%r' % (self.node_id,),
            'kind=%r' % (self.kind,),
            'path=%r' % (self.path,),
        ]
        if self.value is not NO_VALUE:
            parts.append('value=%r' % (self.value,))
        if self.error is not None:
            parts.append('error=%r' % (self.error,))
        if self.desc is not None:
            parts.append('desc=%r' % (self.desc,))
        return 'TraceEvent(%s)' % ', '.join(parts)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TraceEvent):
            return NotImplemented
        return all(
            getattr(self, field) == getattr(other, field) for field in self.__slots__
        )

    def __ne__(self, other) -> bool:
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq


class NodeInfo(typing.NamedTuple):
    """Static description of a compiled schema node."""

    node_id: typing.Tuple[int, ...]
    kind: str
    desc: str


class ValidationTrace:
    """Ordered record of the events of one traced validation pass.

    A fresh instance is created for every
    :meth:`voluptuous.Schema.validate_with_trace` call; nothing is
    shared between calls, so a failed call cannot leave cached events,
    counters or temporary state behind for a later one.
    """

    def __init__(
        self, redact: typing.Optional[typing.Callable[[typing.Any], typing.Any]] = None
    ) -> None:
        self._redact = redact
        self._events: typing.List[TraceEvent] = []
        self._finished = False

    def _sanitize(self, value):
        if value is NO_VALUE or self._redact is None:
            return value
        return self._redact(value)

    def _record(
        self,
        event: str,
        node: NodeInfo,
        path,
        value=NO_VALUE,
        error: typing.Optional[BaseException] = None,
    ) -> None:
        if self._finished:
            # A closed trace must never absorb new events; this protects
            # retry/atomicity guarantees.
            raise RuntimeError('cannot record into a finished ValidationTrace')
        error_text = None
        if error is not None:
            error_text = '%s: %s' % (type(error).__name__, error)
        self._events.append(
            TraceEvent(
                seq=len(self._events),
                event=event,
                node_id=node.node_id,
                kind=node.kind,
                path=tuple(path),
                value=self._sanitize(value),
                error=error_text,
                desc=node.desc,
            )
        )

    # Recording entry points used by the traced validator tree.  These
    # are internal: they are only reachable while this trace is the
    # active tracer of the current context.
    def node_enter(self, node: NodeInfo, path, data) -> None:
        self._record(ENTER, node, path, data)

    def node_exit(self, node: NodeInfo, path, result) -> None:
        self._record(EXIT, node, path, result)

    def node_error(self, node: NodeInfo, path, data, error: BaseException) -> None:
        self._record(ERROR, node, path, data, error=error)

    def record_default(self, node: NodeInfo, path, key, value) -> None:
        self._record(DEFAULT, node, list(path) + [key], value)

    def record_removed(self, node: NodeInfo, path, value) -> None:
        self._record(REMOVED, node, path, value)

    def finish(self) -> None:
        """Close the trace; further recording attempts raise."""
        self._finished = True

    @property
    def events(self) -> typing.Tuple[TraceEvent, ...]:
        """The recorded events, in recording order."""
        return tuple(self._events)

    def events_for(self, node_id: typing.Tuple[int, ...]) -> typing.List[TraceEvent]:
        """All events recorded against the given schema node."""
        return [event for event in self._events if event.node_id == node_id]

    def child_ids(
        self, node_id: typing.Tuple[int, ...]
    ) -> typing.List[typing.Tuple[int, ...]]:
        """Direct child node ids of ``node_id``, in first-seen order.

        For combinator nodes such as ``Any`` this yields the stable
        branch identities in the order the branches were exercised.
        """
        seen = []
        for event in self._events:
            if (
                len(event.node_id) == len(node_id) + 1
                and event.node_id[:-1] == node_id
                and event.node_id not in seen
            ):
                seen.append(event.node_id)
        return seen


class TraceReport:
    """Outcome of :meth:`voluptuous.Schema.validate_with_trace`.

    Exactly one of :attr:`value` and :attr:`error` is set:

    * on success ``value`` holds the validated data (identical to what
      ``Schema.__call__`` would have returned) and ``error`` is ``None``;
    * on failure ``value`` is ``NO_VALUE`` and ``error`` is the
      ``MultipleInvalid`` that ``Schema.__call__`` would have raised.
    """

    __slots__ = ('_value', '_error', '_trace')

    def __init__(
        self,
        value: typing.Any,
        error: typing.Optional[er.MultipleInvalid],
        trace: ValidationTrace,
    ) -> None:
        self._value = value
        self._error = error
        self._trace = trace

    @property
    def ok(self) -> bool:
        return self._error is None

    @property
    def value(self):
        return self._value

    @property
    def error(self) -> typing.Optional[er.MultipleInvalid]:
        return self._error

    @property
    def trace(self) -> ValidationTrace:
        return self._trace

    def __repr__(self) -> str:
        if self.ok:
            return 'TraceReport(ok=True, value=%r, events=%d)' % (
                self._value,
                len(self._trace.events),
            )
        return 'TraceReport(ok=False, error=%r, events=%d)' % (
            self._error,
            len(self._trace.events),
        )


def classify(schema) -> str:
    """Classify a schema node for trace labelling.

    Imports are lazy to avoid an import cycle with
    :mod:`voluptuous.schema_builder` and :mod:`voluptuous.validators`.
    """
    from voluptuous import schema_builder as sb
    from voluptuous import validators as v

    if schema is sb.Extra:
        return 'extra'
    if schema is sb.Self:
        return 'self'
    if isinstance(schema, v.Coerce):
        return 'coerce'
    if isinstance(schema, v.Any):
        return 'any'
    if isinstance(schema, v.All):
        return 'all'
    if isinstance(schema, v.Union):
        return 'union'
    if isinstance(schema, v._WithSubValidators):
        return type(schema).__name__.lower()
    if isinstance(schema, sb.Remove):
        return 'remove'
    if isinstance(schema, sb.Exclusive):
        return 'exclusive'
    if isinstance(schema, sb.Inclusive):
        return 'inclusive'
    if isinstance(schema, sb.Required):
        return 'required'
    if isinstance(schema, sb.Optional):
        return 'optional'
    if isinstance(schema, sb.Marker):
        return 'marker'
    if isinstance(schema, sb.Object):
        return 'object'
    if isinstance(schema, collections.abc.Mapping):
        return 'mapping'
    if isinstance(schema, (list, tuple)):
        return 'sequence'
    if isinstance(schema, (frozenset, set)):
        return 'set'
    if inspect.isclass(schema):
        return 'type'
    if callable(schema):
        return 'callable'
    return 'value'


_HEX_ADDRESS = re.compile(r'0x[0-9a-fA-F]+')


def describe(schema) -> str:
    """Return an address-independent, run-stable description of a node."""
    if inspect.isclass(schema):
        return schema.__name__
    if callable(schema):
        name = getattr(schema, '__name__', None)
        if name:
            return str(name)
        return type(schema).__name__
    try:
        text = repr(schema)
    except Exception:
        text = object.__repr__(schema)
    text = _HEX_ADDRESS.sub('0x...', text)
    if len(text) > 60:
        text = text[:57] + '...'
    return text


class _CompileRegistry:
    """Assigns stable structural node ids while a traced tree compiles.

    Node ids are tuples of child positions counted per parent in
    deterministic compile order; they never depend on hash values,
    memory addresses or map enumeration coincidences.
    """

    def __init__(self) -> None:
        self._stack: typing.List[list] = []
        self.nodes: typing.Dict[typing.Tuple[int, ...], NodeInfo] = {}

    @property
    def current(self) -> typing.Optional[NodeInfo]:
        if not self._stack:
            return None
        return self._stack[-1][0]

    @contextmanager
    def node(self, schema):
        parent = self.current
        if parent is None:
            node_id: typing.Tuple[int, ...] = ()
        else:
            node_id = parent.node_id + (self._stack[-1][1],)
            self._stack[-1][1] += 1
        info = NodeInfo(node_id=node_id, kind=classify(schema), desc=describe(schema))
        self.nodes[node_id] = info
        self._stack.append([info, 0])
        try:
            yield info
        finally:
            self._stack.pop()


def _wrap_compiled(node: NodeInfo, compiled: typing.Callable) -> typing.Callable:
    """Wrap a compiled validator so it reports to the active tracer.

    The wrapped validator executes ``compiled`` exactly once per call;
    when no tracer is active in the current context it is a transparent
    pass-through.
    """

    def traced_validate(path, data):
        tracer = _active_tracer.get(None)
        if tracer is None:
            return compiled(path, data)
        tracer.node_enter(node, path, data)
        try:
            result = compiled(path, data)
        except er.Invalid as e:
            tracer.node_error(node, path, data, e)
            raise
        except Exception as e:
            tracer.node_error(node, path, data, e)
            raise
        tracer.node_exit(node, path, result)
        return result

    return traced_validate


def record_default(node: NodeInfo, path, key, value) -> None:
    """Record a generated default, if a tracer is active."""
    tracer = _active_tracer.get(None)
    if tracer is not None:
        tracer.record_default(node, path, key, value)


def record_removed(node: NodeInfo, path, value) -> None:
    """Record a value dropped by a ``Remove`` marker, if tracing."""
    tracer = _active_tracer.get(None)
    if tracer is not None:
        tracer.record_removed(node, path, value)
