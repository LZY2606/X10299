"""Opt-in structured validation traces for :class:`voluptuous.Schema`.

The tracing machinery is intentionally separated from the validation
algorithm: every traced run executes the *same* compiled closures as an
ordinary ``schema(data)`` call.  Tracing only adds observer wrappers around
those closures, so it never re-executes a validator and cannot change the
validated result or the raised exception.

The public surface consists of:

* :class:`Trace` -- per-run trace collector, enabled by passing
  ``trace=Trace()`` (or ``trace=True``) to ``Schema.__call__``.
* :func:`redact_keys` -- a small redaction helper for callers that want to
  mask values at sensitive paths.

Everything else in this module is an internal implementation detail shared
with the production compilation path in :mod:`voluptuous.schema_builder`.
"""
from __future__ import annotations

import contextvars
import dataclasses
import itertools
import threading
import typing

# fmt: off
from voluptuous.error import Invalid, MultipleInvalid  # noqa: F401
# fmt: on

#: Context variable holding the :class:`Trace` for the currently executing
#: traced run.  Context variables give per-thread / per-task isolation for
#: free, so concurrent traced runs never share events.
_current_trace: contextvars.ContextVar[typing.Optional['Trace']] = (
    contextvars.ContextVar('voluptuous_trace', default=None)
)

#: Event kinds produced while following a value through the schema tree.
ENTER = 'enter'
SUCCESS = 'success'
ERROR = 'error'
DEFAULT = 'default'
REMOVED = 'removed'
SELECT = 'select'


@dataclasses.dataclass(frozen=True)
class TraceNode:
    """A stable identifier for one compiled schema node.

    ``node_id`` is a structural id derived from the schema construction,
    independent of map enumeration order or thread scheduling: the same
    schema position receives the same id across runs, retries and threads.
    """

    node_id: str
    kind: str
    label: str
    origin: typing.Any = None

    def __repr__(self) -> str:
        return 'TraceNode(%r, kind=%r)' % (self.node_id, self.kind)


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    """One observation made while a value passed through a schema node."""

    seq: int
    kind: str
    node_id: str
    path: tuple
    value: typing.Any = dataclasses.field(default=None)
    has_value: bool = False
    detail: typing.Any = None
    error: typing.Optional[Invalid] = None

    def __repr__(self) -> str:
        return 'TraceEvent(seq=%d, kind=%r, node=%r, path=%r)' % (
            self.seq,
            self.kind,
            self.node_id,
            list(self.path),
        )


_REDACTED = '<redacted>'


def redact_keys(*secret_keys, replacement=_REDACTED):
    """Build a ``redact`` callable that masks values at sensitive keys.

    A node value is replaced with ``replacement`` whenever the final path
    component equals one of ``secret_keys`` (compared by equality).

    >>> from voluptuous.trace import redact_keys
    >>> redact = redact_keys('password')
    >>> redact(('user',), 'bob')
    'bob'
    >>> redact(('password',), 'hunter2')
    '<redacted>'
    """

    secrets = set(secret_keys)

    def _redact(path, value):
        if path and path[-1] in secrets:
            return replacement
        return value

    return _redact


class Trace(object):
    """Structured trace of a single ``Schema(...)`` validation run.

    Pass an instance to a schema call to opt in::

        trace = Trace()
        result = schema(data, trace=trace)

    On success the validated value is also available as ``trace.result``;
    on failure the raised :class:`~voluptuous.error.Invalid` is stored on
    ``trace.error``.  A :class:`Trace` is single-use: reuse the schema, not
    the trace, for retries.
    """

    def __init__(self) -> None:
        self._events: typing.List[TraceEvent] = []
        self._nodes: typing.Dict[str, TraceNode] = {}
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._started = False
        self._finished = False
        self.result = _UNSET
        self.error: typing.Optional[Invalid] = None

    # -- recording (internal, shared with the production compile path) -----

    def _register(self, node: TraceNode) -> None:
        existing = self._nodes.get(node.node_id)
        if existing is not None and existing.origin is not node.origin:
            # Same structural position must always map to the same node.
            raise AssertionError(
                'trace node id collision: %r reused for %r and %r'
                % (node.node_id, existing.origin, node.origin)
            )
        self._nodes[node.node_id] = node

    def _emit(
        self,
        kind: str,
        node_id: str,
        path: typing.Sequence,
        value=_UNSET,
        detail=None,
        error=None,
    ) -> TraceEvent:
        event = TraceEvent(
            seq=next(self._counter),
            kind=kind,
            node_id=node_id,
            path=tuple(path),
            value=value,
            has_value=value is not _UNSET,
            detail=detail,
            error=error,
        )
        with self._lock:
            self._events.append(event)
        return event

    @contextlib_contextmanager
    def _run_guard(self):
        if self._started:
            raise RuntimeError('a Trace instance can only be used for a single run')
        self._started = True
        token = _current_trace.set(self)
        try:
            yield
        finally:
            _current_trace.reset(token)
            self._finished = True

    def _succeed(self, result) -> None:
        self.result = result

    def _fail(self, error) -> None:
        self.error = error

    # -- public read-only views --------------------------------------------

    @property
    def events(self) -> typing.Tuple[TraceEvent, ...]:
        """All events, in execution order."""
        return tuple(self._events)

    @property
    def nodes(self) -> typing.Mapping[str, TraceNode]:
        """Structural nodes encountered, keyed by stable ``node_id``."""
        return types_MappingProxyType(self._nodes)

    def node(self, node_id: str) -> TraceNode:
        return self._nodes[node_id]

    def events_for(self, node_id: str) -> typing.Tuple[TraceEvent, ...]:
        return tuple(e for e in self._events if e.node_id == node_id)

    def value_events(self) -> typing.Tuple[TraceEvent, ...]:
        """Events carrying an input or produced value, in execution order."""
        return tuple(e for e in self._events if e.has_value)

    def provenance(self, path) -> typing.Tuple[TraceEvent, ...]:
        """Value-carrying events at ``path`` (a tuple), in execution order."""
        path = tuple(path)
        return tuple(e for e in self._events if e.has_value and e.path == path)

    def errors(self) -> typing.Tuple[Invalid, ...]:
        """All recorded errors, in the order they were raised."""
        return tuple(e.error for e in self._events if e.error is not None)

    def redacted_copy(self, redact=_default_redact):
        """Return a value-scrubbed copy; the trace itself is left untouched.

        ``redact(path, value)`` returns either the original value or a
        replacement.  Raw values in events are the only masked data: error
        objects are passed through untouched so callers remain in control of
        what their validators put in messages.
        """
        if not callable(redact):
            raise TypeError('redact must be callable')
        copy = object.__new__(Trace)
        copy._events = []
        copy._nodes = self._nodes
        copy._counter = itertools.count()
        copy._lock = threading.Lock()
        copy._started = self._started
        copy._finished = self._finished
        copy.result = self.result
        copy.error = self.error
        for event in self._events:
            if event.has_value:
                event = dataclasses.replace(
                    event, value=redact(event.path, event.value)
                )
            copy._events.append(event)
        return copy


class _Unset(object):
    __slots__ = ()

    def __repr__(self):
        return '<unset>'


_UNSET = _Unset()


def _default_redact(path, value):  # pragma: no cover - only used as a guard
    raise TypeError(
        'Trace.redacted_copy() requires a redact(path, value) callable'
    )


def contextlib_contextmanager(func):
    import contextlib

    return contextlib.contextmanager(func)


def types_MappingProxyType(mapping):
    import types

    return types.MappingProxyType(mapping)


# ---------------------------------------------------------------------------
# Internal compilation support
#
# A single _TraceSupport is created per traced Schema build.  It is shared by
# every compiled closure of that schema tree, but it holds *no* per-run
# state: per-run state lives on Trace (passed via a context variable).
# ---------------------------------------------------------------------------


class _Scope(object):
    """Prefix stack used while a traced schema tree is compiled."""

    __slots__ = ('segments',)

    def __init__(self, segments=()):
        self.segments = tuple(segments)

    def child(self, segment):
        return _Scope(self.segments + (segment,))


def _safe_label(obj) -> str:
    try:
        text = repr(obj)
    except Exception:
        text = object.__repr__(obj)
    return text.replace('}', '\\}')[:80]


class _TraceSupport(object):
    def __init__(self, root, scope=None):
        self.root = root
        self.scope = scope or _Scope()

    def child(self, segment):
        return _TraceSupport(self.root, self.scope.child(segment))

    def node_id(self, segment):
        segments = self.scope.segments + (segment,)
        return '{' + '}{'.join(segments) + '}'

    def register(self, kind, segment, label, origin, runtime=False):
        node_id = self.node_id(segment)
        self.root.register_static(
            TraceNode(node_id=node_id, kind=kind, label=label, origin=origin),
            runtime=runtime,
        )
        return node_id

    # The methods below are invoked from instrumented closures at run time.

    @staticmethod
    def current():
        return _current_trace.get()

    @staticmethod
    def emit(kind, node_id, path, value=_UNSET, detail=None, error=None):
        trace = _current_trace.get()
        if trace is not None:
            trace._emit(kind, node_id, path, value, detail=detail, error=error)

    @staticmethod
    def wrap(node_id, func, on_return=None):
        """Instrument one compiled validator closure.

        The wrapped closure calls ``func`` exactly once for every invocation
        and otherwise behaves identically; ``on_return`` is an observer used
        to derive :data:`REMOVED`/:data:`SELECT` events without re-running
        anything.
        """

        def traced(path, value):
            _TraceSupport.emit(ENTER, node_id, path, value)
            try:
                result = func(path, value)
            except Invalid as error:
                _TraceSupport.emit(
                    ERROR, node_id, path, error=error, detail=_error_detail(error)
                )
                raise
            if on_return is not None:
                on_return(path, value, result)
            _TraceSupport.emit(SUCCESS, node_id, path, result)
            return result

        traced._trace_node_id = node_id
        return traced


def _error_detail(error):
    if isinstance(error, MultipleInvalid):
        return [str(sub) for sub in error.errors]
    return None


class _TraceRoot(object):
    """Build-time registry shared by all nodes of one traced Schema tree."""

    def __init__(self):
        self.static_nodes = {}
        self.runtime_seq = itertools.count()
        self._lock = threading.Lock()

    def register_static(self, node, runtime=False):
        with self._lock:
            existing = self.static_nodes.get(node.node_id)
            if existing is not None:
                if existing.kind != node.kind or existing.label != node.label:
                    raise AssertionError(
                        'trace node id collision at %r' % (node.node_id,)
                    )
                return
            self.static_nodes[node.node_id] = node

    def runtime_segment(self):
        return 'runtime:%d' % next(self.runtime_seq)

    def bind(self, trace):
        with self._lock:
            for node in self.static_nodes.values():
                trace._register(node)
