"""Opt-in structured validation trace.

This module adds an *optional* provenance layer on top of the regular
validation machinery.  It is only active when a caller explicitly attaches a
:class:`ValidationTrace` to a validation call::

    >>> from voluptuous import Schema, Coerce
    >>> from voluptuous.trace import ValidationTrace
    >>> trace = ValidationTrace()
    >>> schema = Schema(Coerce(int))
    >>> schema('42', trace=trace)
    42
    >>> events = trace.events
    >>> events[0].kind == 'enter'
    True
    >>> events[-1].kind == 'result'
    True

When tracing is not requested the library behaves exactly as before: the
production validators share a single code path with the traced one; the trace
arguments are simply threaded through the *same* compiled closures and default
to ``None`` (so the fast path is a single ``is None`` check per node).

Design rules enforced by this module:

* Validators are never executed a second time to record provenance; every
  event is emitted from inside the real validation call.
* No mutable trace state ever lives on a schema or on a validator object.
  Traces are per-call, isolated by construction and additionally bound to a
  ``contextvars.ContextVar`` so re-entrant validation (nested ``Schema`` and
  ``Marker`` calls) joins the same trace without sharing across threads.
* Raw values are handed to a caller-supplied ``redact`` callable *before*
  being stored, so sensitive inputs never have to appear in a trace.
* Nodes are identified by stable content paths
  (``['map:<key>', 'any#1', ...]``), never by object identity or map
  enumeration accident, so repeated runs produce identical identities.
"""

from __future__ import annotations

import contextvars
import itertools
import threading
import typing

# fmt: on


# Event kind constants ------------------------------------------------------

ENTER = 'enter'
EXIT = 'exit'
FAILURE = 'failure'
RESULT = 'result'
DEFAULT = 'default'
COERCE = 'coerce'
BRANCH_FAILURE = 'branch_failure'
BRANCH_SELECTED = 'branch_selected'
COMBINE = 'combine'
REMOVE = 'remove'
AGGREGATE = 'aggregate'


class NodeRef(typing.NamedTuple):
    """Stable, content based identity of a schema node.

    ``path`` is a tuple of short, human readable segments locating the node
    inside the compiled schema tree (``map:'k'``, ``seq#0``, ``any#1`` ...).
    ``kind`` names the structural kind of the node (``mapping``, ``sequence``,
    ``any``, ``all``, ``coerce``, ``callable`` ...).  Two validations of the
    same schema produce equal :class:`NodeRef` values for the same node even
    across processes, threads and independent compilations.
    """

    path: typing.Tuple[str, ...]
    kind: str

    def __str__(self) -> str:
        return '%s{%s}' % (self.kind, ' -> '.join(self.path) or '<root>')


class TraceEvent(typing.NamedTuple):
    """A single recorded validation event, in emission order."""

    seq: int
    kind: str
    node: typing.Optional[NodeRef]
    path: typing.Tuple[typing.Hashable, ...]
    detail: typing.Mapping[str, typing.Any]

    def __repr__(self) -> str:
        location = '[%s]' % ']['.join(repr(p) for p in self.path) if self.path else ''
        return 'TraceEvent(#%d %s %s%s %r)' % (
            self.seq,
            self.kind,
            self.node,
            location,
            dict(self.detail),
        )


class ValidationTrace(object):
    """Collector for a single structured validation run.

    :param redact: optional callable applied to every raw input value and
        every generated value before it is stored in an event.  Returning for
        example a constant string keeps secrets out of the trace while the
        real values keep flowing through validation untouched.
    """

    __slots__ = (
        '_events',
        '_seq',
        '_lock',
        '_redact',
        '_prefix',
        '_frozen',
        'outcome',
        'error',
    )

    def __init__(
        self,
        redact: typing.Optional[typing.Callable[[typing.Any], typing.Any]] = None,
    ) -> None:
        self._events: typing.List[TraceEvent] = []
        self._seq = itertools.count()
        self._lock = threading.RLock()
        self._redact = redact
        self._prefix: typing.List[str] = []
        self._frozen = False
        self.outcome: typing.Optional[str] = None
        self.error: typing.Optional[BaseException] = None

    # -- introspection -----------------------------------------------------

    @property
    def events(self) -> typing.Tuple[TraceEvent, ...]:
        """All recorded events, in deterministic emission order."""
        with self._lock:
            return tuple(self._events)

    @property
    def ok(self) -> bool:
        """True iff the validation this trace belongs to completed."""
        return self.outcome == 'ok'

    def nodes(self) -> typing.Tuple[NodeRef, ...]:
        """Every node that was entered, without duplicates, in first order."""
        seen = {}
        for event in self._events:
            if event.node is not None and event.node not in seen:
                seen[event.node] = True
        return tuple(seen)

    def events_for(self, node: NodeRef) -> typing.Tuple[TraceEvent, ...]:
        return tuple(e for e in self._events if e.node == node)

    def find(
        self, kind: str, node: typing.Optional[NodeRef] = None
    ) -> typing.Tuple[TraceEvent, ...]:
        return tuple(
            e
            for e in self._events
            if e.kind == kind and (node is None or e.node == node)
        )

    # -- value handling ----------------------------------------------------

    def _value(self, value: typing.Any) -> typing.Any:
        if self._redact is None:
            return value
        try:
            return self._redact(value)
        except Exception:
            return '<redactor-failed>'

    # -- recording (used by the production validation path) ----------------

    def _emit(
        self,
        kind: str,
        node: typing.Optional[NodeRef],
        path: typing.Sequence[typing.Hashable],
        detail: typing.Mapping[str, typing.Any],
    ) -> None:
        with self._lock:
            if self._frozen:
                return
            event = TraceEvent(
                next(self._seq), kind, node, tuple(path), dict(detail)
            )
            self._events.append(event)

    def _node_ref(self, node_path, kind: str) -> NodeRef:
        if isinstance(node_path, NodeRef):
            return node_path
        segments = tuple(self._prefix) + tuple(node_path)
        return NodeRef(segments, kind)

    @typing.overload
    def scope(self, segment: str):
        ...

    def scope(self, segment: typing.Optional[str]):
        """Context manager / decorator pushing a schema-tree prefix.

        Used by re-entrant boundaries (a compiled callable invoking a nested
        :class:`~voluptuous.schema_builder.Schema` or
        :class:`~voluptuous.schema_builder.Marker`) so inner nodes report
        absolute, stable node paths instead of restarting at the root.
        """
        trace = self

        class _Scope:
            def __enter__(self):
                if segment is not None:
                    trace._prefix.append(segment)
                return trace

            def __exit__(self, exc_type, exc, tb):
                if segment is not None and trace._prefix:
                    trace._prefix.pop()
                return False

        return _Scope()

    # Typed helpers. They are deliberately tiny: all of them are called from
    # the real validation code, never from a duplicated test-only algorithm.

    def enter(self, node_path, kind, path, value):
        node = self._node_ref(node_path, kind)
        self._emit(ENTER, node, path, {'input': self._value(value)})
        return node

    def exit_(self, node, path, value):
        self._emit(EXIT, node, path, {'output': self._value(value)})

    def failure(self, node, path, error, **extra):
        detail = {'error_type': type(error).__name__, 'error': str(error)}
        detail.update(extra)
        self._emit(FAILURE, node, path, detail)

    def result(self, path, value):
        self._emit(RESULT, None, path, {'output': self._value(value)})

    def default_event(self, node, path, value):
        self._emit(DEFAULT, node, path, {'generated': self._value(value)})

    def coerce(self, node, path, before, after, type_name):
        self._emit(
            COERCE,
            node,
            path,
            {
                'input': self._value(before),
                'output': self._value(after),
                'to': type_name,
            },
        )

    def branch_failure(self, node, path, index, error):
        self._emit(
            BRANCH_FAILURE,
            node,
            path,
            {
                'branch': index,
                'error_type': type(error).__name__,
                'error': str(error),
            },
        )

    def branch_selected(self, node, path, index, value):
        self._emit(
            BRANCH_SELECTED,
            node,
            path,
            {'branch': index, 'output': self._value(value)},
        )

    def combine(self, node, path, index, value):
        self._emit(
            COMBINE,
            node,
            path,
            {'step': index, 'output': self._value(value)},
        )

    def remove(self, node, path, value):
        self._emit(
            REMOVE, node, path, {'input': self._value(value)}
        )

    def aggregate(self, node, path, errors):
        self._emit(
            AGGREGATE,
            node,
            path,
            {
                'count': len(errors),
                'errors': [
                    {
                        'error_type': type(e).__name__,
                        'error': str(e),
                        'path': tuple(getattr(e, 'path', ()) or ()),
                    }
                    for e in errors
                ],
            },
        )

    def finish_ok(self, path, value):
        with self._lock:
            if self._frozen:
                return
            self.result(path, value)
            self.outcome = 'ok'
            self._frozen = True

    def finish_error(self, path, error):
        with self._lock:
            if self._frozen:
                return
            errors = getattr(error, 'errors', None)
            if errors:
                self.aggregate(None, path, errors)
            else:
                self.failure(None, path, error)
            self.outcome = 'error'
            self.error = error
            self._frozen = True


# Active trace binding ------------------------------------------------------
#
# The context variable holds the trace for *re-entrant* validation only: when
# a compiled callable happens to invoke a freshly built ``Schema`` or a
# ``Marker`` (which itself compiles a private ``Schema``), that nested call
# joins the ambient trace instead of silently dropping provenance. Top level
# calls are driven by an explicit ``trace=`` argument, so the context var is
# never used to smuggle state between threads and is reset in a ``finally``.

_active_trace: contextvars.ContextVar[typing.Optional[ValidationTrace]] = (
    contextvars.ContextVar('voluptuous_active_trace', default=None)
)


def bind_trace(trace: typing.Optional[ValidationTrace]):
    """Bind ``trace`` for the duration of a (possibly nested) validation."""
    token = _active_trace.set(trace)
    return token


def reset_trace(token) -> None:
    _active_trace.reset(token)


def active_trace() -> typing.Optional[ValidationTrace]:
    return _active_trace.get()
