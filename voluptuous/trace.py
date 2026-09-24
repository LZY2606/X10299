"""Opt-in structured validation traces.

A :class:`Trace` collects a structured, ordered record of how an input value
traveled through a compiled :class:`~voluptuous.schema_builder.Schema`: which
schema nodes were entered and what they produced, where default values were
generated, values coerced, ``Any`` branches selected or rejected, keys
removed, extras handled or errors aggregated.

Tracing is strictly opt-in::

    >>> from voluptuous import Schema, Coerce
    >>> trace = Trace()
    >>> result = Schema(Coerce(int))('42', trace=trace)
    >>> result
    42
    >>> [(event.kind, event.flavor) for event in trace.events]
    [('node', 'coerce'), ('node', 'callable')]

Calling a schema without ``trace=`` takes the ordinary code path; no trace
state is touched. Traces are per-call objects isolated through a
:class:`contextvars.ContextVar`, so concurrent validations (threads or
asyncio tasks) can never share events, node stacks or counters.

Every recorded value (input, output, generated default, key or path
component) passes through the trace's ``redact`` callback *before* it is
stored. The default keeps values as-is; callers handling sensitive data can
supply a redactor so that raw secrets never enter the trace.

Failures are atomic with respect to trace state: a trace is activated for a
single schema call, and whether the call succeeds or raises, the events
captured up to the failure point remain available. After a failure the trace
is inert again and the very same input can be retried; activating the trace
resets it deterministically, so the retry yields the same event sequence.
"""

from __future__ import annotations

import contextvars
import dataclasses
import typing

#: A compiled schema node was entered. The event's ``detail`` mapping carries
#: ``"input"`` (the redacted incoming value) and, when the node returned
#: normally, ``"output"`` (the redacted outgoing value). A node that raised
#: carries an ``"error"`` record instead.
NODE = 'node'

#: A default value/factory was materialized for a missing mapping key.
DEFAULT = 'default'

#: One branch of an ``Any`` validator was tried. ``detail`` has ``"index"``
#: (the branch's stable declaration-order position), either ``"output"`` on
#: success or ``"error"`` on failure.
ANY_BRANCH = 'any-branch'

#: Emitted on the node that is the winning ``Any`` branch; ``detail`` has the
#: stable ``"index"`` of the selected branch.
ANY_SELECT = 'any-select'

#: One step of an ``All`` pipeline completed. ``detail`` has ``"index"`` and
#: the redacted ``"input"``/``"output"`` of that step.
ALL_STEP = 'all-step'

#: A mapping key matched a ``Remove`` marker. ``detail`` has the redacted
#: ``"key"`` and ``"path"`` (stringified, redacted data path).
REMOVE = 'remove'

#: A mapping/sequence element did not match any schema key/validator and was
#: handled by the schema's ``extra`` policy. ``detail`` has ``"action"`` of
#: ``"allow"``, ``"drop"`` or ``"prevent"`` plus the redacted ``"key"`` or
#: ``"index"``.
EXTRA = 'extra'

#: An ``Exclusive`` group was violated. ``detail`` has the redacted
#: ``"group"`` label and the conflicting ``"keys"``.
EXCLUSIVE = 'exclusive'

#: Several child errors were aggregated into one ``MultipleInvalid``.
#: ``detail`` has the redacted ``"path"`` and a tuple of ``"errors"``
#: records ``{"error": <exception class name>, "path": (str, ...)}``.
ERROR_GROUP = 'error-group'


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    """A single, immutable fact recorded during validation.

    :ivar seq: zero-based, per-call ordinal; defines the deterministic order.
    :ivar node: stable identity of the schema node, as a tuple of structural
        labels (independent of object identity, enumeration order of input
        mappings or thread scheduling).
    :ivar kind: one of the module-level event constants.
    :ivar flavor: what kind of schema node produced the event, e.g.
        ``"mapping"``, ``"sequence"``, ``"set"``, ``"default"`` (a key
        marker), ``"coerce"``, ``"callable"``, ``"type"``, ``"literal"``,
        ``"any"``, ``"all"`` or ``"marker"``.
    :ivar detail: structured, JSON-friendly payload. Values are exactly what
        the trace's ``redact`` callback returned; raw input values are never
        stored unless the callback returns them.
    """

    seq: int
    node: typing.Tuple[str, ...]
    kind: str
    flavor: str
    detail: typing.Mapping[str, typing.Any]

    def as_dict(self) -> typing.Mapping[str, typing.Any]:
        return dataclasses.asdict(self)


Redactor = typing.Callable[[typing.Any, typing.Tuple[str, ...]], typing.Any]


def _identity_redact(value, node):
    return value


_current: contextvars.ContextVar[typing.Optional['Recorder']] = contextvars.ContextVar(
    'voluptuous_trace', default=None
)


def current_recorder() -> typing.Optional['Recorder']:
    """Return the active per-call recorder, or ``None`` on the plain path.

    This is a narrow internal hook: production validators call it to ask
    whether a trace is in progress. The recording logic lives only here and
    in :class:`Recorder`; validators do not duplicate any validation
    algorithm, they merely announce facts about the single execution that is
    already happening.
    """
    return _current.get()


def node_label(schema: typing.Any) -> str:
    """Return a stable, human-readable structural label for a schema piece.

    Labels never use :func:`id`/memory addresses, so the same schema declared
    twice produces identical node identities across processes and machines.
    """
    if schema is None:
        return 'none'
    # Avoid importing the marker classes here (module layering).
    if isinstance(schema, bool):
        return 'literal:%r' % (schema,)
    if isinstance(schema, type):
        return 'type:%s' % schema.__name__
    flavor = type(schema).__name__
    # Coerce/Any/All/Match/etc. expose deterministic repr; cap the length so
    # pathological callables cannot blow up the identity.
    text = repr(schema)
    if len(text) > 80:
        text = text[:77] + '...'
    return '%s:%s' % (flavor, text)


@dataclasses.dataclass
class Recorder:
    """Per-call, per-``Trace`` recording surface.

    A fresh recorder is created by :meth:`Trace.activate` for every schema
    call and installed in a context variable, which guarantees isolation
    across threads and asyncio tasks even when the same :class:`Trace` object
    is reused for sequential retries.
    """

    trace: 'Trace'
    stack: typing.List[str] = dataclasses.field(default_factory=list)
    seq: int = 0

    @property
    def node(self) -> typing.Tuple[str, ...]:
        return tuple(self.stack)

    def _redact(self, value: typing.Any) -> typing.Any:
        return self.trace.redact(value, self.node)

    def _redact_path(self, path: typing.Iterable[typing.Any]) -> typing.Tuple[str, ...]:
        redacted = [self.trace.redact(component, self.node) for component in path]
        return tuple(str(component) for component in redacted)

    @staticmethod
    def _error_record(
        error: BaseException, path: typing.Iterable[typing.Any]
    ) -> typing.Mapping[str, typing.Any]:
        if isinstance(error, BaseException) and error.args:
            # Error messages may echo literals from the schema, never input,
            # but stay conservative and do not redact: callers only need the
            # stable error class and data path here.
            return {'error': type(error).__name__, 'path': tuple(map(str, path))}
        return {'error': type(error).__name__, 'path': tuple(map(str, path))}

    def _record_errors(
        self, errors: typing.Iterable[BaseException], path: typing.Iterable[typing.Any]
    ) -> None:
        records = tuple(self._error_record(e, path) for e in errors)
        self.emit(
            ERROR_GROUP,
            'aggregate',
            {'path': self._redact_path(path), 'errors': records},
        )

    def emit(self, kind: str, flavor: str, detail: typing.Mapping[str, typing.Any]):
        event = TraceEvent(
            seq=self.seq,
            node=self.node,
            kind=kind,
            flavor=flavor,
            detail=dict(detail),
        )
        self.trace._events.append(event)
        self.seq += 1
        return event

    def run_node(self, label, flavor, func, path, value):
        """Run ``func(path, value)`` once, recording an enter/exit fact.

        The wrapped validator is executed exactly one time; tracing never
        re-runs validation to gather information. On success the recorded
        output is ``func``'s return value; on failure the raised exception is
        recorded and re-raised as the very same object.
        """
        recorder = _current.get()
        if recorder is None or recorder.trace is not self.trace:
            return func(path, value)
        recorder.stack.append(label)
        detail = {'path': recorder._redact_path(path), 'input': recorder._redact(value)}
        try:
            result = func(path, value)
        except BaseException as error:
            if isinstance(error, Exception) and hasattr(error, 'errors'):
                detail['error'] = {
                    'error': type(error).__name__,
                    'path': recorder._redact_path(
                        getattr(error, 'path', path) or path
                    ),
                    'errors': tuple(
                        {
                            'error': type(child).__name__,
                            'path': recorder._redact_path(
                                getattr(child, 'path', path) or path
                            ),
                        }
                        for child in error.errors
                    ),
                }
            else:
                detail['error'] = recorder._error_record(
                    error, getattr(error, 'path', path)
                )
            recorder.emit(NODE, flavor, detail)
            recorder.stack.pop()
            raise
        detail['output'] = recorder._redact(result)
        recorder.emit(NODE, flavor, detail)
        recorder.stack.pop()
        return result

    # -- specialized facts -------------------------------------------------

    def record_default(self, flavor, key, path, value):
        self.emit(
            DEFAULT,
            flavor,
            {
                'key': self.trace.redact(key, self.node),
                'path': self._redact_path(path),
                'value': self.trace.redact(value, self.node),
            },
        )

    def record_any_branch(self, index, path, *, output=None, error=None):
        detail = {'index': index, 'path': self._redact_path(path)}
        if error is not None:
            detail['error'] = self._error_record(error, getattr(error, 'path', path))
        else:
            detail['output'] = self._redact(output)
        self.emit(ANY_BRANCH, 'any', detail)

    def record_any_select(self, index, path, output):
        self.emit(
            ANY_SELECT,
            'any',
            {
                'index': index,
                'path': self._redact_path(path),
                'output': self._redact(output),
            },
        )

    def record_all_step(self, index, path, value, result):
        self.emit(
            ALL_STEP,
            'all',
            {
                'index': index,
                'path': self._redact_path(path),
                'input': self._redact(value),
                'output': self._redact(result),
            },
        )

    def record_remove(self, key, path):
        self.emit(
            REMOVE,
            'remove',
            {
                'key': self.trace.redact(key, self.node),
                'path': self._redact_path(path),
            },
        )

    def record_extra(self, action, path, *, key=None, index=None):
        detail = {'action': action, 'path': self._redact_path(path)}
        if key is not None:
            detail['key'] = self.trace.redact(key, self.node)
        if index is not None:
            detail['index'] = index
        self.emit(EXTRA, 'extra', detail)

    def record_exclusive(self, group, keys, path):
        self.emit(
            EXCLUSIVE,
            'exclusive',
            {
                'group': self.trace.redact(group, self.node),
                'keys': tuple(
                    self.trace.redact(key, self.node) for key in keys
                ),
                'path': self._redact_path(path),
            },
        )

    def record_error_group(self, errors, path):
        self._record_errors(errors, path)


class Trace:
    """Collects structured trace events for one (or a sequence of retries).

    :param redact: callable ``(value, node) -> redacted_value`` invoked for
        every value before it is stored. The default returns the value
        unchanged. Returning a constant like ``"***"`` implements whole-trace
        redaction of sensitive raw inputs.

    Typical use::

        trace = Trace(redact=lambda value, node: '***' if node[-1:] ... )
        schema(data, trace=trace)
        trace.events
    """

    def __init__(self, redact: Redactor = _identity_redact) -> None:
        self.redact = redact
        self._events: typing.List[TraceEvent] = []
        self._active = False
        self.succeeded: typing.Optional[bool] = None

    @property
    def events(self) -> typing.Tuple[TraceEvent, ...]:
        """Immutable view of the recorded events, in deterministic order."""
        return tuple(self._events)

    def activate(self) -> '_ActivatedTrace':
        if self._active:
            raise RuntimeError(
                'Trace is already active in this context; use a fresh Trace '
                'for concurrent validations'
            )
        # Reset deterministically: a retry after a recoverable failure starts
        # from exactly the same empty state as the first attempt.
        self._events = []
        self.succeeded = None
        self._active = True
        recorder = Recorder(trace=self)
        token = _current.set(recorder)
        return _ActivatedTrace(self, token, recorder)

    def _deactivate(self, token, succeeded):
        _current.reset(token)
        self.succeeded = succeeded
        self._active = False


class _ActivatedTrace:
    def __init__(self, trace, token, recorder):
        self.trace = trace
        self._token = token
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.trace._deactivate(self._token, exc_type is None)
        return False
