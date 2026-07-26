"""Runtime bridge between instrumented operations and ``debrix.control.v1``."""

from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from debrix.control import (
    SDK_VERSION,
    ControlDecision,
    ControlInvoke,
    ControlOperation,
    ControlRequest,
    ControlReturn,
    ControlUnmanaged,
    DebrixControlProtocolError,
    resolve_control,
    resolve_control_async,
)
from debrix.mocks import MockToolError, _json_safe
from debrix.semconv import Attr
from debrix.span import DebrixSpan, _normalize_messages

_IMMUTABLE_RECEIVERS = frozenset({"self", "cls"})


def _json_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return "array"
    raise ValueError("input values must be losslessly JSON-compatible")


def _python_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _json_input(value: Any) -> Any:
    kind = _json_kind(value)
    if kind == "object":
        assert isinstance(value, Mapping)
        if any(not isinstance(key, str) for key in value):
            raise ValueError("input object keys must be strings")
        return {key: _json_input(item) for key, item in value.items()}
    if kind == "array":
        assert isinstance(value, Sequence)
        return [_json_input(item) for item in value]
    return value


def _schema_errors(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "input",
) -> list[str]:
    allowed = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
    }
    unsupported = sorted(set(schema) - allowed)
    if unsupported:
        return [f"{path} schema contains unsupported fields: {', '.join(unsupported)}"]
    errors: list[str] = []
    expected_type = schema.get("type")
    type_matches = {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, Mapping),
    }
    if expected_type is not None:
        if expected_type not in type_matches:
            errors.append(f"{path} schema type is unsupported")
        elif not type_matches[expected_type]:
            errors.append(f"{path} must be JSON Schema {expected_type}")
            return errors
    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            errors.append(f"{path} schema enum must be a non-empty array")
        elif value not in enum:
            errors.append(f"{path} must match the captured enum")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            errors.append(f"{path} schema properties must be an object")
            return errors
        required = schema.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) for name in required)
        ):
            errors.append(f"{path} schema required must be a string array")
            return errors
        for name in required:
            if name not in value:
                errors.append(f"{path}.{name} is required")
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, bool):
            errors.append(
                f"{path} schema additionalProperties must be a boolean"
            )
        elif not additional:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}.{name} is not allowed by the schema")
        for name, property_schema in properties.items():
            if not isinstance(name, str) or not isinstance(property_schema, Mapping):
                errors.append(f"{path} schema properties are invalid")
                continue
            if name in value:
                errors.extend(
                    _schema_errors(
                        value[name],
                        property_schema,
                        path=f"{path}.{name}",
                    )
                )
    if isinstance(value, list) and "items" in schema:
        items = schema["items"]
        if not isinstance(items, Mapping):
            errors.append(f"{path} schema items must be an object")
        else:
            for index, item in enumerate(value):
                errors.extend(
                    _schema_errors(item, items, path=f"{path}[{index}]")
                )
    return errors


def _shape_errors(
    recorded: Any,
    candidate: Any,
    *,
    path: str,
) -> list[str]:
    try:
        expected_kind = _json_kind(recorded)
        actual_kind = _json_kind(candidate)
    except ValueError:
        return [f"{path} must remain losslessly JSON-compatible"]
    if actual_kind != expected_kind:
        return [f"{path} must remain JSON {expected_kind}"]
    errors: list[str] = []
    if isinstance(recorded, Mapping):
        assert isinstance(candidate, Mapping)
        if set(candidate) != set(recorded):
            return [f"{path} keys must exactly match recorded"]
        for name, value in recorded.items():
            errors.extend(
                _shape_errors(
                    value,
                    candidate[name],
                    path=f"{path}.{name}",
                )
            )
    elif isinstance(recorded, list):
        assert isinstance(candidate, list)
        if len(candidate) != len(recorded):
            return [f"{path} entry count must exactly match recorded"]
        for index, value in enumerate(recorded):
            errors.extend(
                _shape_errors(
                    value,
                    candidate[index],
                    path=f"{path}[{index}]",
                )
            )
    return errors


@dataclass(frozen=True)
class CapturedLlmMessages:
    """A lossless role-preserving view over one provider message list."""

    recorded_input: dict[str, Any]
    descriptor: dict[str, Any]

    def reconstruct(self, edited: Mapping[str, Any]) -> list[dict[str, Any]]:
        if set(edited) != {"messages"}:
            raise ValueError("message input keys must exactly match recorded")
        candidate = edited.get("messages")
        recorded = self.recorded_input["messages"]
        errors = _shape_errors(recorded, candidate, path="messages")
        if errors:
            raise ValueError(errors[0])
        assert isinstance(candidate, list)
        for index, (before, after) in enumerate(zip(recorded, candidate, strict=True)):
            assert isinstance(before, Mapping)
            assert isinstance(after, Mapping)
            if after.get("role") != before.get("role"):
                raise ValueError(
                    f"messages[{index}].role must remain {before.get('role')}"
                )
        return [dict(message) for message in candidate]


def capture_llm_messages(
    messages: Sequence[Mapping[str, Any]],
) -> CapturedLlmMessages | None:
    """Capture complete JSON-safe messages and their additive descriptor."""

    try:
        normalized = _normalize_messages(messages)
    except (TypeError, ValueError):
        return None
    recorded_input = {"messages": normalized}
    return CapturedLlmMessages(
        recorded_input=recorded_input,
        descriptor={
            "schemaVersion": 1,
            "operationKind": "llm",
            "jsonKind": "object",
            "parameters": [
                {
                    "name": "messages",
                    "kind": "sequence",
                    "required": True,
                    "hasDefault": False,
                    "editable": True,
                    "jsonKind": "array",
                    "pythonType": "builtins.list",
                }
            ],
        },
    )


def validate_model_output(
    recorded: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete edited model response against its recorded shape."""

    try:
        normalized_recorded = _json_input(recorded)
        normalized_candidate = _json_input(candidate)
    except ValueError as error:
        raise ValueError(
            "model output must be losslessly JSON-compatible"
        ) from error
    assert isinstance(normalized_recorded, dict)
    assert isinstance(normalized_candidate, dict)
    if not isinstance(normalized_recorded.get("content"), str):
        raise ValueError("recorded model output.content must be a string")
    if not isinstance(normalized_candidate.get("content"), str):
        raise ValueError("output.content must remain JSON string")
    errors = _shape_errors(
        normalized_recorded,
        normalized_candidate,
        path="output",
    )
    if errors:
        raise ValueError(errors[0])
    return normalized_candidate


@dataclass(frozen=True)
class CapturedBoundCall:
    """A lossless editable view over one Python ``inspect.BoundArguments``."""

    signature: inspect.Signature
    original_arguments: dict[str, Any]
    recorded_input: dict[str, Any]
    descriptor: dict[str, Any]

    def reconstruct(self, edited: Mapping[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        editable = {
            parameter["name"]
            for parameter in self.descriptor["parameters"]
            if parameter["editable"]
        }
        if set(edited) != editable:
            raise ValueError("input keys must exactly match editable parameters")
        schema = self.descriptor.get("jsonSchema")
        if isinstance(schema, Mapping):
            schema_errors = _schema_errors(edited, schema)
            if schema_errors:
                raise ValueError(schema_errors[0])

        arguments = dict(self.original_arguments)
        parameters = self.signature.parameters
        for name in editable:
            parameter = parameters[name]
            value = edited[name]
            expected = next(
                item["jsonKind"]
                for item in self.descriptor["parameters"]
                if item["name"] == name
            )
            try:
                actual = _json_kind(value)
            except ValueError as error:
                raise ValueError(
                    f"input parameter {name} must remain JSON {expected}"
                ) from error
            if actual != expected:
                raise ValueError(
                    f"input parameter {name} must remain JSON {expected}"
                )
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                arguments[name] = tuple(value)
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                arguments[name] = dict(value)
            elif isinstance(arguments[name], tuple) and isinstance(value, list):
                arguments[name] = tuple(value)
            else:
                arguments[name] = value

        rebound = inspect.BoundArguments(self.signature, arguments)
        return rebound.args, rebound.kwargs


def capture_bound_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    json_schema: Mapping[str, Any] | None = None,
) -> CapturedBoundCall | None:
    """Capture a call only when every editable value is losslessly JSON-safe."""

    try:
        signature = inspect.signature(fn)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        recorded_input = {
            name: _json_input(value)
            for name, value in bound.arguments.items()
            if name not in _IMMUTABLE_RECEIVERS
        }
    except (TypeError, ValueError):
        return None

    descriptor_parameters = []
    for name, parameter in signature.parameters.items():
        if name not in bound.arguments:
            continue
        value = bound.arguments[name]
        editable = name not in _IMMUTABLE_RECEIVERS
        descriptor_parameters.append(
            {
                "name": name,
                "kind": parameter.kind.name.lower(),
                "required": parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                },
                "hasDefault": parameter.default is not inspect.Parameter.empty,
                "editable": editable,
                "jsonKind": _json_kind(value) if editable else None,
                "pythonType": _python_type(value),
            }
        )

    if json_schema is not None and _schema_errors(recorded_input, json_schema):
        return None
    descriptor = {
        "schemaVersion": 1,
        "operationKind": "tool",
        "jsonKind": "object",
        "parameters": descriptor_parameters,
    }
    if json_schema is not None:
        descriptor["jsonSchema"] = _json_input(json_schema)
    return CapturedBoundCall(
        signature=signature,
        original_arguments=dict(bound.arguments),
        recorded_input=recorded_input,
        descriptor=descriptor,
    )


def capture_mapping_input(
    value: Mapping[str, Any],
    *,
    operation_kind: str,
    json_schema: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Capture a deterministic editable descriptor for an object-style call."""

    try:
        recorded = _json_input(value)
    except ValueError:
        return None
    assert isinstance(recorded, dict)
    parameters = [
        {
            "name": name,
            "kind": "mapping_key",
            "required": True,
            "hasDefault": False,
            "editable": True,
            "jsonKind": _json_kind(value[name]),
            "pythonType": _python_type(value[name]),
        }
        for name in sorted(value)
    ]
    if json_schema is not None and _schema_errors(recorded, json_schema):
        return None
    descriptor = {
        "schemaVersion": 1,
        "operationKind": operation_kind,
        "jsonKind": "object",
        "parameters": parameters,
    }
    if json_schema is not None:
        descriptor["jsonSchema"] = _json_input(json_schema)
    return (
        recorded,
        descriptor,
    )


@dataclass(frozen=True)
class _LiveExecution:
    trace_id: str
    branch_id: str
    attempt_id: str


_LIVE_EXECUTION: ContextVar[_LiveExecution | None] = ContextVar(
    "debrix.control.live_execution",
    default=None,
)


def _request(
    span: DebrixSpan,
    *,
    operation_kind: str,
    operation_name: str,
    operation_server: str | None,
    agent_scope: str | None,
    sequence_index: int,
    input_value: Any,
    input_descriptor: Mapping[str, Any] | None,
    execution_model: str,
    capabilities: tuple[str, ...] | None = None,
) -> ControlRequest:
    return ControlRequest(
        request_id=f"control_request_{uuid.uuid4().hex}",
        trace_id=span.trace_id_hex,
        runtime_span_id=span.span_id_hex,
        parent_runtime_span_id=span.parent_span_id_hex,
        sequence_index=sequence_index,
        operation=ControlOperation(
            kind=operation_kind,
            name=operation_name,
            server=operation_server,
            agent_scope=agent_scope,
        ),
        input_value=_json_safe(input_value),
        capabilities=capabilities
        or (
            ("input", "result", "error")
            if input_descriptor is not None
            else ("result", "error")
        ),
        execution_model=execution_model,
        sdk_version=SDK_VERSION,
        input_descriptor=input_descriptor,
    )


def resolve_runtime_control(
    span: DebrixSpan,
    *,
    operation_kind: str,
    operation_name: str,
    operation_server: str | None,
    agent_scope: str | None,
    sequence_index: int,
    input_value: Any,
    input_descriptor: Mapping[str, Any] | None = None,
    capabilities: tuple[str, ...] | None = None,
    endpoint: str | None = None,
) -> ControlDecision | ControlUnmanaged:
    """Probe and, when claimed, resolve one synchronous operation boundary."""
    return resolve_control(
        _request(
            span,
            operation_kind=operation_kind,
            operation_name=operation_name,
            operation_server=operation_server,
            agent_scope=agent_scope,
            sequence_index=sequence_index,
            input_value=input_value,
            input_descriptor=input_descriptor,
            execution_model="sync",
            capabilities=capabilities,
        ),
        endpoint=endpoint,
    )


async def resolve_runtime_control_async(
    span: DebrixSpan,
    *,
    operation_kind: str,
    operation_name: str,
    operation_server: str | None,
    agent_scope: str | None,
    sequence_index: int,
    input_value: Any,
    input_descriptor: Mapping[str, Any] | None = None,
    capabilities: tuple[str, ...] | None = None,
    endpoint: str | None = None,
) -> ControlDecision | ControlUnmanaged:
    """Probe and resolve without blocking the caller's asyncio event loop."""
    return await resolve_control_async(
        _request(
            span,
            operation_kind=operation_kind,
            operation_name=operation_name,
            operation_server=operation_server,
            agent_scope=agent_scope,
            sequence_index=sequence_index,
            input_value=input_value,
            input_descriptor=input_descriptor,
            execution_model="asyncio",
            capabilities=capabilities,
        ),
        endpoint=endpoint,
    )


def mark_live_span(span: DebrixSpan) -> None:
    """Mark a later operation as Live when it belongs to the bound trace."""
    execution = _LIVE_EXECUTION.get()
    if execution is None or execution.trace_id != span.trace_id_hex:
        return
    span.set_attribute(Attr.CONTROL_BRANCH_ID, execution.branch_id)
    span.set_attribute(Attr.CONTROL_ATTEMPT_ID, execution.attempt_id)
    span.set_attribute(Attr.CONTROL_PROVENANCE, "live")


def is_live_control_trace(span: DebrixSpan) -> bool:
    """Return whether this span belongs to an execution after its first edit."""

    execution = _LIVE_EXECUTION.get()
    return execution is not None and execution.trace_id == span.trace_id_hex


def apply_runtime_invoke(
    span: DebrixSpan,
    decision: ControlInvoke,
    captured: CapturedBoundCall | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Validate one managed Tool input and reconstruct its real Python call."""

    if captured is None:
        raise DebrixControlProtocolError(
            "input control requires a lossless bound Python call"
        )
    if not decision.capture_live_result:
        raise DebrixControlProtocolError(
            "input control must capture the live operation result"
        )
    if decision.input.provenance not in {"recorded", "edited", "live"}:
        raise DebrixControlProtocolError(
            "input control provenance must be recorded, edited, or live"
        )
    if not isinstance(decision.input.value, Mapping):
        raise DebrixControlProtocolError("controlled input must be a JSON object")
    if (
        decision.input.provenance in {"recorded", "live"}
        and decision.input.value != captured.recorded_input
    ):
        raise DebrixControlProtocolError(
            "Unchanged input must use the exact runtime value"
        )
    try:
        args, kwargs = captured.reconstruct(decision.input.value)
    except ValueError as error:
        raise DebrixControlProtocolError(str(error)) from error

    span.set_attribute(Attr.CONTROL_BRANCH_ID, decision.branch_id)
    span.set_attribute(Attr.CONTROL_ATTEMPT_ID, decision.attempt_id)
    span.set_attribute(Attr.CONTROL_OCCURRENCE_ID, decision.occurrence_id)
    span.set_attribute(Attr.CONTROL_PROVENANCE, decision.input.provenance)
    span.set_attribute(Attr.CONTROL_INPUT_PROVENANCE, decision.input.provenance)
    span.set_attribute(Attr.CONTROL_RESULT_PROVENANCE, "live")
    span.set_attribute(
        Attr.REPLAY_INPUT,
        json.dumps(_json_safe(decision.input.value), ensure_ascii=False),
    )
    if decision.input.provenance == "edited":
        _LIVE_EXECUTION.set(
            _LiveExecution(
                trace_id=span.trace_id_hex,
                branch_id=decision.branch_id,
                attempt_id=decision.attempt_id,
            )
        )
    return args, kwargs


def apply_mapping_invoke(
    span: DebrixSpan,
    decision: ControlInvoke,
    captured: tuple[dict[str, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Validate a managed object input before one real MCP invocation."""

    if captured is None:
        raise DebrixControlProtocolError(
            "input control requires a lossless JSON object"
        )
    recorded, descriptor = captured
    if not decision.capture_live_result:
        raise DebrixControlProtocolError(
            "input control must capture the live operation result"
        )
    if decision.input.provenance not in {"recorded", "edited", "live"}:
        raise DebrixControlProtocolError(
            "input control provenance must be recorded, edited, or live"
        )
    if not isinstance(decision.input.value, Mapping):
        raise DebrixControlProtocolError("controlled input must be a JSON object")
    if decision.input.provenance in {"recorded", "live"} and decision.input.value != recorded:
        raise DebrixControlProtocolError(
            "Unchanged input must use the exact runtime value"
        )
    candidate = dict(decision.input.value)
    if set(candidate) != set(recorded):
        raise DebrixControlProtocolError(
            "input keys must exactly match editable parameters"
        )
    expected_kinds = {
        parameter["name"]: parameter["jsonKind"]
        for parameter in descriptor["parameters"]
    }
    for name, expected in expected_kinds.items():
        try:
            actual = _json_kind(candidate[name])
        except ValueError as error:
            raise DebrixControlProtocolError(
                f"input parameter {name} must remain JSON {expected}"
            ) from error
        if actual != expected:
            raise DebrixControlProtocolError(
                f"input parameter {name} must remain JSON {expected}"
            )
    schema = descriptor.get("jsonSchema")
    if isinstance(schema, Mapping):
        errors = _schema_errors(candidate, schema)
        if errors:
            raise DebrixControlProtocolError(errors[0])

    span.set_attribute(Attr.CONTROL_BRANCH_ID, decision.branch_id)
    span.set_attribute(Attr.CONTROL_ATTEMPT_ID, decision.attempt_id)
    span.set_attribute(Attr.CONTROL_OCCURRENCE_ID, decision.occurrence_id)
    span.set_attribute(Attr.CONTROL_PROVENANCE, decision.input.provenance)
    span.set_attribute(Attr.CONTROL_INPUT_PROVENANCE, decision.input.provenance)
    span.set_attribute(Attr.CONTROL_RESULT_PROVENANCE, "live")
    span.set_attribute(
        Attr.REPLAY_INPUT,
        json.dumps(_json_safe(candidate), ensure_ascii=False),
    )
    if decision.input.provenance == "edited":
        _LIVE_EXECUTION.set(
            _LiveExecution(
                trace_id=span.trace_id_hex,
                branch_id=decision.branch_id,
                attempt_id=decision.attempt_id,
            )
        )
    return candidate


def apply_llm_messages_invoke(
    span: DebrixSpan,
    decision: ControlInvoke,
    captured: CapturedLlmMessages | None,
) -> list[dict[str, Any]]:
    """Validate edited LLM messages before exactly one provider invocation."""

    if captured is None:
        raise DebrixControlProtocolError(
            "message control requires a lossless JSON message list"
        )
    if not decision.capture_live_result:
        raise DebrixControlProtocolError(
            "message control must capture the live model result"
        )
    if decision.input.provenance not in {"recorded", "edited", "live"}:
        raise DebrixControlProtocolError(
            "message control provenance must be recorded, edited, or live"
        )
    if not isinstance(decision.input.value, Mapping):
        raise DebrixControlProtocolError(
            "controlled messages must be a JSON object"
        )
    if (
        decision.input.provenance in {"recorded", "live"}
        and decision.input.value != captured.recorded_input
    ):
        raise DebrixControlProtocolError(
            "Unchanged messages must use the exact runtime value"
        )
    try:
        messages = captured.reconstruct(decision.input.value)
    except ValueError as error:
        raise DebrixControlProtocolError(str(error)) from error

    span.set_attribute(Attr.CONTROL_BRANCH_ID, decision.branch_id)
    span.set_attribute(Attr.CONTROL_ATTEMPT_ID, decision.attempt_id)
    span.set_attribute(Attr.CONTROL_OCCURRENCE_ID, decision.occurrence_id)
    span.set_attribute(Attr.CONTROL_PROVENANCE, decision.input.provenance)
    span.set_attribute(Attr.CONTROL_INPUT_PROVENANCE, decision.input.provenance)
    span.set_attribute(Attr.CONTROL_RESULT_PROVENANCE, "live")
    if decision.input.provenance == "edited":
        _LIVE_EXECUTION.set(
            _LiveExecution(
                trace_id=span.trace_id_hex,
                branch_id=decision.branch_id,
                attempt_id=decision.attempt_id,
            )
        )
    return messages


def apply_llm_model_output_return(
    span: DebrixSpan,
    decision: ControlReturn,
) -> dict[str, Any]:
    """Validate and return one complete controlled model response object."""

    if decision.output.provenance not in {"recorded", "edited", "live"}:
        raise DebrixControlProtocolError(
            "model-output provenance must be recorded, edited, or live"
        )
    if decision.output.kind != "result":
        raise DebrixControlProtocolError(
            "model-output control requires a result response"
        )
    if not isinstance(decision.output.value, Mapping):
        raise DebrixControlProtocolError(
            "controlled model output must be a JSON object"
        )
    try:
        response = _json_input(decision.output.value)
    except ValueError as error:
        raise DebrixControlProtocolError(
            "controlled model output must be losslessly JSON-compatible"
        ) from error
    assert isinstance(response, dict)
    if not isinstance(response.get("content"), str):
        raise DebrixControlProtocolError(
            "controlled model output.content must be a string"
        )

    span.set_attribute(Attr.CONTROL_BRANCH_ID, decision.branch_id)
    span.set_attribute(Attr.CONTROL_ATTEMPT_ID, decision.attempt_id)
    span.set_attribute(Attr.CONTROL_OCCURRENCE_ID, decision.occurrence_id)
    span.set_attribute(Attr.CONTROL_PROVENANCE, decision.output.provenance)
    span.set_attribute(
        Attr.CONTROL_RESULT_PROVENANCE, decision.output.provenance
    )
    if decision.live_suffix and decision.output.provenance == "edited":
        _LIVE_EXECUTION.set(
            _LiveExecution(
                trace_id=span.trace_id_hex,
                branch_id=decision.branch_id,
                attempt_id=decision.attempt_id,
            )
        )
    return response


def apply_runtime_control(
    span: DebrixSpan,
    decision: ControlDecision | ControlUnmanaged,
) -> tuple[bool, Any]:
    """Apply a result/error decision.

    Returns ``(False, None)`` for an unmanaged operation. A managed result
    returns ``(True, value)``; a managed error raises ``MockToolError``.
    """
    if isinstance(decision, ControlUnmanaged):
        return False, None
    if isinstance(decision, ControlInvoke):
        raise DebrixControlProtocolError(
            "FW2.3 result boundaries cannot receive an invoke decision"
        )
    if not isinstance(decision, ControlReturn):
        raise DebrixControlProtocolError(
            "unsupported managed runtime control decision"
        )

    span.set_attribute(Attr.CONTROL_BRANCH_ID, decision.branch_id)
    span.set_attribute(Attr.CONTROL_ATTEMPT_ID, decision.attempt_id)
    span.set_attribute(Attr.CONTROL_OCCURRENCE_ID, decision.occurrence_id)
    span.set_attribute(Attr.CONTROL_PROVENANCE, decision.output.provenance)
    if decision.live_suffix and decision.output.provenance == "edited":
        _LIVE_EXECUTION.set(
            _LiveExecution(
                trace_id=span.trace_id_hex,
                branch_id=decision.branch_id,
                attempt_id=decision.attempt_id,
            )
        )

    if decision.output.kind == "error":
        error = decision.output.error
        if error is None:
            raise DebrixControlProtocolError(
                "controlled error output is missing its error value"
            )
        span.set_attribute(
            Attr.REPLAY_OUTPUT,
            json.dumps(
                {"error": error.kind, "message": error.message},
                ensure_ascii=False,
            ),
        )
        raise MockToolError(error.kind, error.message)

    span.set_attribute(
        Attr.REPLAY_OUTPUT,
        json.dumps(_json_safe(decision.output.value), ensure_ascii=False),
    )
    return True, decision.output.value
