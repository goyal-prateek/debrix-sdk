import debrix
from debrix.semconv import SPAN_KINDS, Attr, SpanKind

EXPECTED_KINDS = {
    "agent",
    "llm",
    "tool",
    "mcp",
    "memory",
    "evaluation",
    "human",
    "custom",
}


def test_all_span_kinds_present_and_unique() -> None:
    assert len(SPAN_KINDS) == 8
    assert len(set(SPAN_KINDS)) == 8
    assert set(SPAN_KINDS) == EXPECTED_KINDS


def test_span_kind_values_are_lowercase() -> None:
    for value in SPAN_KINDS:
        assert value == value.lower()


def test_span_kind_class_matches_span_kinds_tuple() -> None:
    class_values = {
        SpanKind.AGENT,
        SpanKind.LLM,
        SpanKind.TOOL,
        SpanKind.MCP,
        SpanKind.MEMORY,
        SpanKind.EVALUATION,
        SpanKind.HUMAN,
        SpanKind.CUSTOM,
    }
    assert class_values == set(SPAN_KINDS)


def test_attribute_keys_use_debrix_prefix() -> None:
    keys = [
        value
        for name, value in vars(Attr).items()
        if not name.startswith("_") and isinstance(value, str)
    ]
    assert keys, "expected at least one attribute key"
    for key in keys:
        assert key.startswith("debrix."), key


def test_reexported_from_package_root() -> None:
    assert debrix.SpanKind is SpanKind
    assert debrix.Attr is Attr
    assert debrix.SPAN_KINDS == SPAN_KINDS
