from debrix import configure
from debrix.config import DEFAULT_OTLP_ENDPOINT, reset_for_tests
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider


def setup_function() -> None:
    reset_for_tests()


def test_configure_installs_tracer_provider() -> None:
    provider = configure(batch=False, service_name="debrix-test")
    assert isinstance(provider, TracerProvider)
    assert trace.get_tracer_provider() is provider


def test_configure_is_idempotent() -> None:
    first = configure(batch=False)
    second = configure(batch=False)
    assert first is second


def test_default_endpoint_constant() -> None:
    assert DEFAULT_OTLP_ENDPOINT == "http://127.0.0.1:4318"
