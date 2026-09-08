"""Phase 2 observability tests: Prometheus endpoint + stdlib-only writer.

All offline and deterministic: the writer tests build a registry in-process,
the endpoint tests use fastapi.testclient (httpx, no server, no network).
The contract under test:

- GET /metrics/prometheus returns 200 with media type
  "text/plain; version=0.0.4; charset=utf-8".
- Families: stock_prediction_requests_total (counter; endpoint, method,
  status), stock_prediction_errors_total (counter; endpoint, method,
  error_class), stock_prediction_request_latency_seconds (histogram with
  _bucket/_sum/_count; endpoint, method), stock_prediction_up (gauge).
- Labels are LOW CARDINALITY ONLY: route templates / bounded status codes /
  bounded error classes -- never URLs, request ids, symbols, or user content.
- The JSON GET /metrics response is untouched (covered by test_app.py; here
  we additionally assert its documented keys still exist).
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from stock_prediction.app import app, metrics, prom_metrics
from stock_prediction.prom import (
    LATENCY_BUCKETS,
    PROM_CONTENT_TYPE,
    PrometheusMetrics,
)

client = TestClient(app)

PREFIX = "stock_prediction"
FAMILIES = (
    f"{PREFIX}_requests_total",
    f"{PREFIX}_errors_total",
    f"{PREFIX}_request_latency_seconds",
    f"{PREFIX}_up",
)

# The bounded vocabularies the contract allows in labels.
ALLOWED_ENDPOINTS = {
    "/health",
    "/forecast",
    "/drift",
    "/metrics",
    "/metrics/prometheus",
    "unmatched",
}
ALLOWED_METHODS = {"GET", "POST"}
ALLOWED_STATUSES = {"200", "400", "403", "404", "422", "500"}
ALLOWED_ERROR_CLASSES = {"http_400", "http_403", "http_404", "http_422", "http_500"}


@pytest.fixture(autouse=True)
def _reset_all_metrics():
    metrics.reset()
    prom_metrics.reset()
    yield
    metrics.reset()
    prom_metrics.reset()


def fetch_prom() -> str:
    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == PROM_CONTENT_TYPE
    return resp.text


SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{.*\})? (?P<value>[^ ]+)$")
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_samples(text: str) -> list[tuple[str, dict[str, str], str]]:
    """Parse data lines (non-HELP/TYPE) into (name, labels, raw value)."""
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = SAMPLE_RE.match(line)
        assert m is not None, f"unparseable exposition line: {line!r}"
        labels = dict(LABEL_RE.findall(m.group("labels") or ""))
        samples.append((m.group("name"), labels, m.group("value")))
    return samples


def sample_index(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], str]:
    """Map (metric name, sorted label items) -> raw value for exact lookups."""
    return {
        (name, tuple(sorted(labels.items()))): value for name, labels, value in parse_samples(text)
    }


# ---- writer unit tests (no HTTP) --------------------------------------


def test_render_is_deterministic() -> None:
    a = PrometheusMetrics()
    b = PrometheusMetrics()
    for reg in (a, b):
        reg.record(endpoint="/forecast", method="POST", status=200, latency_s=0.02)
        reg.record(endpoint="/health", method="GET", status=200, latency_s=0.001)
        reg.record(
            endpoint="/forecast", method="POST", status=400, latency_s=0.01, error_class="http_400"
        )
    assert a.render() == b.render()
    # Same instance twice: byte-identical text with unchanged data.
    assert a.render() == a.render()


def test_writer_emits_documented_families_and_types() -> None:
    reg = PrometheusMetrics()
    reg.record(endpoint="/health", method="GET", status=200, latency_s=0.001)
    reg.record(
        endpoint="/forecast", method="POST", status=400, latency_s=0.01, error_class="http_400"
    )
    text = reg.render()
    lines = text.splitlines()
    for fam in FAMILIES:
        assert f"# TYPE {fam}" in lines or any(
            line.startswith(f"# TYPE {fam} ") for line in lines
        ), f"missing TYPE for {fam}"
    assert "# TYPE stock_prediction_requests_total counter" in lines
    assert "# TYPE stock_prediction_errors_total counter" in lines
    assert "# TYPE stock_prediction_request_latency_seconds histogram" in lines
    assert "# TYPE stock_prediction_up gauge" in lines
    assert re.search(r"^stock_prediction_requests_total\{", text, re.MULTILINE)
    assert re.search(r"^stock_prediction_errors_total\{", text, re.MULTILINE)
    assert re.search(r"^stock_prediction_request_latency_seconds_bucket\{", text, re.MULTILINE)
    assert re.search(r"^stock_prediction_request_latency_seconds_sum\{", text, re.MULTILINE)
    assert re.search(r"^stock_prediction_request_latency_seconds_count\{", text, re.MULTILINE)
    assert re.search(r"^stock_prediction_up 1$", text, re.MULTILINE)


def test_writer_histogram_buckets_sum_and_inf() -> None:
    reg = PrometheusMetrics()
    reg.record(endpoint="/health", method="GET", status=200, latency_s=0.007)
    reg.record(endpoint="/health", method="GET", status=200, latency_s=0.3)
    text = reg.render()
    idx = sample_index(text)
    ep = {"endpoint": "/health", "method": "GET"}
    # 0.007 lands in buckets >= 0.01 only; 0.3 in buckets >= 0.5 only.
    assert (
        idx[
            (
                f"{PREFIX}_request_latency_seconds_bucket",
                tuple(sorted({**ep, "le": "0.005"}.items())),
            )
        ]
        == "0"
    )
    assert (
        idx[
            (
                f"{PREFIX}_request_latency_seconds_bucket",
                tuple(sorted({**ep, "le": "0.01"}.items())),
            )
        ]
        == "1"
    )
    assert (
        idx[
            (f"{PREFIX}_request_latency_seconds_bucket", tuple(sorted({**ep, "le": "0.5"}.items())))
        ]
        == "2"
    )
    assert (
        idx[
            (
                f"{PREFIX}_request_latency_seconds_bucket",
                tuple(sorted({**ep, "le": "+Inf"}.items())),
            )
        ]
        == "2"
    )
    assert idx[(f"{PREFIX}_request_latency_seconds_count", tuple(sorted(ep.items())))] == "2"
    assert float(
        idx[(f"{PREFIX}_request_latency_seconds_sum", tuple(sorted(ep.items())))]
    ) == pytest.approx(0.307)
    # Buckets are monotone non-decreasing in le.
    bucket_vals = [
        int(v)
        for name, labels, v in parse_samples(text)
        if name.endswith("_bucket") and labels.get("endpoint") == "/health"
    ]
    assert bucket_vals == sorted(bucket_vals)
    # Every documented bucket appears with le, in seconds.
    rendered_les = [
        labels["le"] for name, labels, _ in parse_samples(text) if name.endswith("_bucket")
    ]
    assert set(rendered_les) == {repr(float(b)) for b in LATENCY_BUCKETS} | {"+Inf"}


def test_writer_escapes_label_values() -> None:
    reg = PrometheusMetrics()
    value = "weird" + chr(34) + "endpoint" + chr(92)  # weird"endpoint\
    reg.record(endpoint=value, method="GET", status=200, latency_s=0.001)
    text = reg.render()
    expected = "weird" + chr(92) + chr(34) + "endpoint" + chr(92) + chr(92)
    assert 'endpoint="' + expected + '"' in text
    # The escaped render must still parse back to the same label value.
    parsed = parse_samples(text)
    assert parsed
    ep_values = [labels.get("endpoint") for _, labels, _ in parsed]
    assert expected in ep_values  # escaped form survives the round trip


# ---- endpoint contract tests -------------------------------------------


def test_prometheus_endpoint_status_and_content_type() -> None:
    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    assert resp.text.endswith("\n")


def test_expected_metric_names_present() -> None:
    assert client.get("/health").status_code == 200
    assert client.post("/forecast", json={"fixture": "nope"}).status_code == 400
    text = fetch_prom()
    names = {name for name, _, _ in parse_samples(text)}
    assert f"{PREFIX}_requests_total" in names
    assert f"{PREFIX}_errors_total" in names
    assert f"{PREFIX}_request_latency_seconds_bucket" in names
    assert f"{PREFIX}_request_latency_seconds_sum" in names
    assert f"{PREFIX}_request_latency_seconds_count" in names
    assert f"{PREFIX}_up" in names


def test_labels_are_bounded() -> None:
    # Generate every status path the app actually produces (offline).
    assert client.get("/health").status_code == 200
    assert client.get("/drift").status_code == 200
    assert client.post("/forecast", json={"model": "persistence"}).status_code == 200
    assert client.post("/forecast", json={"fixture": "nope"}).status_code == 400
    assert client.post("/forecast", json={"model": "deep_learning"}).status_code == 422
    assert client.get("/no_such_route").status_code == 404
    text = fetch_prom()
    endpoints: set[str] = set()
    methods: set[str] = set()
    statuses: set[str] = set()
    error_classes: set[str] = set()
    for name, labels, _ in parse_samples(text):
        if name == f"{PREFIX}_up":
            assert not labels
            continue
        assert set(labels) <= {"endpoint", "method", "status", "error_class", "le"}
        endpoints.add(labels["endpoint"])
        methods.add(labels["method"])
        if "status" in labels:
            statuses.add(labels["status"])
        if "error_class" in labels:
            error_classes.add(labels["error_class"])
        # histograms carry le and never status/error_class
        assert "?" not in labels.get("endpoint", "")
        assert "http://" not in labels.get("endpoint", "")
    assert endpoints <= ALLOWED_ENDPOINTS, endpoints
    assert methods <= ALLOWED_METHODS, methods
    assert statuses <= ALLOWED_STATUSES, statuses
    assert error_classes <= ALLOWED_ERROR_CLASSES, error_classes


def test_request_increments_requests_total() -> None:
    assert client.get("/health").status_code == 200
    text = fetch_prom()
    idx = sample_index(text)
    key = (
        f"{PREFIX}_requests_total",
        (("endpoint", "/health"), ("method", "GET"), ("status", "200")),
    )
    assert idx[key] == "1"
    # The exposition call itself is recorded by the middleware after the
    # response is rendered (same semantics as the JSON /metrics counters).
    idx2 = sample_index(fetch_prom())
    key2 = (
        f"{PREFIX}_requests_total",
        (("endpoint", "/metrics/prometheus"), ("method", "GET"), ("status", "200")),
    )
    assert idx2[key2] == "1"


def test_latency_metrics_exposed_after_a_request() -> None:
    assert client.get("/health").status_code == 200
    text = fetch_prom()
    idx = sample_index(text)
    ep = {"endpoint": "/health", "method": "GET"}
    count = idx[(f"{PREFIX}_request_latency_seconds_count", tuple(sorted(ep.items())))]
    total = idx[
        (f"{PREFIX}_request_latency_seconds_bucket", tuple(sorted({**ep, "le": "+Inf"}.items())))
    ]
    assert count == total == "1"
    assert float(idx[(f"{PREFIX}_request_latency_seconds_sum", tuple(ep.items()))]) >= 0.0


def test_error_counter_increments_after_triggering_request() -> None:
    assert client.post("/forecast", json={"fixture": "not_a_fixture"}).status_code == 400
    text = fetch_prom()
    idx = sample_index(text)
    key = (
        f"{PREFIX}_errors_total",
        (("endpoint", "/forecast"), ("error_class", "http_400"), ("method", "POST")),
    )
    assert idx[key] == "1"
    assert f"{PREFIX}_up" in {name for name, _, _ in parse_samples(text)}


def test_unmatched_route_uses_bounded_endpoint_label() -> None:
    assert client.get("/definitely/not/a/route").status_code == 404
    text = fetch_prom()
    idx = sample_index(text)
    key = (
        f"{PREFIX}_requests_total",
        (("endpoint", "unmatched"), ("method", "GET"), ("status", "404")),
    )
    assert idx[key] == "1"


def test_up_gauge_present() -> None:
    text = fetch_prom()
    assert re.search(r"^# HELP stock_prediction_up ", text, re.MULTILINE)
    assert re.search(r"^stock_prediction_up 1$", text, re.MULTILINE)


def test_json_metrics_endpoint_unchanged_shape() -> None:
    """The pre-existing JSON /metrics contract is untouched by Phase 2."""
    assert client.get("/health").status_code == 200
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert set(body) == {
        "request_count",
        "error_count",
        "error_rate",
        "latency_ms",
        "last_drift",
        "note",
    }
