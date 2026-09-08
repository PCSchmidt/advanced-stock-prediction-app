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

import csv
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stock_prediction.app import _load_static_prom_metrics, app, metrics, prom_metrics
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
# Phase 3 bounded vocabularies.
ALLOWED_MODELS = {"persistence", "hist_gradient_boosting", "both"}
ALLOWED_OUTCOMES = {"fired", "quiet"}


@pytest.fixture(autouse=True)
def _reset_all_metrics():
    metrics.reset()
    prom_metrics.reset()
    _load_static_prom_metrics()  # reload eval gauges + model_info after reset
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
    models: set[str] = set()
    outcomes: set[str] = set()
    sklearn_versions: set[str] = set()
    for name, labels, _ in parse_samples(text):
        if name == f"{PREFIX}_up":
            assert not labels
            continue
        if name.startswith(f"{PREFIX}_forecast_"):
            assert set(labels) <= {"model", "status", "le"}
            if "model" in labels:
                models.add(labels["model"])
            if "status" in labels:
                statuses.add(labels["status"])
        elif name.startswith(f"{PREFIX}_drift_checks"):
            assert set(labels) == {"outcome"}
            outcomes.add(labels["outcome"])
        elif name == f"{PREFIX}_drift_state":
            assert not labels
        elif name.startswith(f"{PREFIX}_eval_"):
            assert set(labels) == {"model"}
            models.add(labels["model"])
        elif name == f"{PREFIX}_model_info":
            assert set(labels) == {"model", "sklearn_version"}
            models.add(labels["model"])
            sklearn_versions.add(labels["sklearn_version"])
        else:
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
    assert models <= ALLOWED_MODELS, models
    assert outcomes <= ALLOWED_OUTCOMES, outcomes
    # model_info carries exactly one bounded sklearn version.
    assert len(sklearn_versions) == 1, sklearn_versions


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


# ---- Phase 3: app-domain families --------------------------------------

PHASE3_FAMILIES = (
    f"{PREFIX}_forecast_requests_total",
    f"{PREFIX}_forecast_latency_seconds",
    f"{PREFIX}_drift_checks_total",
    f"{PREFIX}_drift_state",
    f"{PREFIX}_eval_rmse",
    f"{PREFIX}_eval_mae",
    f"{PREFIX}_eval_directional_accuracy",
    f"{PREFIX}_eval_edge",
    f"{PREFIX}_model_info",
)


def test_phase3_families_exposed_after_requests() -> None:
    """All Phase 3 families are exposed on the same endpoint, after the
    generic HTTP families, once real (offline) traffic has flowed."""
    assert (
        client.post("/forecast", json={"model": "persistence", "max_rows": 120}).status_code == 200
    )
    assert client.get("/drift?fixture=vol_regime_shift").status_code == 200
    text = fetch_prom()
    names = {name for name, _, _ in parse_samples(text)}
    for fam in PHASE3_FAMILIES:
        base = f"{fam}_bucket" if fam.endswith("_seconds") else fam
        assert base in names, f"missing family {fam}"
    lines = text.splitlines()
    for fam in PHASE3_FAMILIES:
        assert any(line.startswith(f"# TYPE {fam} ") for line in lines), f"no TYPE for {fam}"
    # Generic families render first, Phase 3 families after `up`.
    first_pos = {
        name: text.index(f"# HELP {name} ")
        for name in (
            f"{PREFIX}_requests_total",
            f"{PREFIX}_up",
            f"{PREFIX}_forecast_requests_total",
        )
    }
    assert first_pos[f"{PREFIX}_requests_total"] < first_pos[f"{PREFIX}_up"]
    assert first_pos[f"{PREFIX}_up"] < first_pos[f"{PREFIX}_forecast_requests_total"]


def test_forecast_counter_and_latency_increment() -> None:
    assert (
        client.post("/forecast", json={"model": "persistence", "max_rows": 120}).status_code == 200
    )
    text = fetch_prom()
    idx = sample_index(text)
    key = (
        f"{PREFIX}_forecast_requests_total",
        (("model", "persistence"), ("status", "200")),
    )
    assert idx[key] == "1"
    labels = {"model": "persistence"}
    assert idx[(f"{PREFIX}_forecast_latency_seconds_count", tuple(sorted(labels.items())))] == "1"
    assert (
        idx[
            (
                f"{PREFIX}_forecast_latency_seconds_bucket",
                tuple(sorted({**labels, "le": "+Inf"}.items())),
            )
        ]
        == "1"
    )
    assert (
        float(idx[(f"{PREFIX}_forecast_latency_seconds_sum", tuple(sorted(labels.items())))]) >= 0.0
    )


def test_forecast_error_status_recorded_with_requested_model() -> None:
    assert client.post("/forecast", json={"fixture": "not_a_fixture"}).status_code == 400
    text = fetch_prom()
    idx = sample_index(text)
    key = (
        f"{PREFIX}_forecast_requests_total",
        (("model", "both"), ("status", "400")),
    )
    assert idx[key] == "1"
    # Early validation exits record a counter but NO latency sample.
    assert f"{PREFIX}_forecast_latency_seconds_count" not in {n for n, _, _ in parse_samples(text)}


def test_schema_rejections_422_never_reach_forecast_family() -> None:
    """422s are rejected by pydantic before the handler; the user's invalid
    model string must never appear as a label value."""
    assert client.post("/forecast", json={"model": "deep_learning"}).status_code == 422
    text = fetch_prom()
    assert "deep_learning" not in text
    for name, labels, _ in parse_samples(text):
        if name.startswith(f"{PREFIX}_forecast_"):
            assert labels.get("model", "both") in ALLOWED_MODELS


def test_drift_counters_increment_after_checks() -> None:
    # vol_regime_shift halves fire (recorded: PSI 1.13, significant).
    assert client.get("/drift?fixture=vol_regime_shift").status_code == 200
    text = fetch_prom()
    idx = sample_index(text)
    fired_key = (f"{PREFIX}_drift_checks_total", (("outcome", "fired"),))
    assert idx[fired_key] == "1"
    assert idx[(f"{PREFIX}_drift_state", ())] == "1"
    # A quiet check afterwards moves the state gauge back to 0.
    assert client.get("/drift").status_code == 200  # sample_daily: quiet
    idx = sample_index(fetch_prom())
    assert idx[(f"{PREFIX}_drift_checks_total", (("outcome", "quiet"),))] == "1"
    assert idx[(f"{PREFIX}_drift_state", ())] == "0"


def test_drift_via_forecast_counts_too() -> None:
    """The forecast path runs the detector on the served closes; completed
    checks there count exactly like /drift checks."""
    assert (
        client.post(
            "/forecast", json={"fixture": "vol_regime_shift", "model": "persistence"}
        ).status_code
        == 200
    )
    idx = sample_index(fetch_prom())
    assert idx[(f"{PREFIX}_drift_checks_total", (("outcome", "fired"),))] == "1"
    assert idx[(f"{PREFIX}_drift_state", ())] == "1"


def test_skipped_drift_checks_are_not_counted() -> None:
    """max_rows=90 is too short for two stable windows: the check is skipped
    and must change neither the counter nor the state gauge."""
    before = sample_index(fetch_prom())
    fired_before = int(before.get((f"{PREFIX}_drift_checks_total", (("outcome", "fired"),)), "0"))
    assert (
        client.post("/forecast", json={"model": "persistence", "max_rows": 90}).status_code == 200
    )
    after = sample_index(fetch_prom())
    fired_after = int(after.get((f"{PREFIX}_drift_checks_total", (("outcome", "fired"),)), "0"))
    assert fired_after == fired_before
    assert f"{PREFIX}_drift_state" not in {name for name, _, _ in parse_samples(fetch_prom())}


def test_eval_gauges_exact_label_sets_and_values() -> None:
    """Eval gauges are loaded from the committed Stage 2 CSV and match it
    exactly: model labels from the CSV rows, values from the CSV cells."""
    text = fetch_prom()
    idx = sample_index(text)
    csv_rows: dict[str, dict[str, str]] = {}
    with open(
        Path(__file__).resolve().parents[1] / "experiments" / "results.csv",
        newline="",
        encoding="utf-8",
    ) as fh:
        for row in csv.DictReader(fh):
            if row["fixture"] == "sample_daily" and row["window"] == "full":
                csv_rows[row["model"]] = row
    assert set(csv_rows) == {"persistence", "hist_gradient_boosting"}
    for metric in ("rmse", "mae", "edge"):
        for model, row in csv_rows.items():
            key = (f"{PREFIX}_eval_{metric}", (("model", model),))
            assert key in idx, f"missing {metric} for {model}"
            assert float(idx[key]) == pytest.approx(float(row[metric]))
    # directional_accuracy is undefined for persistence (empty cell): the
    # series must be absent for that model, present for the GBM.
    assert (f"{PREFIX}_eval_directional_accuracy", (("model", "persistence"),)) not in idx
    gbm_key = (f"{PREFIX}_eval_directional_accuracy", (("model", "hist_gradient_boosting"),))
    assert float(idx[gbm_key]) == pytest.approx(
        float(csv_rows["hist_gradient_boosting"]["directional_accuracy"])
    )
    # The exact evaluation context lives in HELP text, never as labels.
    assert "# HELP stock_prediction_eval_rmse" in text
    help_line = next(
        line for line in text.splitlines() if line.startswith("# HELP stock_prediction_eval_rmse")
    )
    assert "fixture=sample_daily" in help_line
    assert "window=full" in help_line
    assert f"n={csv_rows['persistence']['n']} walk-forward origins" in help_line
    assert "EVALUATION-CONTEXT" in help_line


def test_eval_loader_ignores_incompatible_windows() -> None:
    """Only fixture=sample_daily, window=full rows are loaded; other windows
    and fixtures in the same CSV can never surface as series."""
    text = fetch_prom()
    # trending_up (other fixture) and first_half/second_half values must not
    # appear anywhere in the exposition.
    assert "trending_up" not in text
    assert "first_half" not in text
    assert "second_half" not in text
    idx = sample_index(text)
    persistence_rmse = idx[(f"{PREFIX}_eval_rmse", (("model", "persistence"),))]
    assert persistence_rmse == "0.014733"  # sample_daily/full value, not 0.014310 (first_half)


def test_model_info_gauge() -> None:
    text = fetch_prom()
    idx = sample_index(text)
    for model in ("persistence", "hist_gradient_boosting"):
        keys = [k for k in idx if k[0] == f"{PREFIX}_model_info" and dict(k[1])["model"] == model]
        assert len(keys) == 1, f"expected exactly one {model} identity series"
        assert idx[keys[0]] == "1"
        assert dict(keys[0][1])["sklearn_version"]
