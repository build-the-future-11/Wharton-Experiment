"""Lossless weekly state generations, issuance and maturity audit evidence.

Compression is an engineering representation only. Every state generation is
retained; it cannot authorize market resume. Verification decodes saved records
and checks identities, never refits a model or reads source observations.
"""

import base64
import zlib
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np

from frem_s10.domain import ForecastDistribution
from frem_s10.errors import ArtifactIOError, ContractError, IntegrityError
from frem_s10.identities import JSONObject, JSONValue, checked_json, object_json, scientific_digest
from frem_s10.model.checkpoint import _decode, _encode, dump_state, load_state
from frem_s10.model.config import ForecastConfig
from frem_s10.model.engine import initial_state
from frem_s10.model.records import (
    ForecastTransition,
    IssuanceBatch,
    IssuedForecast,
    MaturedFeedback,
    MixtureBundle,
    scientific_id,
)
from frem_s10.runtime.artifacts import _plain_directory, publish_json, read_verified_json

from .matrix import FORECAST_ARMS, MODEL_VARIANTS

_CHECKPOINT_LIMIT = 16 * 1024 * 1024
_UNITS: JSONObject = {
    "assets": ["SPY", "TLT", "GLD"],
    "g": "price_gross",
    "d": "cash_dividend_per_start_dollar",
    "probabilities_and_alpha": "probability",
    "context": "fixed_burnin_standardized_dimension_specific",
    "request_x_and_library_contexts": "raw_context_dimension_specific",
    "maturity_a_b": "standardized_total_return_euclidean_distance",
    "mixture_objective": "regularized_standardized_total_return_energy_score",
    "clock": "UTC_microseconds",
}


def _text(value: JSONValue) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrityError("nonempty archive identity required")
    return value


def _checkpoint(text: str) -> str:
    raw = text.encode()
    if len(raw) > _CHECKPOINT_LIMIT:
        raise ContractError("state generation exceeds the declared checkpoint decoder bound")
    return base64.b85encode(zlib.compress(raw, level=6)).decode("ascii")


def _unpack(text: str) -> str:
    if len(text) > 2 * _CHECKPOINT_LIMIT:
        raise IntegrityError("compressed checkpoint exceeds its byte bound")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b85decode(text), _CHECKPOINT_LIMIT + 1)
        if len(raw) > _CHECKPOINT_LIMIT or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise IntegrityError("checkpoint compression is truncated, excessive or concatenated")
        return raw.decode("utf-8")
    except (ValueError, zlib.error, UnicodeError) as exc:
        raise IntegrityError("checkpoint compression rejected") from exc


def _issued_identity(issued: IssuedForecast) -> str:
    request = issued.request
    return scientific_id(
        "issued-forecast-v2",
        (
            issued.arm,
            (
                request.record_id,
                request.origin_us,
                request.endpoint_us,
                request.available_us,
                request.feature_available_us,
                request.anchor_index,
            ),
            issued.experts.record_id,
            issued.state_before_id,
        ),
        request.x,
        issued.g,
        issued.d,
        issued.probabilities,
        issued.alpha,
    )


def _validate_transition(step: ForecastTransition) -> None:
    if tuple(item.arm for item in step.batch.forecasts) != FORECAST_ARMS:
        raise IntegrityError("weekly archive requires all eight original issued arms")
    representative = step.batch.get("contextual_stack")
    if not step.state.pending or step.state.pending[-1].record_id != representative.record_id:
        raise IntegrityError("issued representative is absent from saved pending state")
    if representative.request.decision_us != step.state.cursor_us:
        raise IntegrityError("issued request differs from state decision cursor")
    if _encode(representative.experts.library) != _encode(step.state.library):
        raise IntegrityError("issued expert atoms differ from the post-maturity library")
    for issued in step.batch.forecasts:
        if (
            issued.request.record_id != representative.request.record_id
            or issued.state_before_id != representative.state_before_id
            or issued.experts.record_id != representative.experts.record_id
            or issued.record_id != _issued_identity(issued)
        ):
            raise IntegrityError("issued laws disagree with their common immutable information state")
    retained = {item.record_id: item for item in step.state.feedback}
    if len({item.record_id for item in step.matured_feedback}) != len(step.matured_feedback) or any(
        item.record_id not in retained
        or item.available_us >= step.state.cursor_us
        or _encode(item) != _encode(retained[item.record_id])
        for item in step.matured_feedback
    ):
        raise IntegrityError("maturity evidence is duplicate, absent or not yet available")
    _validate_mixture_links(step)


def _validate_mixture_links(step: ForecastTransition) -> None:
    mixture = step.batch.mixtures
    for arm, alpha in (
        ("global_stack", mixture.global_alpha),
        ("contextual_stack", mixture.contextual_alpha),
        ("individual_score_mix", mixture.individual_alpha),
    ):
        if not np.array_equal(step.batch.get(arm).alpha, alpha):
            raise IntegrityError("mixture certificate weights differ from issued law")
    for certificate, alpha in (
        (mixture.global_certificate, mixture.global_alpha),
        (mixture.local_certificate, mixture.local_alpha),
    ):
        if certificate is not None and not np.array_equal(certificate.alpha, alpha):
            raise IntegrityError("solver certificate differs from retained mixture")


def _payload(step: ForecastTransition) -> JSONObject:
    _validate_transition(step)
    return {
        "state_id": step.state.record_id,
        "parent_state_id": step.state.previous_state_id,
        "transform_id": step.state.transform.record_id,
        "config": checked_json(asdict(step.state.config)),
        "checkpoint_codec": "zlib-base85-utf8",
        "checkpoint": _checkpoint(dump_state(step.state)),
        "issued": [
            {"arm": item.arm, "record_id": item.record_id, "alpha": checked_json(_encode(item.alpha))}
            for item in step.batch.forecasts
        ],
        "mixtures": checked_json(_encode(step.batch.mixtures)),
        "matured_feedback": checked_json(_encode(step.matured_feedback)),
    }


def decode_transition(payload: JSONObject) -> ForecastTransition:
    """Reconstruct compact laws from the saved exact common expert atoms."""
    if (
        set(payload)
        != {
            "state_id",
            "parent_state_id",
            "transform_id",
            "config",
            "checkpoint_codec",
            "checkpoint",
            "issued",
            "mixtures",
            "matured_feedback",
        }
        or payload["checkpoint_codec"] != "zlib-base85-utf8"
    ):
        raise IntegrityError("weekly transition schema differs from its exact contract")
    config_fields = object_json(payload["config"])
    if set(config_fields) != set(asdict(ForecastConfig())):
        raise IntegrityError("archived model configuration fields differ")
    decoded_config = _decode({"record": "ForecastConfig", "fields": config_fields})
    if not isinstance(decoded_config, ForecastConfig):
        raise IntegrityError("invalid archived model configuration")
    state = load_state(
        _unpack(_text(payload["checkpoint"])),
        expected_state_id=_text(payload["state_id"]),
        expected_transform_id=_text(payload["transform_id"]),
        expected_config=decoded_config,
    )
    if state.previous_state_id != payload["parent_state_id"] or not state.pending:
        raise IntegrityError("archived checkpoint parent or pending representative differs")
    issued = payload["issued"]
    if not isinstance(issued, list):
        raise IntegrityError("ordered issued law registry required")
    forecasts = tuple(_restore_issued(object_json(item), state.pending[-1]) for item in issued)
    mixtures = _decode(payload["mixtures"])
    matured = _decode(payload["matured_feedback"])
    if (
        not isinstance(mixtures, MixtureBundle)
        or not isinstance(matured, tuple)
        or any(not isinstance(item, MaturedFeedback) for item in matured)
    ):
        raise IntegrityError("typed mixture certificates and maturity records required")
    step = ForecastTransition(state, IssuanceBatch(forecasts, mixtures), matured)
    _validate_transition(step)
    return step


def _restore_issued(value: JSONObject, representative: IssuedForecast) -> IssuedForecast:
    if set(value) != {"arm", "record_id", "alpha"}:
        raise IntegrityError("issued law schema differs")
    alpha = _decode(value["alpha"])
    if not isinstance(alpha, np.ndarray) or alpha.dtype != np.dtype("float64") or alpha.shape != (3,):
        raise IntegrityError("archived expert mixture must be binary64 length three")
    arm, identity = _text(value["arm"]), _text(value["record_id"])
    library = representative.experts.library
    g, d, probabilities = library.g, library.d, alpha @ representative.experts.probabilities
    if arm == "zero":
        g, d, probabilities = np.ones((1, 3)), np.zeros((1, 3)), np.ones(1)
    return IssuedForecast(
        identity,
        arm,
        representative.request,
        ForecastDistribution(identity, g, d, probabilities),
        representative.experts,
        alpha,
        representative.state_before_id,
    )


class ModelArchiveWriter:
    """Commit a complete immutable generation synchronously before the next issue."""

    def __init__(
        self,
        directory: Path,
        *,
        experiment_id: str,
        evidence_scope: str,
        binding_id: str,
        variants: tuple[str, ...] = MODEL_VARIANTS,
    ) -> None:
        """Create a fresh archive bound to an explicit operation/freeze identity."""
        self.directory = _plain_directory(directory)
        if any(self.directory.iterdir()) or variants not in (MODEL_VARIANTS, ("primary",)):
            raise ArtifactIOError("new empty archive and exact operation variant registry required")
        if evidence_scope not in (
            "KERNEL_REPLAY",
            "GENERATED_ENGINEERING",
            "SOURCE_VIEW_WITHOUT_SCIENTIFIC_ADMISSION",
            "RETROSPECTIVE_DEVELOPMENT",
        ):
            raise ContractError("explicit nonconfirmatory archive scope required")
        self.header: JSONObject = {
            "schema": "s10-weekly-model-archive-v1",
            "experiment_id": _text(experiment_id),
            "evidence_scope": evidence_scope,
            "binding_id": _text(binding_id),
            "variants": list(variants),
            "units": _UNITS,
            "market_resume_allowed": False,
            "scientific_support_claimed": False,
        }
        self.variants = variants
        self.rows: list[JSONValue] = []
        self.last: dict[str, tuple[str, int]] = {}
        self.origin: tuple[int, str, int] | None = None
        self.transform_identity: str | None = None
        self.shape_counts = _empty_counts()
        self.failed = False
        self.closed = False

    def sink(self, variant: str, transition: ForecastTransition) -> None:
        """Preserve a transition exactly once; a failed write poisons this writer."""
        if self.closed or self.failed:
            raise IntegrityError("archive is finalized or failed; automatic continuation is forbidden")
        try:
            self._append(variant, transition)
        except Exception:
            self.failed = True
            raise

    def _append(self, variant: str, step: ForecastTransition) -> None:
        ordinal = len(self.rows)
        if ordinal >= 1000 * len(self.variants):
            raise IntegrityError("archive exceeds declared weekly issue budget")
        if variant != self.variants[ordinal % len(self.variants)]:
            raise IntegrityError("weekly model variants are missing, repeated or out of order")
        self.transform_identity = _fixed_transform(step, self.transform_identity)
        request = step.batch.get("contextual_stack").request
        origin = (request.anchor_index, request.record_id, request.decision_us)
        if ordinal % len(self.variants) == 0:
            if self.origin is not None and (origin[0] <= self.origin[0] or origin[2] <= self.origin[2]):
                raise IntegrityError("weekly model origin regressed or repeated")
            self.origin = origin
        elif origin != self.origin:
            raise IntegrityError("weekly variants disagree about their common origin")
        previous = self.last.get(variant)
        expected_parent = (
            initial_state(step.state.transform, step.state.config).record_id
            if previous is None
            else previous[0]
        )
        if step.state.previous_state_id != expected_parent or step.state.config != ForecastConfig().variant(
            variant
        ):
            raise IntegrityError("weekly state parent or variant configuration differs")
        payload: JSONObject = {
            **self.header,
            "ordinal": ordinal,
            "variant": variant,
            "previous_record_sha256": None if not self.rows else object_json(self.rows[-1])["sha256"],
            "transition": _payload(step),
        }
        name = f"weekly-{ordinal:04}.json"
        digest = publish_json(self.directory, name, payload)
        self.rows.append(
            {"name": name, "sha256": digest, "variant": variant, "state_id": step.state.record_id}
        )
        self.last[variant] = (step.state.record_id, request.decision_us)
        _count_shapes(self.shape_counts, step, len(self.variants))

    def finish(self) -> str:
        """Publish the complete inventory after every variant's last weekly issue."""
        if self.failed or self.closed or not self.rows or len(self.rows) % len(self.variants):
            raise IntegrityError("archive cannot finish with a failure or incomplete weekly registry")
        self.closed = True
        return publish_json(
            self.directory,
            "manifest.json",
            {
                **self.header,
                "files": self.rows,
                "shape_counts": dict(self.shape_counts),
            },
        )


def _manifest_registry(manifest: JSONObject) -> tuple[list[JSONValue], list[JSONValue], JSONObject]:
    if set(manifest) != {
        "schema",
        "experiment_id",
        "evidence_scope",
        "binding_id",
        "variants",
        "units",
        "market_resume_allowed",
        "scientific_support_claimed",
        "files",
        "shape_counts",
    }:
        raise IntegrityError("archive manifest schema differs")
    rows = manifest["files"]
    variants = manifest["variants"]
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) > 6000
        or not isinstance(variants, list)
        or variants not in (list(MODEL_VARIANTS), ["primary"])
    ):
        raise IntegrityError("archive inventory or operation variant registry differs")
    if (
        manifest["schema"] != "s10-weekly-model-archive-v1"
        or manifest["units"] != _UNITS
        or manifest["market_resume_allowed"] is not False
        or manifest["scientific_support_claimed"] is not False
    ):
        raise IntegrityError("archive schema, units or claim scope differs")
    header = {key: value for key, value in manifest.items() if key not in ("files", "shape_counts")}
    _text(header["experiment_id"])
    _text(header["binding_id"])
    if header["evidence_scope"] not in (
        "KERNEL_REPLAY",
        "GENERATED_ENGINEERING",
        "SOURCE_VIEW_WITHOUT_SCIENTIFIC_ADMISSION",
        "RETROSPECTIVE_DEVELOPMENT",
    ) or len(rows) > 1000 * len(variants):
        raise IntegrityError("archive scope or weekly issue budget differs")
    return rows, variants, header


def verify_model_archive(directory: Path, manifest_sha256: str) -> JSONObject:
    """Read and validate every saved state chain without refitting or continuation."""
    manifest = object_json(
        read_verified_json(directory / "manifest.json", manifest_sha256, maximum_bytes=4_194_304)
    )
    rows, variants, header = _manifest_registry(manifest)
    names = {f"weekly-{index:04}.json" for index in range(len(rows))} | {"manifest.json"}
    if {path.name for path in directory.iterdir()} != names or len(rows) % len(variants):
        raise IntegrityError("archive has missing/extra files or incomplete weekly variants")
    previous: str | None = None
    states: dict[str, str] = {}
    origin: tuple[int, str, int] | None = None
    transform_identity: str | None = None
    shape_counts = _empty_counts()
    for ordinal, raw in enumerate(rows):
        row = object_json(raw)
        if (
            set(row) != {"name", "sha256", "variant", "state_id"}
            or row["name"] != f"weekly-{ordinal:04}.json"
            or row["variant"] != variants[ordinal % len(variants)]
        ):
            raise IntegrityError("archive inventory order or identity differs")
        value = object_json(
            read_verified_json(
                directory / _text(row["name"]), _text(row["sha256"]), maximum_bytes=2 * _CHECKPOINT_LIMIT
            )
        )
        if (
            set(value) != set(header) | {"ordinal", "variant", "previous_record_sha256", "transition"}
            or any(value[key] != expected for key, expected in header.items())
            or value["ordinal"] != ordinal
            or value["variant"] != row["variant"]
            or value["previous_record_sha256"] != previous
        ):
            raise IntegrityError("archive event envelope or hash chain differs")
        step = decode_transition(object_json(value["transition"]))
        transform_identity = _fixed_transform(step, transform_identity)
        _count_shapes(shape_counts, step, len(variants))
        variant = _text(row["variant"])
        parent = states.get(variant, initial_state(step.state.transform, step.state.config).record_id)
        if (
            step.state.record_id != row["state_id"]
            or step.state.previous_state_id != parent
            or step.state.config != ForecastConfig().variant(variant)
        ):
            raise IntegrityError("archive state identity, parent or configuration differs")
        request = step.batch.get("contextual_stack").request
        current = (request.anchor_index, request.record_id, request.decision_us)
        if ordinal % len(variants) == 0:
            if origin is not None and (current[0] <= origin[0] or current[2] <= origin[2]):
                raise IntegrityError("archive origin order regressed")
            origin = current
        elif current != origin:
            raise IntegrityError("archive variants disagree about origin")
        states[variant] = step.state.record_id
        previous = _text(row["sha256"])
    recorded = object_json(manifest["shape_counts"])
    if (
        set(recorded) != set(shape_counts)
        or any(type(value) is not int for value in recorded.values())
        or recorded != shape_counts
    ):
        raise IntegrityError("archive shape counts differ from independently decoded generations")
    return manifest


def _fixed_transform(step: ForecastTransition, expected: str | None) -> str:
    identity = scientific_digest(checked_json(_encode(step.state.transform)))
    if expected is not None and identity != expected:
        raise IntegrityError("archived fixed burn-in transform changed between issues or variants")
    return identity


def _empty_counts() -> dict[str, int]:
    return {"transitions": 0, "weekly_issues": 0, "max_library": 0, "max_feedback": 0, "max_pending": 0}


def _count_shapes(counts: dict[str, int], step: ForecastTransition, variants: int) -> None:
    counts["transitions"] += 1
    counts["weekly_issues"] = counts["transitions"] // variants
    for name, size in (
        ("max_library", len(step.state.library.record_ids)),
        ("max_feedback", len(step.state.feedback)),
        ("max_pending", len(step.state.pending)),
    ):
        counts[name] = max(counts[name], size)


def iter_verified_transitions(
    directory: Path, manifest_sha256: str
) -> Iterator[tuple[str, ForecastTransition]]:
    """Yield exact saved states/laws after complete chain validation, with fresh hashes.

    A second bounded streaming pass prevents callers from receiving a trusted
    prefix of a later-invalid archive. Each yielded file is hash-checked again.
    """
    manifest = verify_model_archive(directory, manifest_sha256)
    rows = manifest["files"]
    if not isinstance(rows, list):
        raise IntegrityError("verified archive registry is not a list")
    for raw in rows:
        row = object_json(raw)
        payload = object_json(
            read_verified_json(
                directory / _text(row["name"]), _text(row["sha256"]), maximum_bytes=2 * _CHECKPOINT_LIMIT
            )
        )
        yield _text(row["variant"]), decode_transition(object_json(payload["transition"]))
