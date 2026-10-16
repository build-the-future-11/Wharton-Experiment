# Independent semantic artifact review

The earlier readback receipt remains intact. Subsequent independent review found
that individual frame hashes were checked but the tape metadata and complete
variant structure were not reconstructed, and that manifest/header attempt identity
could disagree. These were genuine verification gaps, not scientific outcomes.

The corrected verifier recomputes the tape hash from exact transform, scope,
weekly-origin count, final-state identities and verified frame hashes. Numeric
frames remain streamed; only small origin metadata is retained. It requires unique
ordered anchor/variant keys, exactly the declared six variants per retained origin,
and common origin identity and decision time. Manifest/header experiment identity,
frozen header axes, exact forecast record fields and explicit numeric array schema
are enforced. Boolean and string values cannot be silently coerced into forecast
arrays.

Twenty-two readback tests passed with warnings as errors in 16.96 seconds. The
strongest new negative test rewrites a duplicated frame, index, tape hash, header,
cell provenance links and final manifest consistently; the invalid variant
structure is still rejected. Other tests reject missing/reordered variants,
different origin identities or times, changed tape metadata, coerced array values
and axis relabeling. Strict mypy passes. Ruff passes excluding import ordering,
which the parent owns for the final global verification; format checking passes.

This is generated engineering evidence. The exact source snapshot, logs, prior
diagnostics, remaining scope and the one subsequent type-compatible list copy are
recorded in `readback_semantic_review_verification.json`. The parent's combined
receipt will supersede these local snapshots after global import sorting and
complete-suite verification. No scientific data or protected outcomes were read.
