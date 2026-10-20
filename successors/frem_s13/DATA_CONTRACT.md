# Input and chronology contract

CSV schema: `timestamp,asset,available_at,return,feature_*`. Feature columns are
sorted lexicographically; their order is therefore reproducible. `return` is the
backward interval's **simple total return**, including distributions and terminal
outcomes, expressed as a decimal. It is the first model channel. Other features
must be observed at or before `available_at`. All timestamps need explicit UTC
offsets. `available_at` cannot precede `timestamp`.

Every retained date must contain every declared asset exactly once. Duplicate,
missing, nonfinite, unbalanced and below-minus-one return rows are rejected.
No imputation, current-universe reconstruction, yield substitution or download
occurs. A supplier must normalize differing exchange calendars and disclose
universe selection before using this adapter. Balanced panels restrict the
claim; excluding dead securities is not a valid way to manufacture one.

`panel.csv.manifest.json` contains exactly these keys:

| Key | Required value or meaning |
| --- | --- |
| `schema` | Integer `1` |
| `sha256` | Actual SHA-256 of the entire CSV |
| `source` | Named provider/export and relevant source version |
| `retrieved_at` | Actual timezone-qualified retrieval timestamp |
| `license` | Actual terms or rights reference supplied with the data |
| `universe_policy` | Actual historical selection/membership policy and limits |
| `return_definition` | `simple_total_return_including_terminal_outcomes` |
| `feature_availability` | `available_at_bounds_all_features` |
| `purpose` | `development` |

The manifest establishes byte identity and records assertions; it cannot prove
the truth of a license or point-in-time feature lineage. No example manifest
with invented rights or eligibility is provided.

Rows on or after `protected_start` are discarded before any return/feature
number is parsed. Rows whose availability is on or after that boundary are
also discarded. The entire file is hashed as opaque bytes. The runner only
constructs development windows. Previously observed periods remain development
even if a user changes the string called `protected_start`.

The decision for date t occurs at the latest availability of its asset rows.
Decision times must increase strictly. Context consists of L trailing rows.
The outcome is `product(1 + return[t+1:t+h+1]) - 1`; maturity is the latest
availability at t+h. The fixed universe makes date-wise clustering explicit.

Chronological train/validation/evaluation origins use the declared fractions
and embargo. Training labels must mature strictly before the first validation
decision; validation labels strictly before the first evaluation decision.
Median/MAD normalization is fitted only on rows used as training contexts.
The target scale is the training-label standard deviation, floored at 1e-4.
Future target windows are used only in the training objective. The evaluation
tail in this runner is an exploratory development tail, not a protected test.
