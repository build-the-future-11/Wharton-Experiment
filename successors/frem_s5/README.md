# S5-R1 contextual expert-routing pilot

Implementation and fixed retained-evidence experiment for the additive S5 hypothesis.
Read `research/frem_s5_20261013/REPORT.md`, `specification.json`, and `UPGRADE_BACKLOG.md`.

The completed run is `outputs/frem_s5_20261013_attempt2`; attempt1 failed before scientific outputs and is retained. Do not rerun to seek a positive result or mutate frozen sources. There were no new base-model fits or protected outcomes accessed. Fifteen targeted tests passed; an arithmetic audit verified111 frozen identities,7128 score rows and312 ledgers. The fresh-trained S5 matrix remains unexecuted.

Inspection/reconstruction commands (use a new audit output path):

```sh
rtk proxy python3 -m unittest discover -s tests/s5 -v
rtk proxy python3 -m successors.frem_s5.audit --run outputs/frem_s5_20261013_attempt2 --output /tmp/s5-new-audit.json
```

`router.py` implements the mechanism and strict feedback clock; `experiment.py` handles retained inputs, chronological selection, exact mixture scores, engineering mechanisms and benchmark accounting; `run.py` enforces immutable inputs and lightweight resource limits; `audit.py` independently reconstructs arithmetic using saved weights and holdings. It is not independent retraining.
