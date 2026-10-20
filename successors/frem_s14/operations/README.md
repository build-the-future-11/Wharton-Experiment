# Bounded stage execution

`run_stage.py` is an executable, dependency-free POSIX process runner. It records
startup before launching, streams stdout/stderr to an exact-byte log, emits durable
progress events, and saves a terminal receipt for success, command failure, launch
failure, interruption and timeout. Each invocation requires a new output directory.
Atomic JSON publication cannot replace earlier evidence.

The effective deadline is the smaller of the stage allowance and an optional
absolute `time.monotonic()` deadline on the same machine and boot. Reaching it
terminates only the process group this runner created. Background children in
that group are also terminated when the leader exits. No timeout grace period
extends the computation allowance. The receipt separately reports cleanup and
receipt time. OS scheduling and filesystem/console I/O are not hard real-time
guarantees, and a program that deliberately escapes its process group requires
OS/container isolation. This tool is for the declared trusted research commands.

Example, from the repository root, for an engineering preflight:

```sh
rtk proxy env PYTHONPATH=successors/frem_s14/src .venv/bin/python -B \
  successors/frem_s14/operations/run_stage.py \
  --output /private/tmp/s14-preflight-new-attempt --max-seconds 30 -- \
  .venv/bin/python -B -m frem_s14 preflight \
  --output /private/tmp/s14-preflight-new-receipt.json
```

For Python orchestration, call `run_stage(argv, new_directory, stage_seconds,
deadline=total_deadline)`. Use it for installation too. Pass the existing declared
stage and total limits; this tool grants no additional compute, repetitions or
market access. A zero exit code is only a process result: tests, qualification,
intake, numerical readback and scientific acceptance remain separate gates.

Run its real child-process tests without numerical imports:

```sh
rtk proxy .venv/bin/python -B -W error -m unittest discover \
  -s successors/frem_s14/operations -p 'test_*.py' -v
```

The running Kaggle version 359955736 and its 0.1.3 wheel/notebook remain frozen.
This directory is outside that release and is not retroactively installed in the
active run. A future notebook integration must preserve its predecessor, receive
a new notebook/release identity, undergo exact-launch qualification and avoid
repeating valid completed market cells. No model performance improvement is
claimed from these operational tests.
