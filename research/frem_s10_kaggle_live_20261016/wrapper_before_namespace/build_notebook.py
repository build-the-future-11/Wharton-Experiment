"""Build the self-contained notebook from the sealed source and reviewed helper."""
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEALED = HERE.parents[1] / "research/frem_s10_kaggle_20261016"


def build():
    helper = (HERE / "runner.py").read_bytes()
    bundle = (SEALED / "source_bundle.zip").read_bytes()
    helper_sha = hashlib.sha256(helper).hexdigest()
    bundle_sha = hashlib.sha256(bundle).hexdigest()
    assert bundle_sha == "a03c0e1886ece933427b554c555d2e914dff7f9d8df10e5c6caca94a74b2237a"
    cells = []

    def md(title, body):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [title + "\n\n" + body]})

    def code(body):
        ast.parse(body)
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": body.splitlines(keepends=True)})

    md("# 1. Source retrieval and Wharton / FREM-S10 identity", "The maintained source ZIP is embedded in this notebook and verified in section 4; no network source retrieval is needed. Candidate `6cc8dde2716cfd6085a31d28ee0f9a4cb22adbfa1e6619918a0e6eaa3071a031`. Scientific status remains **NOT_EVALUATED**. Production, market and confirmatory work remain **BLOCKED**. No point-in-time panel, new scientific freeze, protected-outcome access, GPU backend, or profitability claim. Existing negative evidence is unchanged. Use a private notebook, accelerator **None**. Internet **On** is needed only for the isolated environment setup; it can be disabled afterward; do not add private data or credentials.")
    md("# 2. Resolved configuration, manifests and deadline", "Absolute deadline: **16 October 2026, 22:00 IST / 16:30 UTC**. Two independent ceilings constrain this attempt: 600 seconds of conservatively reserved child-command time including cleanup, and 600 seconds of cumulative wrapper wall allowance including a 60-second notebook setup reservation and bounded exports. No runtime promise is made. The model retains its existing cumulative 60-second budget. Source setup has a watchdog; commands and full-state export have deadline watchdogs. Failure stops the sequence. This is not a GPU-hour estimate. `probe` is the default. `smoke` requires a separately downloaded, verified persistence receipt; never set it merely because a file exists in this session.")
    code('''MODE = "probe"  # "probe", "smoke", or "status"; no automatic promotion
ACCELERATOR = "None"
RETRIEVAL_RECEIPT = None  # Paste the exact local tools/kaggle/runner.py attest JSON after retrieval.
RESTORE_ARCHIVE = None  # Optional /kaggle/input/.../state-<id>.zip from this exact campaign.
RESTORE_ARCHIVE_SHA256 = None  # Exact externally recorded digest; never calculate it as approval here.
''')
    md("# 3. Dependencies and exact CPU environment", "The maintained path requires Python **3.12** and NumPy **2.5.4**. Package metadata is inspected without importing NumPy/PyTorch. The next cell installs pinned uv into a private tool directory, creates isolated Python 3.12, and installs hash-pinned NumPy. Kaggle kernel packages remain unchanged. Setup has a separate 300-second ceiling and retained logs. A mismatch is retained as a blocked preflight; do not weaken the version gate or substitute a framework. The sealed requirements file lists accepted NumPy hashes but is not a complete production environment. Exact interpreter and installed NumPy artifacts are bound before smoke execution.")
    code('''import importlib.metadata, json, platform, sys
def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
print(json.dumps({"python": sys.version, "system": platform.system(), "machine": platform.machine(),
                  "numpy": distribution_version("numpy"), "torch": distribution_version("torch"),
                  "torch_required": False, "numerical_imports_performed": False}, indent=2))
''')
    runtime = (HERE / "bootstrap.py").read_bytes()
    code("import base64, hashlib\n" +
         "RUNTIME_SOURCE = base64.b64decode(" + repr(base64.b64encode(runtime).decode()) + ")\n" +
         "assert hashlib.sha256(RUNTIME_SOURCE).hexdigest() == " + repr(hashlib.sha256(runtime).hexdigest()) + "\n" +
         "runtime_ns = {'__name__': 'kaggle_runtime'}\nexec(compile(RUNTIME_SOURCE, 'bootstrap.py', 'exec'), runtime_ns)\n" +
         "from pathlib import Path\n" +
         "REQUIREMENTS = " + repr((HERE / "requirements-cpu.txt").read_text()) + "\n" +
         "RUNTIME_PYTHON = Path(sys.executable) if MODE == 'status' else runtime_ns['prepare'](Path('/kaggle/working/frem_s10_runtime'), REQUIREMENTS)\n" +
         "print('Isolated execution interpreter:', RUNTIME_PYTHON)\n")
    md("# 4. Data integrity and verified source setup", "Only runtime-generated data is used: no historical market values, external datasets or protected outcomes. The maintained CLI hashes generation parameters, seeded data and fixed burn-in transform. The original source ZIP is embedded below and checked by SHA-256 before extraction. Every file and the inventory itself are pinned; absolute/traversal paths, links, duplicates and oversized archives are refused. Existing source must match exactly. Paths use only `/kaggle/working` or an explicitly attached input archive. Source is imported through the maintained `python -m frem_s10` CLI in fresh subprocesses; no alternate numerical implementation is embedded.")
    bootstrap = '''import base64, hashlib, importlib.util, signal, uuid
from datetime import datetime, timezone
from pathlib import Path
def setup_timeout(_signal, _frame):
    raise TimeoutError("bounded source setup/deadline expired")
deadline_remaining = (datetime(2026, 10, 16, 16, 30, tzinfo=timezone.utc) - datetime.now(timezone.utc)).total_seconds() - 10
if MODE != "status" and deadline_remaining <= 0:
    raise RuntimeError("tonight deadline expired; no execution")
signal.signal(signal.SIGALRM, setup_timeout)
signal.setitimer(signal.ITIMER_REAL, min(60, deadline_remaining) if MODE != "status" else 60)
ROOT = Path("/kaggle/working/frem_s10_kaggle_fixed_20261016")
ROOT.mkdir(parents=True, exist_ok=True)
BOOTSTRAP_RECORD = ROOT / ("bootstrap-" + uuid.uuid4().hex + ".json")
with BOOTSTRAP_RECORD.open("x") as stream:
    json.dump({"reserved_seconds": 60}, stream)
def exact_file(path, payload, expected):
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError("embedded payload hash mismatch")
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise RuntimeError("existing setup differs; preserve and stop")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
HELPER = ROOT / "tonight_runner.py"
BUNDLE = ROOT / "source_bundle.zip"
exact_file(HELPER, base64.b64decode(HELPER_B64, validate=True), HELPER_SHA256)
exact_file(BUNDLE, base64.b64decode(SOURCE_B64, validate=True), SOURCE_SHA256)
spec = importlib.util.spec_from_file_location("s10_tonight_helper", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
if MODE not in ("probe", "smoke", "status"):
    raise RuntimeError("unknown mode")
if MODE != "status":
    helper.deadline_check()
    helper.cpu_only(ACCELERATOR)
SOURCE = ROOT / "source"
helper.extract_verified(BUNDLE, SOURCE, SOURCE_SHA256, inventory_name="SOURCE_INVENTORY.json")
SESSION = ROOT / "session"
if RESTORE_ARCHIVE is not None:
    if not isinstance(RESTORE_ARCHIVE_SHA256, str) or len(RESTORE_ARCHIVE_SHA256) != 64:
        raise RuntimeError("external full-state archive digest required")
    if SESSION.exists():
        raise RuntimeError("restore requires a fresh session directory; never merge evidence")
    helper.extract_verified(Path(RESTORE_ARCHIVE), SESSION, RESTORE_ARCHIVE_SHA256,
                            inventory_name="STATE_INVENTORY.json")
helper.verify_source(SOURCE)
signal.setitimer(signal.ITIMER_REAL, 0)
print({"source_bundle_sha256": SOURCE_SHA256, "runner_sha256": HELPER_SHA256,
       "candidate": helper.CANDIDATE_IDENTITY, "mode": MODE})
'''
    code("HELPER_SHA256 = " + repr(helper_sha) + "\nSOURCE_SHA256 = " + repr(bundle_sha)
         + "\nHELPER_B64 = " + repr(base64.b64encode(helper).decode())
         + "\nSOURCE_B64 = " + repr(base64.b64encode(bundle).decode()) + "\n" + bootstrap)
    md("# 5. Small CPU preflight and persistence probe", "In `probe` mode the helper runs only maintained metadata preflight, exact runtime-byte identity and generated-manifest planning. It writes a unique nonce, identities and observations to `session/persistence-probe.json`, plus a full-state ZIP and durable command logs. It does not import numerical libraries or fit the model. Save the private notebook version, retrieve the probe outside Kaggle, and verify those bytes before promotion. A blocked environment can still produce useful persistence evidence but cannot authorize smoke.")
    code('''import subprocess, os
def invoke(mode, receipt=None):
    argv = [str(RUNTIME_PYTHON), "-u", "-B", str(HELPER), mode, "--source", str(SOURCE),
            "--session", str(SESSION), "--accelerator", ACCELERATOR]
    if receipt is not None:
        argv += ["--receipt", str(receipt)]
    # The helper owns child process groups. This additional watchdog bounds helper startup.
    allowed = min(600, (datetime(2026, 10, 16, 16, 30, tzinfo=timezone.utc) - datetime.now(timezone.utc)).total_seconds() - 10) if mode != "status" else 10
    if allowed <= 0:
        raise RuntimeError("deadline expired")
    child_env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        child_env.pop(key, None)
    child_env["PYTHONNOUSERSITE"] = "1"
    child = subprocess.Popen(argv, start_new_session=True, env=child_env)
    try:
        result = child.wait(timeout=allowed)
    except BaseException:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        raise
    if result:
        raise subprocess.CalledProcessError(result, argv)
if MODE == "probe":
    invoke("probe")
''')
    md("# 6. GPU preflight and profiling: not applicable", "S10 uses NumPy float64 on CPU. CUDA/MPS are unsupported; no fallback, AMP, gradient scaler, optimizer, scheduler, training epochs, neural batches, compilation or multi-GPU execution is claimed. Accelerator None is required and visible NVIDIA devices cause refusal. A metadata observation of PyTorch/CUDA is not a GPU test. **GPU checks: NOT_RUN; GPU pass: false.** No GPU profiling or unmeasured speed claim. Do not launch a GPU session for this notebook.")
    md("# 7. Credentials and the declared generated experiment", "**No secrets or API credentials are needed in this notebook.** Do not paste tokens, mount credential files or dump arbitrary environment variables. Private output retrieval uses the separate authenticated browser/CLI outside the notebook. The sealed `configs/generated_smoke.json` declares primary variant, seed 11, 12 origins, generated dataset `s10-generated-smoke-data-v1`, one thread, float64, checkpoints after every origin and a cumulative 60-second model budget. No field is overridden here. The 72-origin development candidate and 192-ledger scientific design are **not launched**; they remain unfrozen and gated.")
    code('''print((SOURCE / "configs/generated_smoke.json").read_text())
''')
    md("# 8. Persisted-output retrieval and resume gate", "Run the probe once in one interactive CPU session. Record the displayed `probe_sha256`. Use **Quick Save → Advanced Settings → Save output for this version**; the observed default Never save output is insufficient. Confirm the saved version, then download `persistence-probe.json` through that exact private version's Output tab. Current official CLI also documents authenticated `kaggle kernels output OWNER/SLUG/VERSION -p NEW_DIRECTORY`; verify installed CLI support before using it. Locally run `tools/kaggle/runner.py attest --downloaded-probe PATH --expected-probe-sha256 DISPLAYED_DIGEST --kernel-version OWNER/SLUG/VERSION --output NEW_RECEIPT.json --ack-retrieved`. Paste its output into `RETRIEVAL_RECEIPT`, then set MODE to `smoke`. The receipt is an explicit operator attestation bound to exact probe/source/runner/runtime bytes—not a cryptographic Kaggle signature. Do not fabricate it inside this notebook. One receipt authorizes one bounded engineering attempt; parent campaign coordination is required across notebooks. Each repeated bootstrap is conservatively charged again, even if setup files already match.")
    code('''RECEIPT_PATH = None
if MODE == "smoke":
    if not isinstance(RETRIEVAL_RECEIPT, dict):
        raise RuntimeError("STOP: externally retrieved and locally verified persistence receipt required")
    helper.validate_receipt(RETRIEVAL_RECEIPT, helper_sha=HELPER_SHA256)
    RECEIPT_PATH = ROOT / "external-retrieval-receipt.json"
    helper.publish(RECEIPT_PATH, RETRIEVAL_RECEIPT)
''')
    md("# 9. Bounded execution, interruption and full-state resume", "The helper first requires the same exact CPU runtime as the retrieved probe. Each numerical command receives a fresh three-sample `check-host` PASS with the unchanged 512 MiB/stable-swap/cgroup ancestry rules. Unsupported Kaggle cgroups block execution. It runs `smoke --stop-after 3 --execute` (expected exit 130), then `smoke-resume --execute`. Every committed origin contains all causal model/pending-feedback state; data regenerates from its private seed and is hash-checked. No validation-selected best checkpoint exists. Unknown hard-kill time or incomplete command receipts stop recovery; do not delete session records, relax admission or reset budgets.")
    code('''if MODE == "smoke":
    invoke("smoke", RECEIPT_PATH)
''')
    md("# 10. Analysis, frozen evaluation and export", "After resume, the helper calls `smoke-evaluate`, validates the completed cell using `smoke-resume`, then calls `smoke-export`. Evaluation reconstructs saved scores without fitting; exported artifacts retain identity and logs. A file or 12 checkpoint names alone never proves completion. Stage errors stop the sequence. `ENGINEERING_RECEIPT.json` records model stages with final export pending. Only `FINALIZATION_RECEIPT.json`, written after successful full-state export, reports generated CPU engineering completion. Neither establishes Linux production qualification, platform session termination or scientific support.")
    md("# 11. Live status, persistence and full-state restore", "Actual child stdout is streamed live while durable `session/control/command-*.log` files retain all bytes. Inspect immutable command receipts, `session/smoke/metrics-*.jsonl`, checkpoint/status histories and `exports/state-*.json`. Intermediate ZIPs are diagnostic snapshots with an active invocation and cannot be resumed; use the terminal snapshot whose sidecar says resumable=true. Each exported state ZIP includes the complete saved session and exact inventory; its external hash is printed. To restore, attach an externally retrieved ZIP as a private input, put its path and externally recorded hash in section 2, and start with a fresh session directory. Restoration never merges state. It preserves command reservations and exact source/runtime binding; unfamiliar or incomplete state fails closed. Cross-machine continuation is not promised; changed interpreter/NumPy/OS bytes refuse resume. No ETA or synthetic dashboard is shown.")
    code('''if MODE == "status":
    invoke("status")
for path in sorted((ROOT / "exports").glob("state-*.json")):
    print(path.name, path.read_text())
print("Output root:", ROOT)
''')
    md("# 12. Save, retrieve, verify and explicitly stop", "**Save & Run All starts a fresh clean execution; it is not a passive checkpoint save.** Quick Save skips execution; confirm its output-saving option and verify the exact saved version's output inventory. Interactive persistence is best effort. Retrieve the final state ZIP, its JSON digest, engineering receipt, logs and smoke export from the private version; hash-check locally and retain the source/notebook identity. Use explicit Kaggle **Stop Session/Cancel** for this notebook's interactive and batch sessions, then verify no related active session remains. Closing the tab, Python exit, this cell, or a saved receipt does not prove platform termination. Finish before 22:00 IST; this notebook does not autonomously control account sessions. Official guidance: [versions](https://www.kaggle.com/docs/notebooks), [CLI output](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [session management](https://www.kaggle.com/general/197557).")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "s10": {"source_bundle_sha256": bundle_sha, "runner_sha256": helper_sha,
                "scope": "generated CPU engineering only", "deadline": "2026-10-16T16:30:00Z"}},
        "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"s10-tonight-{index:02}"
    target = HERE / "FREM_S10_KAGGLE_CPU.ipynb"
    target.write_text(json.dumps(notebook, indent=1) + "\n")
    identity = {"status": "AUTHORED_NOT_EXECUTED_ON_KAGGLE", "source_bundle_sha256": bundle_sha,
                "runner_sha256": helper_sha, "notebook_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "candidate_identity": "6cc8dde2716cfd6085a31d28ee0f9a4cb22adbfa1e6619918a0e6eaa3071a031",
                "sections": 12, "cloud_execution": False, "scientific_status": "NOT_EVALUATED"}
    (HERE / "NOTEBOOK_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    build()
