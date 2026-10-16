"""Tiny archive fixtures exercise package boundaries without building or installing."""

import base64
import csv
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from frem_s10.errors import IntegrityError
from frem_s10.runtime.packaging import _audit_archive_bytes, _inspect, _run
from frem_s10.runtime.provenance import PACKAGE

METADATA = (
    b"Metadata-Version: 2.4\nName: frem-s10\nVersion: 0.1.0\n"
    b"Summary: Generated package fixture\nRequires-Python: >=3.12,<3.13\n"
    b"Description-Content-Type: text/markdown\nRequires-Dist: numpy==2.5.4\n\n"
    b"Generated packaging boundary fixture.\n"
)
ENTRY_POINTS = b"[console_scripts]\ns10-research = frem_s10.cli:main\n"


def _fixture(root: Path) -> tuple[Path, Path, dict[str, bytes], dict[str, bytes]]:
    package = root / PACKAGE
    source = package / "src/frem_s10"
    source.mkdir(parents=True)
    payloads = {
        "frem_s10/__init__.py": b'"""Generated fixture; never imported."""\n',
        "frem_s10/cli.py": b"def main():\n    return 0\n",
        "frem_s10/science.json": b'{"scope":"GENERATED_ENGINEERING"}\n',
        "frem_s10/py.typed": b"",
    }
    for name, payload in payloads.items():
        (package / "src" / name).write_bytes(payload)
    (package / "README.md").write_text("Generated packaging boundary fixture.\n")
    (package / "pyproject.toml").write_text(
        '[project]\nname="frem-s10"\nversion="0.1.0"\n'
        'description="Generated package fixture"\nreadme="README.md"\n'
        'dependencies=["numpy==2.5.4"]\nrequires-python=">=3.12,<3.13"\n'
        '[project.scripts]\ns10-research="frem_s10.cli:main"\n'
    )
    (package / "MANIFEST.in").write_text("recursive-include tests *.py\ninclude uv.lock\n")
    (package / "uv.lock").write_text("version = 1\n")
    (package / "tests").mkdir()
    (package / "tests/test_fixture.py").write_text("# generated, never executed\n")
    wheel = payloads | {
        "frem_s10-0.1.0.dist-info/METADATA": METADATA,
        "frem_s10-0.1.0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: setuptools (84.0.0)\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
        ),
        "frem_s10-0.1.0.dist-info/entry_points.txt": ENTRY_POINTS,
        "frem_s10-0.1.0.dist-info/top_level.txt": b"frem_s10\n",
    }
    sdist = {f"frem_s10-0.1.0/src/{name}": payload for name, payload in payloads.items()}
    sdist.update(
        {
            "frem_s10-0.1.0/README.md": (package / "README.md").read_bytes(),
            "frem_s10-0.1.0/pyproject.toml": (package / "pyproject.toml").read_bytes(),
            "frem_s10-0.1.0/PKG-INFO": METADATA,
            "frem_s10-0.1.0/setup.cfg": b"[egg_info]\ntag_build = \ntag_date = 0\n\n",
        }
    )
    for name in ("MANIFEST.in", "uv.lock", "tests/test_fixture.py"):
        sdist[f"frem_s10-0.1.0/{name}"] = (package / name).read_bytes()
    egg = "frem_s10-0.1.0/src/frem_s10.egg-info/"
    sdist.update(
        {
            egg + "PKG-INFO": METADATA,
            egg + "dependency_links.txt": b"\n",
            egg + "entry_points.txt": ENTRY_POINTS,
            egg + "requires.txt": b"numpy==2.5.4\n",
            egg + "top_level.txt": b"frem_s10\n",
        }
    )
    manifest = {name.removeprefix("frem_s10-0.1.0/") for name in sdist} - {"PKG-INFO", "setup.cfg"}
    manifest.add("src/frem_s10.egg-info/SOURCES.txt")
    sdist[egg + "SOURCES.txt"] = ("\n".join(sorted(manifest)) + "\n").encode()
    return root / "frem_s10-0.1.0-py3-none-any.whl", root / "frem_s10-0.1.0.tar.gz", wheel, sdist


def _archives(wheel: Path, sdist: Path, wheels: dict[str, bytes], sources: dict[str, bytes]) -> None:
    record_name = "frem_s10-0.1.0.dist-info/RECORD"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    for name, payload in wheels.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        writer.writerow((name, f"sha256={digest}", str(len(payload))))
    writer.writerow((record_name, "", ""))
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in wheels.items():
            archive.writestr(name, payload)
        archive.writestr(record_name, buffer.getvalue())
    with tarfile.open(sdist, "w:gz") as archive:
        for name, payload in sources.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_exact_generated_archive_payloads_pass(tmp_path: Path) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    _archives(wheel, sdist, wheels, sources)
    assert _inspect(wheel, sdist, tmp_path)["status"] == "PASS"


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_package_member_cannot_be_dropped_or_changed(tmp_path: Path, damage: str) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    if damage == "missing":
        del wheels["frem_s10/science.json"]
    else:
        wheels["frem_s10/science.json"] = b'{"changed":true}'
    _archives(wheel, sdist, wheels, sources)
    with pytest.raises(IntegrityError):
        _inspect(wheel, sdist, tmp_path)


@pytest.mark.parametrize(
    "member", ["sitecustomize.py", "injected.pth", "frem_s10-0.1.0.data/purelib/injected.py"]
)
def test_unbound_installable_wheel_members_rejected(tmp_path: Path, member: str) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    wheels[member] = b"# inert generated injection fixture, never executed\n"
    _archives(wheel, sdist, wheels, sources)
    with pytest.raises(IntegrityError):
        _inspect(wheel, sdist, tmp_path)


@pytest.mark.parametrize("member", ["frem_s10-0.1.0/retained_prices.json", "alien-root/README.md"])
def test_unbound_sdist_payload_cannot_distribute_arbitrary_data(tmp_path: Path, member: str) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    sources[member] = b'{"fixture":"no financial observations"}\n'
    _archives(wheel, sdist, wheels, sources)
    with pytest.raises(IntegrityError):
        _inspect(wheel, sdist, tmp_path)


def test_source_symlink_cannot_supply_verified_package_bytes(tmp_path: Path) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    source = tmp_path / PACKAGE / "src/frem_s10/science.json"
    external = tmp_path / "unbound.json"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)
    _archives(wheel, sdist, wheels, sources)
    with pytest.raises(IntegrityError):
        _inspect(wheel, sdist, tmp_path)


def test_current_but_unbound_source_data_cannot_enter_distribution(tmp_path: Path) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    payload = b'{"fixture":"not a registered scientific configuration"}\n'
    name = "frem_s10/retained_prices.json"
    (tmp_path / PACKAGE / "src" / name).write_bytes(payload)
    wheels[name] = payload
    sources[f"frem_s10-0.1.0/src/{name}"] = payload
    _archives(wheel, sdist, wheels, sources)
    with pytest.raises(IntegrityError):
        _inspect(wheel, sdist, tmp_path)


def test_wheel_entry_point_cannot_differ_from_frozen_project(tmp_path: Path) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    wheels["frem_s10-0.1.0.dist-info/entry_points.txt"] = (
        b"[console_scripts]\ns10-research = frem_s10.runtime.worker:main\n"
    )
    _archives(wheel, sdist, wheels, sources)
    with pytest.raises(IntegrityError):
        _inspect(wheel, sdist, tmp_path)


def test_source_distribution_build_contract_cannot_change(tmp_path: Path) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    sources["frem_s10-0.1.0/pyproject.toml"] = b'[build-system]\nbuild-backend="unbound.backend"\n'
    _archives(wheel, sdist, wheels, sources)
    with pytest.raises(IntegrityError):
        _inspect(wheel, sdist, tmp_path)


def test_wheel_record_must_match_payload_hash_and_size(tmp_path: Path) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    _archives(wheel, sdist, wheels, sources)
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    name = "frem_s10-0.1.0.dist-info/RECORD"
    contents[name] = contents[name].replace(b",sha256=", b",sha256=changed", 1)
    with zipfile.ZipFile(wheel, "w") as archive:
        for member, data in contents.items():
            archive.writestr(member, data)
    with pytest.raises(IntegrityError, match="RECORD"):
        _inspect(wheel, sdist, tmp_path)


def test_sdist_cannot_drop_a_frozen_test(tmp_path: Path) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    del sources["frem_s10-0.1.0/tests/test_fixture.py"]
    _archives(wheel, sdist, wheels, sources)
    with pytest.raises(IntegrityError, match="inventory"):
        _inspect(wheel, sdist, tmp_path)


def test_command_timeout_preserves_actual_log_and_failed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(("fixture",), 120)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(IntegrityError, match="command failed"):
        _run("fixture", ("fixture",), output=tmp_path, cwd=tmp_path)
    receipt = json.loads((tmp_path / "fixture.json").read_text())
    assert receipt["exit_code"] == -1
    assert "TimeoutExpired" in receipt["failure"]
    assert receipt["log_sha256"] == hashlib.sha256((tmp_path / "fixture.log").read_bytes()).hexdigest()
    assert "TimeoutExpired" in (tmp_path / "fixture.log").read_text()


def test_compact_audit_streams_archive_identities_and_detects_later_mutation(tmp_path: Path) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    _archives(wheel, sdist, wheels, sources)
    receipt = _inspect(wheel, sdist, tmp_path)
    (tmp_path / "dist").mkdir()
    retained_wheel = wheel.rename(tmp_path / "dist" / wheel.name)
    sdist.rename(tmp_path / "dist" / sdist.name)
    _audit_archive_bytes(tmp_path, receipt, "frem_s10-0.1.0")
    with retained_wheel.open("ab") as stream:
        stream.write(b"changed after checked build")
    with pytest.raises(IntegrityError, match="bytes changed"):
        _audit_archive_bytes(tmp_path, receipt, "frem_s10-0.1.0")


@pytest.mark.parametrize("archive_type", ["wheel", "sdist"])
def test_archive_symlinks_rejected(tmp_path: Path, archive_type: str) -> None:
    wheel, sdist, wheels, sources = _fixture(tmp_path)
    _archives(wheel, sdist, wheels, sources)
    if archive_type == "wheel":
        with zipfile.ZipFile(wheel, "a") as archive:
            member = zipfile.ZipInfo("frem_s10/link")
            member.create_system = 3
            member.external_attr = 0o120777 << 16
            archive.writestr(member, "science.json")
    else:
        with tarfile.open(sdist, "w:gz") as tar:
            link = tarfile.TarInfo("frem_s10-0.1.0/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "src/frem_s10/science.json"
            tar.addfile(link)
    with pytest.raises(IntegrityError):
        _inspect(wheel, sdist, tmp_path)
