"""Offline target-wheel selection checks without building or installing anything."""

import platform
from pathlib import Path

import pytest

from frem_s10.errors import IntegrityError
from frem_s10.runtime import packaging
from frem_s10.runtime.provenance import PACKAGE


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_linux_wheel_is_selected_from_exact_existing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: architecture)
    project = tmp_path / PACKAGE
    project.mkdir(parents=True)
    filename = f"numpy-2.5.4-cp312-cp312-manylinux_2_27_{architecture}.manylinux_2_28_{architecture}.whl"
    (project / "uv.lock").write_text(
        '[[package]]\nname="numpy"\nversion="2.5.4"\nwheels=[{url="https://invalid.test/'
        + filename
        + '",hash="sha256:'
        + "a" * 64
        + '"}]\n'
    )
    path, identity = packaging.dependency_wheel(tmp_path)
    assert path == tmp_path / "research/frem_s10_wheelhouse" / filename
    assert identity == "a" * 64
    assert not path.exists()  # Selection never downloads or installs a dependency.


def test_macos_wheel_identity_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert packaging.dependency_wheel(tmp_path) == (tmp_path / packaging.NUMPY_WHEEL, packaging.NUMPY_SHA256)


def test_missing_linux_wheel_and_unsupported_architecture_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    project = tmp_path / PACKAGE
    project.mkdir(parents=True)
    (project / "uv.lock").write_text('[[package]]\nname="numpy"\nversion="2.5.4"\nwheels=[]\n')
    with pytest.raises(IntegrityError, match="unique"):
        packaging.dependency_wheel(tmp_path)
    monkeypatch.setattr(platform, "machine", lambda: "unsupported")
    with pytest.raises(IntegrityError, match="selection"):
        packaging.dependency_wheel(tmp_path)


def test_linux_build_stages_only_frozen_inputs_inside_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    package = source / PACKAGE
    package.mkdir(parents=True)
    output.mkdir()
    item = package / "pyproject.toml"
    item.write_text("frozen-input")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(packaging, "expected_code_files", lambda root: (item,))
    staged = packaging._build_directory(source, output)
    assert staged.is_relative_to(output)
    assert (staged / "pyproject.toml").read_bytes() == item.read_bytes()
