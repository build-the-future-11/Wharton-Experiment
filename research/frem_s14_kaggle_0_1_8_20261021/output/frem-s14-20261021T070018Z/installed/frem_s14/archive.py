"""Streaming, anchored readback of a downloaded Kaggle ZIP; never extracts files."""

import argparse
import io
import stat
import zipfile
from hashlib import sha256
from pathlib import PurePosixPath

from .runtime import decode_json, digest, identity, write_json


def inventory(archive):
    files = {}
    seen = set()
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or name in seen
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or str(path) != name.rstrip("/")
            or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)
        ):
            raise ValueError("unsafe, duplicate or special ZIP member")
        seen.add(name)
        if info.is_dir():
            continue
        h, count = sha256(), 0
        with archive.open(info) as stream:
            for chunk in iter(lambda: stream.read(512 * 1024), b""):
                count += len(chunk)
                h.update(chunk)
        if count != info.file_size:
            raise ValueError("ZIP member size differs")
        files[name] = {"sha256": h.hexdigest(), "bytes": count}
    return files


def small_read(archive, name, maximum=16 * 1024 * 1024):
    if archive.getinfo(name).file_size > maximum:
        raise ValueError("metadata/release member exceeds readback memory bound")
    return archive.read(name)


def verify_download(download, release, expected_release_sha256):
    """Integrity/provenance only: numeric semantics require the separate verifier.

    The expected release digest comes from a trusted local release receipt, never
    from the downloaded archive's own claims. Partial/failed runs are retained.
    """
    if digest(release) != expected_release_sha256:
        raise ValueError("trusted release checksum differs")
    with zipfile.ZipFile(release) as source:
        expected = decode_json(small_read(source, "PAYLOAD.json"))
        actual = inventory(source)
        if {k: v for k, v in actual.items() if k != "PAYLOAD.json"} != expected:
            raise ValueError("trusted release inventory differs")
        wheels = [name for name in expected if name.endswith(".whl")]
        if len(wheels) != 1:
            raise ValueError("exactly one wheel required")
        wheel_name = wheels[0]
        with zipfile.ZipFile(io.BytesIO(small_read(source, wheel_name))) as wheel:
            wheel_files = inventory(wheel)
            code = {
                k: v
                for k, v in wheel_files.items()
                if k.startswith("frem_s14/") and k.endswith(".py")
            }
        if not code:
            raise ValueError("wheel contains no S14 source")
    with zipfile.ZipFile(download) as output:
        files = inventory(output)  # Streams all bytes and validates every CRC.
        roots = {PurePosixPath(name).parts[0] for name in files}
        if len(roots) != 1:
            raise ValueError("exactly one run directory required")
        prefix = next(iter(roots)) + "/"
        if files.get(prefix + "release.zip", {}).get("sha256") != expected_release_sha256:
            raise ValueError("saved release ZIP missing or differs")
        for name, spec in actual.items():
            if files.get(prefix + "payload/" + name) != spec:
                raise ValueError(f"saved payload differs: {name}")
        installed = {
            k.removeprefix(prefix + "installed/"): v
            for k, v in files.items()
            if k.startswith(prefix + "installed/frem_s14/") and k.endswith(".py")
        }
        if installed != code:
            raise ValueError("installed Python source differs from anchored wheel")
        delivery = decode_json(small_read(output, prefix + "delivery.json"))
        if (
            delivery["archive_sha256"] != expected_release_sha256
            or delivery["wheel_sha256"] != expected[wheel_name]["sha256"]
            or delivery["mode"] not in ("qualify", "historical_suite")
        ):
            raise ValueError("delivery identity or mode differs")
        source_identity = identity({PurePosixPath(k).name: v["sha256"] for k, v in code.items()})
        manifests, stages = [], {}
        for name in files:
            if (
                name.endswith("/manifest.json")
                and "/payload/" not in name
                and "/installed/" not in name
            ):
                saved = decode_json(small_read(output, name))
                parent = name.removesuffix("manifest.json")
                members = {
                    k.removeprefix(parent): v
                    for k, v in files.items()
                    if k.startswith(parent) and k != name
                }
                if saved != {"schema": 1, "files": members}:
                    raise ValueError(f"saved result manifest differs: {name}")
                manifests.append(name.removeprefix(prefix))
            if (
                name.endswith("/source.json")
                and "/payload/" not in name
                and "/installed/" not in name
            ):
                if decode_json(small_read(output, name))["identity"] != source_identity:
                    raise ValueError("run source identity differs from anchored wheel")
        for stage in (
            "preflight",
            "tests",
            "qualify",
            "verify_qualification",
            "intake",
            "historical_suite",
            "verify_suite",
        ):
            name = prefix + stage + ".json"
            if name in files:
                stages[stage] = decode_json(small_read(output, name))
    return {
        "archive_sha256": digest(download),
        "release_sha256": expected_release_sha256,
        "source_identity": source_identity,
        "files": len(files),
        "payload_members_verified": len(actual),
        "installed_sources_verified": len(code),
        "manifests_verified": manifests,
        "stages": stages,
        "integrity_verified": True,
        "numerical_semantics_verified": False,
        "claim": "Integrity readback only; absent stages and nonzero return codes are not passes.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(
        args.output, verify_download(args.archive, args.release, args.expected_release_sha256)
    )


if __name__ == "__main__":
    main()
