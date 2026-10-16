"""Static source inventory only: never import frem or execute its tests/models."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import tomllib
from pathlib import Path


def strict_json(text: str):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            reject_constant(value)
        return result

    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant,
                      parse_float=finite_float)


def inspect_source(root: Path) -> dict:
    errors: list[dict] = []
    python_files: list[str] = []
    json_files: list[str] = []
    test_functions: list[str] = []
    package = root / "src" / "frem"
    if not (package / "__init__.py").is_file():
        errors.append({"path": "src/frem/__init__.py", "type": "missing_package"})
    for directory in (root / "src", root / "tests", root / "scripts"):
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=relative)
                compile(tree, relative, "exec")  # Creates code object; does not execute it.
                python_files.append(relative)
                for node in ast.walk(tree):
                    if (
                        path.is_relative_to(root / "tests")
                        and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name.startswith("test_")
                    ):
                        test_functions.append(relative + "::" + node.name)
                    if (
                        path.is_relative_to(package)
                        and isinstance(node, ast.ImportFrom)
                        and node.level == 1
                        and node.module
                    ):
                        module_path = package.joinpath(*node.module.split("."))
                        if (
                            not module_path.with_suffix(".py").exists()
                            and not (module_path / "__init__.py").exists()
                        ):
                            errors.append(
                                {"path": relative, "type": "missing_relative_module", "module": node.module}
                            )
            except (SyntaxError, UnicodeError, OSError) as exc:
                errors.append({"path": relative, "type": type(exc).__name__, "message": str(exc)})
    for directory in (root / "configs", root / "research", root / "data" / "templates", root / "docs"):
        for path in sorted(directory.rglob("*.json")):
            if path.name == "STATIC_VERIFICATION.json":
                continue
            try:
                strict_json(path.read_text(encoding="utf-8"))
                json_files.append(path.relative_to(root).as_posix())
            except (ValueError, UnicodeError, OSError) as exc:
                errors.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    try:
        strict_json((root / "RESEARCH_STATE.json").read_text(encoding="utf-8"))
        tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError) as exc:
        errors.append({"path": "project_metadata", "type": type(exc).__name__, "message": str(exc)})
    return {
        "verification_type": "AST parsing, in-memory compilation, local-module presence, JSON/TOML syntax",
        "python_version": sys.version.split()[0],
        "python_file_count": len(python_files),
        "json_file_count": len(json_files) + 1,
        "authored_test_function_count": len(test_functions),
        "python_files": python_files,
        "json_files": json_files + ["RESEARCH_STATE.json"],
        "authored_test_functions": test_functions,
        "errors": errors,
        "static_status": "PASS" if not errors else "FAIL",
        "runtime_tests_executed": False,
        "research_runs_executed": False,
        "limits": "No package import, dependency resolution, runtime API, numerical, gradient, device, model or data correctness is established by this check.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_source(args.root.resolve())
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"python_files", "json_files", "authored_test_functions"}
            },
            indent=2,
        )
    )
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
