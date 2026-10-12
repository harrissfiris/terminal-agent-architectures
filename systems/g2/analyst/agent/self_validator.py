"""
Self-validation gate for Stage 2.5.

Runs rule-based checks (zero LLM cost) on the agent's output after it submits.
Parses the task description for expected output files and formats, then verifies
them with bash commands in the container.

Stage 2.5 change vs Stage 2: Python syntax checks REMOVED.
They caused false-positive regressions on assign-seats and jq-data-processing
(compile() flagged working .py files that had task-specific quirks).
Only file-existence and format-parse checks remain.

If checks fail, produces a structured self-diagnosis string for Attempt 2.
"""
from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from typing import Optional

from terminal.session import SafeTerminal

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationResult:
    """Aggregate result of all validation checks."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks) if self.checks else True

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_diagnosis(self) -> str:
        """Build a self-diagnosis string (~300 tokens) for Attempt 2."""
        if self.all_passed:
            return ""

        lines = [
            "## Self-Validation Failed",
            "Your solution has issues that need fixing:",
            "",
        ]
        for c in self.failed_checks:
            lines.append(f"- {c.name}: {c.detail}")

        lines.append("")
        lines.append(
            "Fix ONLY the issues above. Do not rewrite working code."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "all_passed": self.all_passed,
            "num_checks": len(self.checks),
            "num_failed": len(self.failed_checks),
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns for parsing task descriptions
# ─────────────────────────────────────────────────────────────────────────────

# Match any absolute path with a file extension mentioned in the instruction.
# Captures /app/hello.txt, /app/output.json, /tmp/result.csv, etc.
# Strips trailing punctuation (periods, commas, quotes, parens).
_ABS_PATH_PATTERN = re.compile(
    r"(/(?:app|tmp|home|root|opt|var|usr)\b\S+?\.\w+)",
    re.IGNORECASE,
)

# Match format mentions
_FORMAT_PATTERN = re.compile(
    r"\b(json|csv|parquet|yaml|yml)\b",
    re.IGNORECASE,
)


class SelfValidator:
    """
    Rule-based self-validation gate.

    Parses the task instruction for expected outputs, then runs bash checks
    in the container. Zero LLM cost.
    """

    def __init__(
        self,
        terminal: SafeTerminal,
        instruction: str,
        written_files: Optional[list[str]] = None,
    ):
        self._terminal = terminal
        self._instruction = instruction
        self._written_files = written_files or []

        # Parse expected outputs from instruction
        self._expected_files = self._extract_expected_files()
        self._expected_formats = self._extract_expected_formats()

    def _extract_expected_files(self) -> list[str]:
        """Extract expected output file paths from task description."""
        paths: set[str] = set()
        for match in _ABS_PATH_PATTERN.finditer(self._instruction):
            path = match.group(1).rstrip(".,;:)'\"")
            if path and len(path) > 3:  # skip bare /app
                paths.add(path)
        return sorted(paths)

    def _extract_expected_formats(self) -> dict[str, str]:
        """Extract expected file formats and map to expected output files."""
        formats: dict[str, str] = {}
        for match in _FORMAT_PATTERN.finditer(self._instruction):
            fmt = match.group(1).lower()
            if fmt == "yml":
                fmt = "yaml"
            # Try to associate with a known file
            for path in self._expected_files:
                if path.endswith(f".{fmt}"):
                    formats[path] = fmt
                    break
        return formats

    def run(self) -> ValidationResult:
        """Run all validation checks. Returns ValidationResult."""
        checks: list[CheckResult] = []

        # 1. File existence checks
        for path in self._expected_files:
            checks.append(self._check_file_exists(path))

        # 2. Python syntax checks — REMOVED in Stage 2.5.
        # compile() caused false-positive regressions on assign-seats and
        # jq-data-processing where working .py files were incorrectly flagged.

        # 3. Format parse checks
        for path, fmt in self._expected_formats.items():
            # Only check if file exists (don't duplicate the "not found" error)
            exists = self._terminal.exec(
                f"test -f {shlex.quote(path)} && echo YES || echo NO",
                timeout=5.0,
            )
            if "YES" in exists.output:
                checks.append(self._check_format_parse(path, fmt))

        # Filter out: if no checks could be generated, that's OK (skip validation)
        result = ValidationResult(checks=checks)

        if checks:
            passed = sum(1 for c in checks if c.passed)
            failed = sum(1 for c in checks if not c.passed)
            logger.info(
                f"Self-validation: {passed} passed, {failed} failed "
                f"(files={len(self._expected_files)}, "
                f"format_checks={len(self._expected_formats)})"
            )
        else:
            logger.info("Self-validation: no checks applicable (skipping)")

        return result

    def _check_file_exists(self, path: str) -> CheckResult:
        """Check if an expected output file exists."""
        result = self._terminal.exec(
            f"test -f {shlex.quote(path)} && echo EXISTS || echo MISSING",
            timeout=5.0,
        )
        if "EXISTS" in result.output:
            # Also check non-empty
            size_result = self._terminal.exec(
                f"wc -c < {shlex.quote(path)}", timeout=5.0
            )
            size = size_result.output.strip()
            if size == "0":
                return CheckResult(
                    name=path,
                    passed=False,
                    detail="File exists but is empty (0 bytes)",
                )
            return CheckResult(name=path, passed=True, detail=f"Exists ({size} bytes)")

        return CheckResult(
            name=path,
            passed=False,
            detail="File does not exist (expected by task)",
        )

    def _check_python_syntax(self, path: str) -> CheckResult:
        """Check Python file for syntax errors using compile() (no .pyc, no imports)."""
        result = self._terminal.exec(
            f'python3 -c "compile(open(\'{path}\').read(), \'{path}\', \'exec\')" 2>&1',
            timeout=10.0,
        )
        if result.exit_code == 0:
            return CheckResult(name=f"{path} (syntax)", passed=True)

        # Extract error detail
        detail = result.output.strip().split("\n")[-1] if result.output.strip() else "Syntax check failed"
        return CheckResult(
            name=f"{path} (syntax)",
            passed=False,
            detail=detail[:200],
        )

    def _check_format_parse(self, path: str, fmt: str) -> CheckResult:
        """Check if file parses as the expected format."""
        if fmt == "json":
            cmd = f'python3 -c "import json; json.load(open(\'{path}\'))" 2>&1'
        elif fmt == "csv":
            cmd = f'python3 -c "import csv; list(csv.reader(open(\'{path}\')))" 2>&1'
        elif fmt == "yaml":
            cmd = f'python3 -c "import yaml; yaml.safe_load(open(\'{path}\'))" 2>&1'
        elif fmt == "parquet":
            cmd = f'python3 -c "import pandas; pandas.read_parquet(\'{path}\')" 2>&1'
        else:
            return CheckResult(name=f"{path} ({fmt})", passed=True, detail="No parser available")

        result = self._terminal.exec(cmd, timeout=10.0)
        if result.exit_code == 0:
            return CheckResult(name=f"{path} ({fmt})", passed=True)

        detail = result.output.strip().split("\n")[-1] if result.output.strip() else f"{fmt} parse failed"
        return CheckResult(
            name=f"{path} ({fmt})",
            passed=False,
            detail=detail[:200],
        )
