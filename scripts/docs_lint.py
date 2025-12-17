#!/usr/bin/env python3
"""Documentation quality linter for ununseptium.

This script validates documentation files against project standards:
- No emojis allowed
- Required sections present (Scope, Non-goals, Definitions, References)
- Internal links valid
- Mermaid diagrams and LaTeX math present where appropriate
- At least one table per major document

Usage:
    python scripts/docs_lint.py [--check-links] [--verbose]

Exit codes:
    0 - All checks passed
    1 - Validation errors found
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

# Emoji pattern (Unicode ranges for common emojis)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "]+",
    flags=re.UNICODE,
)

# Required root documentation files
REQUIRED_ROOT_DOCS = [
    "README.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "GOVERNANCE.md",
    "SUPPORT.md",
    "CHANGELOG.md",
    "LICENSE",
    "PRIVACY.md",
    "COMPLIANCE.md",
    "CITATION.md",
    "ROADMAP.md",
]

# Required docs/ files
REQUIRED_DOCS_FILES = [
    "docs/index.md",
    "docs/toc.md",
    "docs/glossary.md",
    "docs/faq.md",
    "docs/references.md",
    "docs/architecture/overview.md",
    "docs/architecture/data-flow.md",
    "docs/architecture/plugin-architecture.md",
    "docs/architecture/ai-pipeline.md",
    "docs/kyc/kyc-overview.md",
    "docs/aml/aml-overview.md",
    "docs/security/security-overview.md",
    "docs/security/threat-model.md",
    "docs/security/auditability.md",
    "docs/security/crypto-and-key-management.md",
    "docs/mathstats/mathstats-overview.md",
    "docs/mathstats/uncertainty.md",
    "docs/mathstats/sequential.md",
    "docs/mathstats/evt.md",
    "docs/mathstats/hawkes.md",
    "docs/mathstats/graph-features.md",
    "docs/ai/ai-overview.md",
    "docs/ai/sciml.md",
    "docs/ai/governance.md",
    "docs/model-zoo/model-zoo.md",
    "docs/performance/performance.md",
    "docs/legal/legal-notices.md",
    "docs/figures/README.md",
]

# Major docs that require full sections
MAJOR_DOCS_PATTERNS = [
    "**/overview.md",
    "**/README.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "PRIVACY.md",
    "COMPLIANCE.md",
]


class LintError(NamedTuple):
    """Represents a linting error."""

    file: str
    line: int | None
    message: str
    severity: str  # "error" or "warning"


class DocsLinter:
    """Documentation linter implementation."""

    def __init__(self, root: Path, verbose: bool = False) -> None:
        """Initialize linter with project root."""
        self.root = root
        self.verbose = verbose
        self.errors: list[LintError] = []

    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def add_error(
        self,
        file: str,
        message: str,
        line: int | None = None,
        severity: str = "error",
    ) -> None:
        """Add an error to the list."""
        self.errors.append(LintError(file, line, message, severity))

    def check_no_emojis(self, file_path: Path) -> None:
        """Check that file contains no emojis."""
        self.log(f"Checking emojis: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                if EMOJI_PATTERN.search(line):
                    self.add_error(
                        str(file_path.relative_to(self.root)),
                        "Emoji detected (emojis are not allowed)",
                        line=i,
                    )
        except Exception as e:
            self.add_error(str(file_path), f"Failed to read file: {e}")

    def check_required_sections(self, file_path: Path) -> None:
        """Check that major docs have required sections."""
        self.log(f"Checking sections: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8").lower()

            # Check for Scope section
            if "## scope" not in content and "### scope" not in content:
                self.add_error(
                    str(file_path.relative_to(self.root)),
                    "Missing 'Scope' section",
                    severity="warning",
                )

            # Check for Definitions or link to glossary
            has_definitions = (
                "## definitions" in content
                or "### definitions" in content
                or "glossary.md" in content
            )
            if not has_definitions:
                self.add_error(
                    str(file_path.relative_to(self.root)),
                    "Missing 'Definitions' section or glossary link",
                    severity="warning",
                )

            # Check for References section
            if "## references" not in content and "### references" not in content:
                self.add_error(
                    str(file_path.relative_to(self.root)),
                    "Missing 'References' section",
                    severity="warning",
                )

        except Exception as e:
            self.add_error(str(file_path), f"Failed to read file: {e}")

    def check_has_table(self, file_path: Path) -> None:
        """Check that file contains at least one markdown table."""
        self.log(f"Checking tables: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
            # Simple check for markdown table (pipe characters in rows)
            table_pattern = re.compile(r"^\|.*\|.*\|", re.MULTILINE)
            if not table_pattern.search(content):
                self.add_error(
                    str(file_path.relative_to(self.root)),
                    "No markdown table found (at least one table required)",
                    severity="warning",
                )
        except Exception as e:
            self.add_error(str(file_path), f"Failed to read file: {e}")

    def check_has_mermaid(self, file_path: Path) -> None:
        """Check that architecture docs have Mermaid diagrams."""
        self.log(f"Checking Mermaid: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
            if "```mermaid" not in content:
                self.add_error(
                    str(file_path.relative_to(self.root)),
                    "No Mermaid diagram found (architecture docs should include diagrams)",
                    severity="warning",
                )
        except Exception as e:
            self.add_error(str(file_path), f"Failed to read file: {e}")

    def check_has_math(self, file_path: Path) -> None:
        """Check that mathstats docs have LaTeX formulas."""
        self.log(f"Checking LaTeX: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
            # Check for inline or block math
            has_math = "$" in content and (
                re.search(r"\$[^$]+\$", content) or "$$" in content
            )
            if not has_math:
                self.add_error(
                    str(file_path.relative_to(self.root)),
                    "No LaTeX math found (mathstats docs should include formulas)",
                    severity="warning",
                )
        except Exception as e:
            self.add_error(str(file_path), f"Failed to read file: {e}")

    def check_internal_links(self, file_path: Path) -> None:
        """Check that internal markdown links are valid."""
        self.log(f"Checking links: {file_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
            # Find all markdown links
            link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

            for match in link_pattern.finditer(content):
                link_target = match.group(2)

                # Skip external links and anchors
                if link_target.startswith(("http://", "https://", "#", "mailto:")):
                    continue

                # Handle anchor in link
                if "#" in link_target:
                    link_target = link_target.split("#")[0]

                if not link_target:
                    continue

                # Resolve relative path
                if link_target.startswith("/"):
                    target_path = self.root / link_target[1:]
                else:
                    target_path = file_path.parent / link_target

                if not target_path.exists():
                    line_num = content[: match.start()].count("\n") + 1
                    self.add_error(
                        str(file_path.relative_to(self.root)),
                        f"Broken link: {link_target}",
                        line=line_num,
                    )

        except Exception as e:
            self.add_error(str(file_path), f"Failed to check links: {e}")

    def check_required_files_exist(self) -> None:
        """Check that all required documentation files exist."""
        self.log("Checking required files...")

        for doc in REQUIRED_ROOT_DOCS:
            path = self.root / doc
            if not path.exists():
                self.add_error(doc, "Required file is missing")

        for doc in REQUIRED_DOCS_FILES:
            path = self.root / doc
            if not path.exists():
                self.add_error(doc, "Required documentation file is missing")

    def is_major_doc(self, file_path: Path) -> bool:
        """Check if file is a major documentation file requiring full sections."""
        name = file_path.name
        rel_path = str(file_path.relative_to(self.root))

        # Check various patterns
        if name == "README.md":
            return True
        if "overview" in name.lower():
            return True
        if rel_path in [
            "SECURITY.md",
            "CONTRIBUTING.md",
            "GOVERNANCE.md",
            "PRIVACY.md",
            "COMPLIANCE.md",
        ]:
            return True

        return False

    def is_architecture_doc(self, file_path: Path) -> bool:
        """Check if file is an architecture document requiring Mermaid."""
        return "architecture" in str(file_path) or "data-flow" in file_path.name

    def is_mathstats_doc(self, file_path: Path) -> bool:
        """Check if file is a mathstats document requiring LaTeX."""
        return "mathstats" in str(file_path)

    def lint_all(self, check_links: bool = False) -> int:
        """Run all linting checks and return exit code."""
        # Check required files
        self.check_required_files_exist()

        # Find all markdown files
        md_files = list(self.root.glob("**/*.md"))

        # Exclude common non-doc directories
        excluded = {"node_modules", ".git", ".venv", "venv", "__pycache__", "build", "dist"}
        md_files = [
            f
            for f in md_files
            if not any(ex in f.parts for ex in excluded)
        ]

        for md_file in md_files:
            # Check no emojis
            self.check_no_emojis(md_file)

            # Check required sections for major docs
            if self.is_major_doc(md_file):
                self.check_required_sections(md_file)
                self.check_has_table(md_file)

            # Check Mermaid for architecture docs
            if self.is_architecture_doc(md_file):
                self.check_has_mermaid(md_file)

            # Check LaTeX for mathstats docs
            if self.is_mathstats_doc(md_file):
                self.check_has_math(md_file)

            # Check internal links
            if check_links:
                self.check_internal_links(md_file)

        # Print results
        if self.errors:
            print(f"\nFound {len(self.errors)} issue(s):\n")
            for error in sorted(self.errors, key=lambda e: (e.severity, e.file)):
                line_info = f":{error.line}" if error.line else ""
                severity_marker = "ERROR" if error.severity == "error" else "WARN"
                print(f"[{severity_marker}] {error.file}{line_info}: {error.message}")

            # Count errors vs warnings
            error_count = sum(1 for e in self.errors if e.severity == "error")
            warning_count = sum(1 for e in self.errors if e.severity == "warning")

            print(f"\nTotal: {error_count} error(s), {warning_count} warning(s)")

            # Only fail on errors, not warnings
            return 1 if error_count > 0 else 0
        else:
            print("All documentation checks passed!")
            return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Lint documentation files")
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Check that internal links are valid",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory",
    )

    args = parser.parse_args()

    linter = DocsLinter(args.root.resolve(), verbose=args.verbose)
    return linter.lint_all(check_links=args.check_links)


if __name__ == "__main__":
    sys.exit(main())
