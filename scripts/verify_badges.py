#!/usr/bin/env python3
"""Badge verification script for ununseptium.

This script verifies that README badges reference existing workflows
and use correct URLs for the repository.

Usage:
    python scripts/verify_badges.py [--fix] [--verbose]

Exit codes:
    0 - All badges valid
    1 - Badge issues found
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


# Repository configuration
REPO_OWNER = "olaflaitinen"
REPO_NAME = "ununseptium"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"


class BadgeIssue(NamedTuple):
    """Represents a badge issue."""

    badge_name: str
    issue: str
    line: int


class BadgeVerifier:
    """Badge verification implementation."""

    def __init__(self, root: Path, verbose: bool = False) -> None:
        """Initialize verifier with project root."""
        self.root = root
        self.verbose = verbose
        self.issues: list[BadgeIssue] = []
        self.workflows: set[str] = set()

    def log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[INFO] {message}")

    def discover_workflows(self) -> None:
        """Discover existing workflow files."""
        workflows_dir = self.root / ".github" / "workflows"

        if not workflows_dir.exists():
            self.log("No workflows directory found")
            return

        for wf_file in workflows_dir.glob("*.yml"):
            self.workflows.add(wf_file.name)
            self.log(f"Found workflow: {wf_file.name}")

    def extract_badges(self, content: str) -> list[tuple[int, str, str]]:
        """Extract badge definitions from README."""
        badges = []

        # Pattern for markdown image badges
        # [![name](url)](link)
        badge_pattern = re.compile(
            r"\[!\[([^\]]*)\]\(([^)]+)\)\](?:\(([^)]+)\))?",
            re.MULTILINE,
        )

        for i, line in enumerate(content.splitlines(), 1):
            for match in badge_pattern.finditer(line):
                name = match.group(1) or "unnamed"
                url = match.group(2)
                badges.append((i, name, url))

        return badges

    def check_workflow_badge(self, line: int, name: str, url: str) -> bool:
        """Check if workflow badge references existing workflow."""
        # Extract workflow name from badge URL
        # Pattern: /actions/workflows/NAME.yml/badge.svg
        wf_pattern = re.compile(r"/actions/workflows/([^/]+)/badge\.svg")
        match = wf_pattern.search(url)

        if match:
            workflow_name = match.group(1)
            self.log(f"Badge '{name}' references workflow: {workflow_name}")

            if workflow_name not in self.workflows:
                self.issues.append(BadgeIssue(
                    badge_name=name,
                    issue=f"Workflow '{workflow_name}' does not exist",
                    line=line,
                ))
                return False

            # Check repo URL is correct
            if REPO_URL not in url:
                self.issues.append(BadgeIssue(
                    badge_name=name,
                    issue=f"Badge URL does not match repo: expected {REPO_URL}",
                    line=line,
                ))
                return False

        return True

    def check_readme(self) -> int:
        """Check README badges."""
        readme_path = self.root / "README.md"

        if not readme_path.exists():
            self.issues.append(BadgeIssue(
                badge_name="README",
                issue="README.md not found",
                line=0,
            ))
            return 1

        content = readme_path.read_text(encoding="utf-8")
        badges = self.extract_badges(content)

        print(f"Found {len(badges)} badges in README")

        issue_count = 0
        workflow_badges = 0

        for line, name, url in badges:
            self.log(f"Checking badge: {name}")

            # Check workflow badges
            if "actions/workflows" in url:
                workflow_badges += 1
                if not self.check_workflow_badge(line, name, url):
                    issue_count += 1

        print(f"Workflow badges: {workflow_badges}")
        print(f"Existing workflows: {len(self.workflows)}")

        return issue_count

    def verify_all(self) -> int:
        """Run all verifications."""
        # Discover workflows first
        self.discover_workflows()

        # Check README badges
        issues = self.check_readme()

        return issues

    def print_report(self) -> None:
        """Print verification report."""
        print(f"\nBadge Verification Results:")
        print(f"  Workflows found: {len(self.workflows)}")
        print(f"  Issues: {len(self.issues)}")

        if self.issues:
            print(f"\nIssues Found:")
            for issue in self.issues:
                print(f"  Line {issue.line}: {issue.badge_name}")
                print(f"    {issue.issue}")

        if self.workflows:
            print(f"\nAvailable Workflows:")
            for wf in sorted(self.workflows):
                print(f"  - {wf}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify README badges")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory",
    )

    args = parser.parse_args()

    verifier = BadgeVerifier(args.root.resolve(), verbose=args.verbose)
    issue_count = verifier.verify_all()
    verifier.print_report()

    if issue_count > 0:
        print(f"\nFailed: {issue_count} badge issue(s) found")
        return 1

    print("\nAll badges valid!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
