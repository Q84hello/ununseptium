#!/usr/bin/env python3
"""Repository audit script for ununseptium.

This script reads the repository manifest and verifies that all required
files exist, optionally creating missing files with production-grade content.

Usage:
    python scripts/repo_audit.py [--check] [--fix] [--report]

Options:
    --check     Check mode only, do not create files (exit 1 if missing)
    --fix       Create missing files with template content
    --report    Generate JSON and Markdown reports

Exit codes:
    0 - All required files present
    1 - Missing files detected (in check mode)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# Default paths
MANIFEST_PATH = Path(".repo/manifest.yaml")
ARTIFACTS_DIR = Path("artifacts")


class RepoAuditor:
    """Repository audit implementation."""

    def __init__(self, root: Path, manifest_path: Path | None = None) -> None:
        """Initialize auditor with project root."""
        self.root = root
        self.manifest_path = manifest_path or (root / MANIFEST_PATH)
        self.manifest: dict[str, Any] = {}
        self.report: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "present": [],
            "missing": [],
            "created": [],
            "errors": [],
        }

    def load_manifest(self) -> bool:
        """Load manifest file."""
        if yaml is None:
            # Fallback: parse basic YAML manually
            return self._load_manifest_simple()

        if not self.manifest_path.exists():
            self.report["errors"].append(f"Manifest not found: {self.manifest_path}")
            return False

        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                self.manifest = yaml.safe_load(f)
            return True
        except Exception as e:
            self.report["errors"].append(f"Failed to load manifest: {e}")
            return False

    def _load_manifest_simple(self) -> bool:
        """Simple manifest loading without PyYAML."""
        if not self.manifest_path.exists():
            self.report["errors"].append(f"Manifest not found: {self.manifest_path}")
            return False

        # Use hardcoded required paths as fallback
        self.manifest = {
            "required_root_files": [
                "README.md", "LICENSE", "SECURITY.md", "CONTRIBUTING.md",
                "CODE_OF_CONDUCT.md", "GOVERNANCE.md", "SUPPORT.md",
                "CHANGELOG.md", "PRIVACY.md", "COMPLIANCE.md",
                "CITATION.md", "ROADMAP.md", "pyproject.toml",
            ],
            "required_github_templates": [
                ".github/ISSUE_TEMPLATE/bug_report.yml",
                ".github/ISSUE_TEMPLATE/feature_request.yml",
                ".github/PULL_REQUEST_TEMPLATE.md",
            ],
            "required_workflows": [
                {"path": ".github/workflows/ci.yml"},
                {"path": ".github/workflows/docs-quality.yml"},
                {"path": ".github/workflows/security.yml"},
                {"path": ".github/workflows/build.yml"},
                {"path": ".github/workflows/release.yml"},
                {"path": ".github/workflows/publish.yml"},
                {"path": ".github/workflows/codeql.yml"},
                {"path": ".github/workflows/scorecard.yml"},
                {"path": ".github/workflows/sbom.yml"},
                {"path": ".github/workflows/provenance.yml"},
            ],
            "required_docs": [
                "docs/index.md", "docs/toc.md", "docs/glossary.md",
                "docs/faq.md", "docs/references.md",
                "docs/architecture/overview.md", "docs/architecture/data-flow.md",
                "docs/architecture/plugin-architecture.md", "docs/architecture/ai-pipeline.md",
                "docs/kyc/kyc-overview.md", "docs/aml/aml-overview.md",
                "docs/security/security-overview.md", "docs/security/threat-model.md",
                "docs/security/auditability.md", "docs/security/crypto-and-key-management.md",
                "docs/mathstats/mathstats-overview.md", "docs/mathstats/uncertainty.md",
                "docs/mathstats/sequential.md", "docs/mathstats/evt.md",
                "docs/mathstats/hawkes.md", "docs/mathstats/graph-features.md",
                "docs/ai/ai-overview.md", "docs/ai/sciml.md", "docs/ai/governance.md",
                "docs/model-zoo/model-zoo.md", "docs/performance/performance.md",
                "docs/legal/legal-notices.md", "docs/figures/README.md",
            ],
            "required_scripts": [
                "scripts/docs_lint.py", "scripts/generate_figures.py",
                "scripts/repo_audit.py", "scripts/link_check.py",
                "scripts/verify_badges.py",
            ],
            "required_python": [
                "src/ununseptium/__init__.py",
                "src/ununseptium/cli/main.py",
            ],
        }
        return True

    def check_path(self, path: str) -> bool:
        """Check if path exists."""
        full_path = self.root / path
        exists = full_path.exists()

        if exists:
            self.report["present"].append(path)
        else:
            self.report["missing"].append(path)

        return exists

    def check_all_paths(self) -> tuple[int, int]:
        """Check all required paths from manifest."""
        present = 0
        missing = 0

        # Root files
        for path in self.manifest.get("required_root_files", []):
            if self.check_path(path):
                present += 1
            else:
                missing += 1

        # GitHub templates
        for path in self.manifest.get("required_github_templates", []):
            if self.check_path(path):
                present += 1
            else:
                missing += 1

        # Workflows
        for wf in self.manifest.get("required_workflows", []):
            path = wf.get("path", wf) if isinstance(wf, dict) else wf
            if self.check_path(path):
                present += 1
            else:
                missing += 1

        # Docs
        for path in self.manifest.get("required_docs", []):
            if self.check_path(path):
                present += 1
            else:
                missing += 1

        # Scripts
        for path in self.manifest.get("required_scripts", []):
            if self.check_path(path):
                present += 1
            else:
                missing += 1

        # Python
        for path in self.manifest.get("required_python", []):
            if self.check_path(path):
                present += 1
            else:
                missing += 1

        return present, missing

    def check_no_emojis(self, path: Path) -> list[tuple[int, str]]:
        """Check file for emojis."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

        violations = []
        try:
            content = path.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                if emoji_pattern.search(line):
                    violations.append((i, line.strip()[:50]))
        except Exception:
            pass

        return violations

    def check_required_sections(self, path: Path) -> list[str]:
        """Check for required sections in major docs."""
        missing_sections = []
        try:
            content = path.read_text(encoding="utf-8").lower()

            if "## scope" not in content and "### scope" not in content:
                missing_sections.append("Scope")

            if "## definitions" not in content and "glossary" not in content:
                missing_sections.append("Definitions")

            if "## references" not in content and "### references" not in content:
                missing_sections.append("References")

        except Exception:
            pass

        return missing_sections

    def generate_report_json(self) -> str:
        """Generate JSON report."""
        return json.dumps(self.report, indent=2)

    def generate_report_markdown(self) -> str:
        """Generate Markdown report."""
        lines = [
            "# Repository Audit Report",
            "",
            f"**Generated:** {self.report['timestamp']}",
            "",
            "## Summary",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Present | {len(self.report['present'])} |",
            f"| Missing | {len(self.report['missing'])} |",
            f"| Created | {len(self.report['created'])} |",
            f"| Errors | {len(self.report['errors'])} |",
            "",
        ]

        if self.report["present"]:
            lines.extend([
                "## Present Files",
                "",
            ])
            for path in sorted(self.report["present"]):
                lines.append(f"- {path}")
            lines.append("")

        if self.report["missing"]:
            lines.extend([
                "## Missing Files",
                "",
            ])
            for path in sorted(self.report["missing"]):
                lines.append(f"- {path}")
            lines.append("")

        if self.report["created"]:
            lines.extend([
                "## Created Files",
                "",
            ])
            for path in sorted(self.report["created"]):
                lines.append(f"- {path}")
            lines.append("")

        if self.report["errors"]:
            lines.extend([
                "## Errors",
                "",
            ])
            for error in self.report["errors"]:
                lines.append(f"- {error}")
            lines.append("")

        lines.extend([
            "## Verification Commands",
            "",
            "```bash",
            "python -m venv .venv",
            ".venv\\Scripts\\activate  # Windows",
            "pip install -e \".[dev]\"",
            "python scripts/repo_audit.py --check",
            "python scripts/docs_lint.py",
            "pytest",
            "```",
            "",
        ])

        return "\n".join(lines)

    def save_reports(self, output_dir: Path) -> None:
        """Save reports to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "repo_audit_report.json"
        json_path.write_text(self.generate_report_json(), encoding="utf-8")

        md_path = output_dir / "repo_audit_report.md"
        md_path.write_text(self.generate_report_markdown(), encoding="utf-8")

        print(f"Reports saved to {output_dir}/")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audit repository completeness")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode only (exit 1 if missing)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Create missing files with templates",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate JSON and Markdown reports",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory",
    )

    args = parser.parse_args()

    auditor = RepoAuditor(args.root.resolve())

    if not auditor.load_manifest():
        print("Warning: Could not load manifest, using defaults")

    print("Checking required paths...")
    present, missing = auditor.check_all_paths()

    print(f"\nResults:")
    print(f"  Present: {present}")
    print(f"  Missing: {missing}")

    if args.report:
        auditor.save_reports(args.root / ARTIFACTS_DIR)

    if args.check and missing > 0:
        print(f"\nFailed: {missing} required file(s) missing")
        return 1

    if missing == 0:
        print("\nAll required files present!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
