#!/usr/bin/env python3
"""Link checker for ununseptium documentation.

This script validates internal links in Markdown files.

Usage:
    python scripts/link_check.py [--fix] [--verbose]

Exit codes:
    0 - All links valid
    1 - Broken links found
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlparse


class BrokenLink(NamedTuple):
    """Represents a broken link."""

    file: str
    line: int
    link: str
    reason: str


class LinkChecker:
    """Documentation link checker."""

    def __init__(self, root: Path, verbose: bool = False) -> None:
        """Initialize checker with project root."""
        self.root = root
        self.verbose = verbose
        self.broken_links: list[BrokenLink] = []
        self.checked_count = 0
        self.valid_count = 0

    def log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[INFO] {message}")

    def extract_links(self, content: str) -> list[tuple[int, str, str]]:
        """Extract markdown links from content."""
        links = []
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

        for i, line in enumerate(content.splitlines(), 1):
            for match in link_pattern.finditer(line):
                text = match.group(1)
                url = match.group(2)
                links.append((i, text, url))

        return links

    def is_external(self, url: str) -> bool:
        """Check if URL is external."""
        return url.startswith(("http://", "https://", "mailto:", "ftp://"))

    def check_link(self, file_path: Path, line: int, url: str) -> bool:
        """Check if a link is valid."""
        self.checked_count += 1

        # Skip external links
        if self.is_external(url):
            self.log(f"Skipping external: {url}")
            self.valid_count += 1
            return True

        # Skip anchor-only links
        if url.startswith("#"):
            self.log(f"Skipping anchor: {url}")
            self.valid_count += 1
            return True

        # Skip placeholder/example URLs (like 'url' in code examples)
        if url in ("url", "link", "example.com", "example"):
            self.log(f"Skipping placeholder: {url}")
            self.valid_count += 1
            return True

        # Handle anchor in link
        url_without_anchor = url.split("#")[0] if "#" in url else url

        if not url_without_anchor:
            self.valid_count += 1
            return True

        # Decode URL encoding
        url_without_anchor = unquote(url_without_anchor)

        # Resolve path
        if url_without_anchor.startswith("/"):
            target_path = self.root / url_without_anchor[1:]
        else:
            target_path = file_path.parent / url_without_anchor

        # Normalize path
        try:
            target_path = target_path.resolve()
        except Exception:
            self.broken_links.append(BrokenLink(
                file=str(file_path.relative_to(self.root)),
                line=line,
                link=url,
                reason="Invalid path",
            ))
            return False

        # Check existence
        if not target_path.exists():
            self.broken_links.append(BrokenLink(
                file=str(file_path.relative_to(self.root)),
                line=line,
                link=url,
                reason="File not found",
            ))
            return False

        self.valid_count += 1
        return True

    def check_file(self, file_path: Path) -> int:
        """Check all links in a file."""
        self.log(f"Checking: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            self.broken_links.append(BrokenLink(
                file=str(file_path.relative_to(self.root)),
                line=0,
                link="",
                reason=f"Could not read file: {e}",
            ))
            return 1

        links = self.extract_links(content)
        broken = 0

        for line, text, url in links:
            if not self.check_link(file_path, line, url):
                broken += 1

        return broken

    def check_all(self) -> int:
        """Check all markdown files."""
        # Find all markdown files
        md_files = list(self.root.glob("**/*.md"))

        # Exclude common non-doc directories
        excluded = {"node_modules", ".git", ".venv", "venv", "__pycache__", "build", "dist"}
        md_files = [
            f for f in md_files
            if not any(ex in f.parts for ex in excluded)
        ]

        print(f"Checking {len(md_files)} markdown files...")

        total_broken = 0
        for md_file in md_files:
            total_broken += self.check_file(md_file)

        return total_broken

    def print_report(self) -> None:
        """Print check report."""
        print(f"\nLink Check Results:")
        print(f"  Checked: {self.checked_count}")
        print(f"  Valid: {self.valid_count}")
        print(f"  Broken: {len(self.broken_links)}")

        if self.broken_links:
            print(f"\nBroken Links:")
            for link in sorted(self.broken_links, key=lambda x: (x.file, x.line)):
                print(f"  {link.file}:{link.line}: {link.link}")
                print(f"    Reason: {link.reason}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check documentation links")
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

    checker = LinkChecker(args.root.resolve(), verbose=args.verbose)
    broken_count = checker.check_all()
    checker.print_report()

    if broken_count > 0:
        print(f"\nFailed: {broken_count} broken link(s) found")
        return 1

    print("\nAll links valid!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
