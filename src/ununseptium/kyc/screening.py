"""Sanctions and PEP screening for KYC.

Provides watchlist matching against sanctions lists, PEP databases,
and other screening requirements.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class WatchlistType(str, Enum):
    """Types of watchlists for screening."""

    OFAC_SDN = "ofac_sdn"
    OFAC_CONS = "ofac_consolidated"
    EU_SANCTIONS = "eu_sanctions"
    UN_SANCTIONS = "un_sanctions"
    UK_SANCTIONS = "uk_sanctions"
    PEP = "pep"
    ADVERSE_MEDIA = "adverse_media"
    CUSTOM = "custom"


class MatchType(str, Enum):
    """Type of screening match."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    PHONETIC = "phonetic"
    ALIAS = "alias"


class WatchlistEntry(BaseModel):
    """Entry in a watchlist.

    Attributes:
        id: Unique entry identifier.
        list_type: Type of watchlist.
        name: Primary name.
        aliases: Alternative names.
        date_of_birth: Date of birth if known.
        nationality: Nationality/country.
        identifiers: Identity document numbers.
        programs: Sanctions programs (for sanctions lists).
        reasons: Listing reasons.
        source_url: Source of the listing.
        added_date: When added to list.
        metadata: Additional data.
    """

    id: str = Field(default_factory=lambda: f"WL-{uuid4().hex[:12].upper()}")
    list_type: WatchlistType
    name: str
    aliases: list[str] = Field(default_factory=list)
    date_of_birth: str | None = None
    nationality: str | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    programs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    source_url: str | None = None
    added_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScreeningMatch(BaseModel):
    """A potential match from screening.

    Attributes:
        entry_id: ID of the matched watchlist entry.
        list_type: Type of watchlist.
        matched_name: Name that matched.
        query_name: Original query name.
        match_type: Type of match.
        score: Match confidence score (0.0 to 1.0).
        matched_fields: Fields that contributed to match.
        entry_data: Data from the watchlist entry.
    """

    entry_id: str
    list_type: WatchlistType
    matched_name: str
    query_name: str
    match_type: MatchType
    score: float = Field(ge=0.0, le=1.0)
    matched_fields: list[str] = Field(default_factory=list)
    entry_data: dict[str, Any] = Field(default_factory=dict)


class ScreeningResult(BaseModel):
    """Result of screening an identity.

    Attributes:
        identity_id: ID of the screened identity.
        screened_at: Timestamp of screening.
        lists_searched: Watchlists that were searched.
        matches: Potential matches found.
        has_matches: Whether any matches were found.
        highest_score: Highest match score.
        status: Processing status.
        metadata: Additional screening data.
    """

    identity_id: str
    screened_at: datetime = Field(default_factory=datetime.utcnow)
    lists_searched: list[WatchlistType] = Field(default_factory=list)
    matches: list[ScreeningMatch] = Field(default_factory=list)
    has_matches: bool = False
    highest_score: float = 0.0
    status: str = "completed"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        """Update computed fields after initialization."""
        self.has_matches = len(self.matches) > 0
        if self.matches:
            self.highest_score = max(m.score for m in self.matches)


class WatchlistMatcher:
    """Match names against watchlist entries.

    Provides fuzzy matching capabilities for name screening
    including phonetic matching and alias handling.

    Attributes:
        threshold: Minimum score for a match (0.0 to 1.0).

    Example:
        ```python
        from ununseptium.kyc import WatchlistMatcher, WatchlistEntry, WatchlistType

        matcher = WatchlistMatcher(threshold=0.8)

        entry = WatchlistEntry(
            list_type=WatchlistType.OFAC_SDN,
            name="John Smith",
            aliases=["J. Smith", "Johnny Smith"]
        )
        matcher.add_entry(entry)

        matches = matcher.search("Jon Smith")
        ```
    """

    def __init__(self, threshold: float = 0.8) -> None:
        """Initialize the matcher.

        Args:
            threshold: Minimum match score (0.0 to 1.0).
        """
        self.threshold = threshold
        self._entries: list[WatchlistEntry] = []
        self._name_index: dict[str, list[int]] = {}

    def add_entry(self, entry: WatchlistEntry) -> None:
        """Add a watchlist entry.

        Args:
            entry: Entry to add.
        """
        idx = len(self._entries)
        self._entries.append(entry)

        # Index by normalized name components
        for name in [entry.name, *entry.aliases]:
            normalized = self._normalize_name(name)
            for token in normalized.split():
                if token not in self._name_index:
                    self._name_index[token] = []
                if idx not in self._name_index[token]:
                    self._name_index[token].append(idx)

    def add_entries(self, entries: list[WatchlistEntry]) -> None:
        """Add multiple watchlist entries.

        Args:
            entries: Entries to add.
        """
        for entry in entries:
            self.add_entry(entry)

    def search(
        self,
        name: str,
        *,
        list_types: list[WatchlistType] | None = None,
    ) -> list[ScreeningMatch]:
        """Search for matches against a name.

        Args:
            name: Name to search for.
            list_types: Optional filter by list types.

        Returns:
            List of matching entries above threshold.
        """
        matches: list[ScreeningMatch] = []
        normalized_query = self._normalize_name(name)
        query_tokens = set(normalized_query.split())

        # Find candidate entries via index
        candidate_indices: set[int] = set()
        for token in query_tokens:
            if token in self._name_index:
                candidate_indices.update(self._name_index[token])

        # Score candidates
        for idx in candidate_indices:
            entry = self._entries[idx]

            # Filter by list type if specified
            if list_types and entry.list_type not in list_types:
                continue

            # Score against primary name and aliases
            best_score = 0.0
            best_name = entry.name
            match_type = MatchType.FUZZY

            for entry_name in [entry.name, *entry.aliases]:
                score = self._compute_similarity(name, entry_name)
                if score > best_score:
                    best_score = score
                    best_name = entry_name
                    match_type = (
                        MatchType.EXACT
                        if score >= 0.99
                        else MatchType.ALIAS
                        if entry_name != entry.name
                        else MatchType.FUZZY
                    )

            if best_score >= self.threshold:
                matches.append(
                    ScreeningMatch(
                        entry_id=entry.id,
                        list_type=entry.list_type,
                        matched_name=best_name,
                        query_name=name,
                        match_type=match_type,
                        score=round(best_score, 4),
                        matched_fields=["name"],
                        entry_data={
                            "name": entry.name,
                            "aliases": entry.aliases,
                            "programs": entry.programs,
                        },
                    )
                )

        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison.

        Args:
            name: Name to normalize.

        Returns:
            Normalized name string.
        """
        # Lowercase and remove punctuation
        name = name.lower()
        name = re.sub(r"[^\w\s]", " ", name)
        # Collapse whitespace
        name = " ".join(name.split())
        return name

    def _compute_similarity(self, name1: str, name2: str) -> float:
        """Compute similarity between two names.

        Uses a combination of token overlap and edit distance.

        Args:
            name1: First name.
            name2: Second name.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)

        if n1 == n2:
            return 1.0

        # Token-based Jaccard similarity
        tokens1 = set(n1.split())
        tokens2 = set(n2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard = intersection / union if union > 0 else 0.0

        # Character-level similarity (simple ratio)
        len_sum = len(n1) + len(n2)
        if len_sum == 0:
            return 1.0

        # Levenshtein distance approximation
        dist = self._levenshtein_distance(n1, n2)
        char_sim = 1.0 - (dist / max(len(n1), len(n2))) if max(len(n1), len(n2)) > 0 else 0.0

        # Weighted combination
        return 0.6 * jaccard + 0.4 * char_sim

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Edit distance.
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class ScreeningEngine:
    """Orchestrate screening across multiple watchlists.

    Coordinates screening operations, managing multiple
    matchers and aggregating results.

    Example:
        ```python
        from ununseptium.kyc import ScreeningEngine, Identity

        engine = ScreeningEngine()

        identity = Identity(name="John Smith", nationality="US")
        result = engine.screen(identity)

        if result.has_matches:
            for match in result.matches:
                print(f"Match: {match.matched_name} ({match.score})")
        ```
    """

    def __init__(self, threshold: float = 0.8) -> None:
        """Initialize the screening engine.

        Args:
            threshold: Default match threshold.
        """
        self.threshold = threshold
        self._matchers: dict[WatchlistType, WatchlistMatcher] = {}

    def add_watchlist(
        self,
        list_type: WatchlistType,
        entries: list[WatchlistEntry],
    ) -> None:
        """Add a watchlist to the engine.

        Args:
            list_type: Type of watchlist.
            entries: Watchlist entries.
        """
        matcher = WatchlistMatcher(threshold=self.threshold)
        matcher.add_entries(entries)
        self._matchers[list_type] = matcher

    def screen(
        self,
        identity: Any,
        *,
        list_types: list[WatchlistType] | None = None,
    ) -> ScreeningResult:
        """Screen an identity against watchlists.

        Args:
            identity: Identity to screen (must have 'id' and 'name' attributes).
            list_types: Optional filter for specific list types.

        Returns:
            ScreeningResult with all matches.
        """
        # Get identity details
        identity_id = getattr(identity, "id", str(uuid4()))
        name = getattr(identity, "name", "")

        all_matches: list[ScreeningMatch] = []
        lists_searched: list[WatchlistType] = []

        # Screen against each matcher
        for list_type, matcher in self._matchers.items():
            if list_types and list_type not in list_types:
                continue

            lists_searched.append(list_type)
            matches = matcher.search(name, list_types=[list_type])
            all_matches.extend(matches)

        return ScreeningResult(
            identity_id=identity_id,
            lists_searched=lists_searched,
            matches=all_matches,
        )
