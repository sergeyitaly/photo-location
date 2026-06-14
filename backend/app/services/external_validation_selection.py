"""Open-data validation winner selection (no heavy ML / numpy imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CandidateProof:
    """Per-candidate open-data + CLIP gate results."""

    index: int
    wiki_proven: bool
    relief_proven: bool
    semantic_proven: bool
    photo_proven: bool
    semantic_similarity: Optional[float] = None
    photo_similarity: Optional[float] = None

    @property
    def core_proven(self) -> bool:
        return self.wiki_proven and self.relief_proven and self.semantic_proven

    @property
    def full_proven(self) -> bool:
        return self.core_proven and self.photo_proven

    def proof_score(self) -> float:
        if not self.wiki_proven or not self.relief_proven:
            return -1.0
        score = float(self.semantic_similarity or 0.0)
        if self.photo_proven:
            score += float(self.photo_similarity or 0.0)
        return score


def select_validation_candidate_index(
    proofs: List[CandidateProof],
    *,
    promote_min_score_delta: float,
) -> Tuple[int, bool, bool]:
    """
    Pick which fusion candidate becomes the pin after open-data validation.

    Returns (selected_index, pin_adjusted, proof_satisfied).
    """
    if not proofs:
        return 0, False, False

    primary = proofs[0]
    proof_satisfied = any(p.full_proven for p in proofs) or primary.core_proven

    if primary.core_proven:
        return 0, False, proof_satisfied

    passing = [p for p in proofs if p.full_proven]
    if not passing:
        return 0, False, False

    best = max(passing, key=lambda p: p.proof_score())
    if best.index == 0:
        return 0, False, True

    base_score = proofs[0].proof_score() if proofs[0].full_proven else (
        float(proofs[0].semantic_similarity or 0.0) if proofs[0].core_proven else -1.0
    )
    if best.proof_score() >= base_score + promote_min_score_delta:
        return best.index, True, True
    return 0, False, True
