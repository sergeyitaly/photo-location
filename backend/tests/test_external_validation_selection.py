"""Tests for Wikipedia/open-data candidate selection (no network)."""

from app.services.external_validation_selection import (
    CandidateProof,
    select_validation_candidate_index,
)


def test_keeps_primary_when_core_gates_pass_even_if_alt_has_photo():
    """Primary text+relief+wiki pass; alt only wins on photo — pin stays #0."""
    proofs = [
        CandidateProof(
            index=0,
            wiki_proven=True,
            relief_proven=True,
            semantic_proven=True,
            photo_proven=False,
            semantic_similarity=0.625,
            photo_similarity=None,
        ),
        CandidateProof(
            index=1,
            wiki_proven=True,
            relief_proven=True,
            semantic_proven=True,
            photo_proven=True,
            semantic_similarity=0.621,
            photo_similarity=0.686,
        ),
    ]
    idx, adjusted, satisfied = select_validation_candidate_index(
        proofs, promote_min_score_delta=0.12
    )
    assert idx == 0
    assert adjusted is False
    assert satisfied is True


def test_promotes_alt_when_primary_fails_core_and_alt_wins_by_delta():
    proofs = [
        CandidateProof(
            index=0,
            wiki_proven=True,
            relief_proven=True,
            semantic_proven=False,
            photo_proven=False,
            semantic_similarity=0.05,
        ),
        CandidateProof(
            index=1,
            wiki_proven=True,
            relief_proven=True,
            semantic_proven=True,
            photo_proven=True,
            semantic_similarity=0.55,
            photo_similarity=0.72,
        ),
    ]
    idx, adjusted, satisfied = select_validation_candidate_index(
        proofs, promote_min_score_delta=0.12
    )
    assert idx == 1
    assert adjusted is True
    assert satisfied is True


def test_no_promotion_when_alt_margin_below_delta():
    proofs = [
        CandidateProof(
            index=0,
            wiki_proven=True,
            relief_proven=True,
            semantic_proven=True,
            photo_proven=False,
            semantic_similarity=0.55,
        ),
        CandidateProof(
            index=1,
            wiki_proven=True,
            relief_proven=True,
            semantic_proven=True,
            photo_proven=True,
            semantic_similarity=0.56,
            photo_similarity=0.10,
        ),
    ]
    idx, adjusted, _ = select_validation_candidate_index(
        proofs, promote_min_score_delta=0.12
    )
    assert idx == 0
    assert adjusted is False
