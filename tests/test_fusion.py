"""Basic tests for the fusion scoring logic."""
from src.models.fusion import fuse_scores


def make_config():
    return {
        "fusion": {
            "weights": {
                "forensic_score": 0.5,
                "metadata_score": 0.2,
                "semantic_score": 0.3,
            },
            "review_threshold": 0.4,
        }
    }


def test_low_scores_do_not_trigger_review():
    result = fuse_scores(0.1, 0.1, 0.1, make_config())
    assert result["route_to_human_review"] is False
    assert result["combined_risk_score"] < 0.4


def test_high_forensic_score_triggers_review():
    result = fuse_scores(0.9, 0.0, 0.0, make_config())
    assert result["route_to_human_review"] is True


def test_combined_moderate_scores_trigger_review():
    result = fuse_scores(0.5, 0.5, 0.5, make_config())
    assert result["route_to_human_review"] is True
