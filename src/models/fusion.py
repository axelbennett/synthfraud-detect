"""
Fusion layer: combines forensic classifier score, metadata suspicion score,
and semantic consistency score into one explainable risk score, plus a
routing decision.

Design principle: never auto-deny. Output a score, an explanation, and a
routing decision (auto-clear / human review). This is the architectural
choice that answers the "accountability gap" critique of AI in claims
decisions -- a human always makes the final call on flagged claims.
"""
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fuse_scores(forensic_score: float, metadata_score: float,
                 semantic_score: float, config: dict) -> dict:
    """
    All input scores should be in [0, 1], where higher = more suspicious.

    forensic_score: P(AI-generated) from the CNN classifier
    metadata_score: metadata_suspicion_score from metadata_check.py
    semantic_score: 1 - confidence if consistent == 'no', else 1 - confidence
                     inverted appropriately by the caller (see docstring below)
    """
    weights = config["fusion"]["weights"]

    combined = (
        weights["forensic_score"] * forensic_score
        + weights["metadata_score"] * metadata_score
        + weights["semantic_score"] * semantic_score
    )

    route_to_review = combined >= config["fusion"]["review_threshold"]

    explanation_parts = []
    if forensic_score >= 0.5:
        explanation_parts.append(f"forensic classifier flagged possible AI generation ({forensic_score:.2f})")
    if metadata_score >= 0.5:
        explanation_parts.append(f"metadata inconsistencies present ({metadata_score:.2f})")
    if semantic_score >= 0.5:
        explanation_parts.append(f"damage may not match claim narrative ({semantic_score:.2f})")

    explanation = "; ".join(explanation_parts) if explanation_parts else "no significant flags"

    return {
        "combined_risk_score": round(combined, 4),
        "route_to_human_review": route_to_review,
        "explanation": explanation,
        "component_scores": {
            "forensic_score": forensic_score,
            "metadata_score": metadata_score,
            "semantic_score": semantic_score,
        },
    }
