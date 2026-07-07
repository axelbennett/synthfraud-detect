"""
Semantic consistency check: does the damage visible in the photo match the
claim narrative (e.g. "rear-ended at a stoplight" should show rear damage,
not a shattered windshield)?

Uses Claude's vision capability directly -- no separate model needed.
This is a genuinely useful, cheap-to-build signal, and it's the layer
where "solution architect" thinking shows: you're not asking an LLM to
detect pixel-level fakery (it can't reliably do that), you're asking it to
reason about physical/narrative consistency, which it's well suited for.
"""
import base64

from anthropic import Anthropic


def check_semantic_consistency(image_path: str, claim_narrative: str) -> dict:
    client = Anthropic()

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    prompt = f"""You are assisting an insurance claims reviewer, not making a final decision.

Claim narrative from the claimant: "{claim_narrative}"

Look at the attached photo of vehicle damage. Assess:
1. Does the damage shown look physically consistent with the claim narrative?
2. Are there any inconsistencies worth flagging (e.g. damage location/type
   doesn't match the described incident)?

Respond in this exact format:
CONSISTENT: yes/no/uncertain
CONFIDENCE: 0-1
EXPLANATION: one or two sentences
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw_text = response.content[0].text
    return parse_response(raw_text)


def parse_response(raw_text: str) -> dict:
    lines = raw_text.strip().split("\n")
    result = {"consistent": None, "confidence": None, "explanation": None, "raw": raw_text}

    for line in lines:
        if line.startswith("CONSISTENT:"):
            result["consistent"] = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line.split(":", 1)[1].strip()

    return result


if __name__ == "__main__":
    import sys
    result = check_semantic_consistency(sys.argv[1], sys.argv[2])
    print(result)
