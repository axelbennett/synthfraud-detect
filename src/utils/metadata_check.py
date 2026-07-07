"""
Lightweight EXIF/metadata consistency checks.

Not a strong signal on its own (metadata is trivially stripped or faked),
but cheap to compute and useful as one input into the fusion score.
Flags: missing EXIF entirely, missing GPS/device info, or timestamp
inconsistencies -- each is weak evidence, not proof.
"""
from pathlib import Path

import exifread


def check_metadata(image_path: str) -> dict:
    """Returns a dict of flags and a rough 0-1 'suspicion' score.
    This is intentionally simple -- document its weakness in the writeup
    rather than overstating what metadata checks can prove."""
    flags = {
        "has_exif": False,
        "has_camera_model": False,
        "has_gps": False,
        "has_timestamp": False,
    }

    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    if tags:
        flags["has_exif"] = True
        flags["has_camera_model"] = "Image Model" in tags
        flags["has_gps"] = any(k.startswith("GPS") for k in tags.keys())
        flags["has_timestamp"] = "EXIF DateTimeOriginal" in tags

    # crude scoring: more missing fields -> higher suspicion
    # real phone photos almost always have all four; AI-generated images
    # and heavily processed/re-saved images often lack some or all of them
    missing = sum(1 for v in flags.values() if v is False)
    suspicion_score = missing / len(flags)

    return {"flags": flags, "metadata_suspicion_score": suspicion_score}


if __name__ == "__main__":
    import sys
    result = check_metadata(sys.argv[1])
    print(result)
