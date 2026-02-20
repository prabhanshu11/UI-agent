# /// script
# dependencies = []
# requires-python = ">=3.13"
# ///
"""Query human-validated experience tags for agent context injection.

Scans logs/experiences/ for validation.json files and returns a compact
summary of tagged experiences. Designed to be called by the Claude Code
experience-feedback skill during live sessions.

Usage:
    uv run scripts/query-experience-tags.py                    # All tagged
    uv run scripts/query-experience-tags.py --verdict problematic  # Filter by verdict
    uv run scripts/query-experience-tags.py --unreviewed       # List unreviewed sessions
    uv run scripts/query-experience-tags.py --session 20260220_155709  # Single session detail
    uv run scripts/query-experience-tags.py --json             # Machine-readable output
"""

import json
import sys
from datetime import datetime
from pathlib import Path

EXPERIENCES_DIR = Path(__file__).parent.parent / "logs" / "experiences"
PASSTHROUGH_DIR = EXPERIENCES_DIR / "passthrough"


def find_sessions() -> list[dict]:
    """Scan for all experience sessions with their metadata and tags."""
    sessions = []

    # Passthrough sessions (subdirs in passthrough/)
    if PASSTHROUGH_DIR.exists():
        for item in sorted(PASSTHROUGH_DIR.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                session = _parse_session_dir(item, session_type="passthrough")
                if session:
                    sessions.append(session)

    # Standalone sessions (root-level .jsonl files)
    if EXPERIENCES_DIR.exists():
        for jsonl in sorted(EXPERIENCES_DIR.glob("*.jsonl")):
            session = _parse_standalone_session(jsonl)
            if session:
                sessions.append(session)

    return sessions


def _parse_session_dir(path: Path, session_type: str) -> dict | None:
    """Parse a session directory for metadata and validation tags."""
    name = path.name
    # Format: YYYYMMDD_HHMMSS_label
    parts = name.split("_", 2)
    if len(parts) < 2:
        return None

    session_id = f"{parts[0]}_{parts[1]}"
    label = parts[2] if len(parts) > 2 else ""

    # Look for validation.json
    validation_file = path / "validation.json"
    tags = None
    reviewed_at = None
    notes = None
    if validation_file.exists():
        try:
            data = json.loads(validation_file.read_text())
            tags = data.get("tags", {})
            reviewed_at = data.get("reviewed_at")
            notes = data.get("notes")
        except (json.JSONDecodeError, OSError):
            pass

    # Look for event log to get metadata
    event_count = 0
    duration_s = 0
    started_at = ""
    for jsonl in path.parent.glob(f"*{session_id.replace('_', '')}*.jsonl"):
        # Actually the JSONL is in parent dir (passthrough/)
        pass
    # Try parent dir for the JSONL matching this session
    for jsonl in path.parent.glob("*.jsonl"):
        if session_id.replace("_", "").replace("20", "", 1) in jsonl.stem.replace("-", "").replace("_", ""):
            lines = jsonl.read_text().strip().split("\n")
            for line in lines:
                try:
                    evt = json.loads(line)
                    if evt.get("event") == "session_start":
                        started_at = evt.get("iso", "")
                    if evt.get("event") == "session_end":
                        duration_s = evt.get("details", {}).get("duration_s", 0)
                    if evt.get("event") not in ("session_start", "session_end"):
                        event_count += 1
                except json.JSONDecodeError:
                    pass
            break

    # Check for narrative
    has_narrative = any(path.glob("**/narrative*"))

    # Check for screenshots
    screenshots_dir = path / "screenshots"
    screenshot_count = len(list(screenshots_dir.glob("*.png"))) if screenshots_dir.exists() else 0

    return {
        "session_id": session_id,
        "label": label,
        "type": session_type,
        "started_at": started_at,
        "duration_s": duration_s,
        "event_count": event_count,
        "screenshot_count": screenshot_count,
        "has_narrative": has_narrative,
        "reviewed": tags is not None,
        "tags": tags,
        "reviewed_at": reviewed_at,
        "notes": notes,
        "directory": str(path),
    }


def _parse_standalone_session(jsonl_path: Path) -> dict | None:
    """Parse a standalone session JSONL file."""
    lines = jsonl_path.read_text().strip().split("\n")
    if not lines:
        return None

    session_id = jsonl_path.stem.replace("-", "_")
    started_at = ""
    duration_s = 0
    event_count = 0

    for line in lines:
        try:
            evt = json.loads(line)
            if evt.get("event") == "session_start":
                started_at = evt.get("iso", "")
            elif evt.get("event") == "session_end":
                duration_s = evt.get("details", {}).get("duration_s", 0)
            else:
                event_count += 1
        except json.JSONDecodeError:
            pass

    # Check for validation.json in a sibling directory
    # Standalone sessions may not have a directory
    return {
        "session_id": session_id,
        "label": "",
        "type": "standalone",
        "started_at": started_at,
        "duration_s": duration_s,
        "event_count": event_count,
        "screenshot_count": 0,
        "has_narrative": False,
        "reviewed": False,
        "tags": None,
        "reviewed_at": None,
        "notes": None,
        "directory": str(jsonl_path.parent),
    }


def format_tags_compact(tags: dict) -> str:
    """Format tags as a compact one-line string."""
    if not tags:
        return "unreviewed"
    parts = []
    symbols = {
        "correct": "+", "incorrect": "X", "partial": "~",
        "orthogonal": "?", "lucky": "L", "premature": "P",
        "excessive": "E", "missed": "M", "failed": "F",
        "stalled": "S", "exploratory": "...",
        "high": "H", "medium": "M", "low": "L",
        "valuable": "++", "neutral": "=", "problematic": "!!", "exemplary": "***",
        "unknown": "?",
    }
    for dim in ["awareness", "conclusion", "decision", "execution", "direction", "certainty", "verdict"]:
        val = tags.get(dim, "")
        if val:
            sym = symbols.get(val, val[:3])
            parts.append(f"{dim[:4]}:{sym}")
    return " | ".join(parts)


def main():
    args = sys.argv[1:]

    json_output = "--json" in args
    unreviewed_only = "--unreviewed" in args
    session_filter = None
    verdict_filter = None

    i = 0
    while i < len(args):
        if args[i] == "--session" and i + 1 < len(args):
            session_filter = args[i + 1]
            i += 2
        elif args[i] == "--verdict" and i + 1 < len(args):
            verdict_filter = args[i + 1]
            i += 2
        else:
            i += 1

    sessions = find_sessions()

    # Apply filters
    if unreviewed_only:
        sessions = [s for s in sessions if not s["reviewed"]]
    if verdict_filter:
        sessions = [s for s in sessions if s.get("tags", {}) and s["tags"].get("verdict") == verdict_filter]
    if session_filter:
        sessions = [s for s in sessions if session_filter in s["session_id"]]

    if json_output:
        print(json.dumps(sessions, indent=2))
        return

    # Single session detail
    if session_filter and len(sessions) == 1:
        s = sessions[0]
        print(f"Session: {s['session_id']} ({s['type']})")
        print(f"Label:   {s['label']}")
        print(f"Started: {s['started_at']}")
        print(f"Duration: {s['duration_s']}s | Events: {s['event_count']} | Screenshots: {s['screenshot_count']}")
        if s["tags"]:
            print(f"\nValidation Tags:")
            for dim, val in s["tags"].items():
                print(f"  {dim:12s} = {val}")
            if s["notes"]:
                print(f"\nNotes: {s['notes']}")
            print(f"\nReviewed: {s['reviewed_at']}")
        else:
            print("\nNot yet reviewed.")
        return

    # Summary table
    reviewed = [s for s in sessions if s["reviewed"]]
    unreviewed = [s for s in sessions if not s["reviewed"]]

    print(f"Experience Validation Summary")
    print(f"{'='*70}")
    print(f"Total: {len(sessions)} | Reviewed: {len(reviewed)} | Unreviewed: {len(unreviewed)}")
    print()

    if reviewed:
        print("REVIEWED EXPERIENCES:")
        print(f"{'Session':<18} {'Label':<16} {'Dur':>5} {'Evts':>5}  Tags")
        print(f"{'-'*18} {'-'*16} {'-'*5} {'-'*5}  {'-'*30}")
        for s in reviewed:
            tags_str = format_tags_compact(s["tags"])
            print(f"{s['session_id']:<18} {s['label']:<16} {s['duration_s']:>4.0f}s {s['event_count']:>5}  {tags_str}")
            if s.get("notes"):
                print(f"{'':>47} note: {s['notes'][:60]}")
        print()

    if unreviewed and not verdict_filter:
        print("UNREVIEWED EXPERIENCES:")
        print(f"{'Session':<18} {'Label':<16} {'Type':<12} {'Dur':>5} {'Evts':>5}")
        print(f"{'-'*18} {'-'*16} {'-'*12} {'-'*5} {'-'*5}")
        for s in unreviewed:
            print(f"{s['session_id']:<18} {s['label']:<16} {s['type']:<12} {s['duration_s']:>4.0f}s {s['event_count']:>5}")

    if not sessions:
        print("No experiences found matching filters.")


if __name__ == "__main__":
    main()
