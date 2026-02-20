"""Scene Scanner — Situational Awareness for the Star Trek Computer.

Passively observes the HDMI capture to understand what's happening on the
target machine. Follows the ASMP pattern:
  Sensor: HDMI capture card (1920x1080 MJPEG)
  Detection: OCR + frame differencing + keyword classification
  Knowledge: Structured scene tags (activity, apps, context)

Primitives (named to match Star Trek Computer thinking):
  - scan_frame()        — Extract structured data from a single frame
  - classify_activity() — What's happening: meeting, coding, browsing, idle
  - detect_transition() — Did the scene change between two frames?
  - scan_recording()    — Process an entire MKV recording into tagged segments
"""

import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np


# Screen regions as fractions (0-1) — scale to any resolution
# Defined relative to 1920x1080 reference, applied proportionally
REGIONS_NORMALIZED = {
    "title_bar":    (0.0, 0.0, 1.0, 0.032),      # Browser/app title + tabs
    "url_bar":      (0.0, 0.032, 1.0, 0.069),     # URL or ribbon area
    "banner":       (0.0, 0.0, 1.0, 0.025),       # Top banner (screen sharing, notifications)
    "taskbar":      (0.0, 0.963, 1.0, 1.0),       # Windows taskbar (bottom)
    "content":      (0.0, 0.074, 1.0, 0.963),     # Main content area (full width)
}

# App detection: (pattern, display_name, category)
# Patterns use regex for word-boundary awareness to avoid false positives
import re

APP_SIGNATURES: list[tuple[re.Pattern, str, str]] = [
    # Communication / Meetings (highest priority signals)
    (re.compile(r"microsoft teams|chat \| .* teams|teams meeting", re.I), "Microsoft Teams", "meeting"),
    (re.compile(r"zoom meeting|zoom\.us|zoom video", re.I), "Zoom", "meeting"),
    (re.compile(r"google meet|meet\.google", re.I), "Google Meet", "meeting"),
    (re.compile(r"\bwebex\b", re.I), "Webex", "meeting"),
    (re.compile(r"sharing your screen|stop sharing|screen share", re.I), "Screen Sharing", "meeting"),
    (re.compile(r"\bslack\b", re.I), "Slack", "communication"),
    # Data / Analytics
    (re.compile(r"\bdatabricks\b", re.I), "Databricks", "data_engineering"),
    (re.compile(r"\bdataiku\b", re.I), "Dataiku", "data_science"),
    (re.compile(r"power\s*bi\b", re.I), "Power BI", "bi_analytics"),
    (re.compile(r"\btableau\b", re.I), "Tableau", "bi_analytics"),
    (re.compile(r"\bjupyter\b", re.I), "Jupyter", "data_science"),
    # Productivity (careful with "word" — require context)
    (re.compile(r"microsoft excel|excel -|\.xlsx", re.I), "Excel", "spreadsheet"),
    (re.compile(r"microsoft word|word -|\.docx", re.I), "Word", "document"),
    (re.compile(r"powerpoint|\.pptx", re.I), "PowerPoint", "presentation"),
    (re.compile(r"\boutlook\b", re.I), "Outlook", "email"),
    (re.compile(r"\bonenote\b", re.I), "OneNote", "notes"),
    (re.compile(r"copilot", re.I), "Copilot", "ai_assistant"),
    # Development
    (re.compile(r"visual studio(?! code)", re.I), "Visual Studio", "coding"),
    (re.compile(r"visual studio code|vscode", re.I), "VS Code", "coding"),
    (re.compile(r"\bcursor\b.*(editor|code|file)", re.I), "Cursor", "coding"),
    (re.compile(r"\bgithub\b", re.I), "GitHub", "coding"),
    (re.compile(r"powershell|cmd\.exe|terminal", re.I), "Terminal", "coding"),
    # Cloud / IT
    (re.compile(r"microsoft azure|azure\.com|portal\.azure", re.I), "Azure", "cloud_platform"),
    (re.compile(r"\baws\b.*console|amazon web services|\.aws\.", re.I), "AWS", "cloud_platform"),
    (re.compile(r"servicenow", re.I), "ServiceNow", "it_service"),
    (re.compile(r"\bjira\b", re.I), "Jira", "project_management"),
    # HR / Admin
    (re.compile(r"\bworkday\b", re.I), "Workday", "hr_system"),
    (re.compile(r"employee center", re.I), "Employee Center", "hr_system"),
    # Browser (low priority — almost always present)
    (re.compile(r"\bchrome\b", re.I), "Chrome", "browsing"),
    (re.compile(r"\bedge\b", re.I), "Edge", "browsing"),
    (re.compile(r"\bfirefox\b", re.I), "Firefox", "browsing"),
]

# Activity priority: if multiple detected, pick the most specific
ACTIVITY_PRIORITY = [
    "meeting", "presentation", "coding", "data_engineering",
    "data_science", "bi_analytics", "spreadsheet", "email",
    "document", "it_service", "cloud_platform", "communication",
    "hr_system", "project_management", "notes", "browsing", "idle",
]


@dataclass
class SceneScan:
    """Result of scanning a single frame."""
    timestamp_s: float = 0.0
    ocr_text: dict = field(default_factory=dict)   # region → text lines
    detected_apps: list = field(default_factory=list)
    activity: str = "unknown"
    change_pct: float = 0.0       # % pixels changed from previous frame
    is_transition: bool = False   # significant scene change?
    is_blank: bool = False        # black/no signal?
    raw_text: str = ""            # all OCR text concatenated


@dataclass
class RecordingTag:
    """Tags for an entire recording."""
    filename: str = ""
    duration_s: float = 0.0
    segments: list = field(default_factory=list)  # list of segment dicts
    dominant_activity: str = "unknown"
    detected_apps: list = field(default_factory=list)
    is_meeting: bool = False
    scan_time_s: float = 0.0


def ocr_region(image: np.ndarray) -> list[str]:
    """Run Tesseract OCR on an image region. Returns text lines."""
    tmp = "/tmp/_scene_ocr.jpg"
    cv2.imwrite(tmp, image)
    try:
        result = subprocess.run(
            ["tesseract", tmp, "stdout", "--psm", "6", "-l", "eng"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n")
                 if l.strip() and len(l.strip()) > 2]
        return lines
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def scan_frame(
    frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    timestamp_s: float = 0.0,
    transition_threshold: float = 5.0,
) -> SceneScan:
    """Extract structured scene data from a single frame.

    This is the core primitive — the computer "looking" at the screen
    and understanding what it sees.
    """
    h, w = frame.shape[:2]
    scan = SceneScan(timestamp_s=timestamp_s)

    # Check for blank/no signal
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < 15:
        scan.is_blank = True
        scan.activity = "no_signal"
        return scan

    # OCR key regions (scaled to actual frame dimensions)
    for region_name, (nx1, ny1, nx2, ny2) in REGIONS_NORMALIZED.items():
        rx1 = max(0, int(nx1 * w))
        ry1 = max(0, int(ny1 * h))
        rx2 = min(w, int(nx2 * w))
        ry2 = min(h, int(ny2 * h))
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        region_img = frame[ry1:ry2, rx1:rx2]
        scan.ocr_text[region_name] = ocr_region(region_img)

    # Concatenate all text for keyword search
    scan.raw_text = " ".join(
        " ".join(lines) for lines in scan.ocr_text.values()
    ).lower()

    # Detect apps by regex pattern matching (avoids false positives)
    seen = set()
    for pattern, display_name, category in APP_SIGNATURES:
        if display_name not in seen and pattern.search(scan.raw_text):
            scan.detected_apps.append({"name": display_name, "category": category})
            seen.add(display_name)

    # Classify activity (highest priority match wins)
    categories = {app["category"] for app in scan.detected_apps}
    scan.activity = "idle"
    for activity in ACTIVITY_PRIORITY:
        if activity in categories:
            scan.activity = activity
            break

    # Frame differencing for change detection
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        scan.change_pct = float((diff > 25).sum() / diff.size * 100)
        scan.is_transition = scan.change_pct > transition_threshold

    return scan


def scan_recording(
    mkv_path: str | Path,
    sample_interval_s: float = 30.0,
    transition_threshold: float = 5.0,
) -> RecordingTag:
    """Process an entire MKV recording into tagged segments.

    Samples frames at intervals, runs scene scanning, detects transitions,
    and groups into activity segments.
    """
    mkv_path = Path(mkv_path)
    t0 = time.monotonic()

    tag = RecordingTag(filename=mkv_path.name)

    cap = cv2.VideoCapture(str(mkv_path))
    if not cap.isOpened():
        return tag

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tag.duration_s = total_frames / fps

    interval_frames = int(sample_interval_s * fps)
    if interval_frames < 1:
        interval_frames = 1

    scans: list[SceneScan] = []
    prev_frame = None

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp_s = frame_idx / fps
        scan = scan_frame(
            frame, prev_frame, timestamp_s, transition_threshold,
        )
        scans.append(scan)
        prev_frame = frame
        frame_idx += interval_frames

    cap.release()

    # Group scans into segments (consecutive same-activity)
    if scans:
        segments = []
        current = {
            "start_s": scans[0].timestamp_s,
            "end_s": scans[0].timestamp_s,
            "activity": scans[0].activity,
            "apps": list({a["name"] for a in scans[0].detected_apps}),
            "transitions": 0,
        }

        for scan in scans[1:]:
            if scan.activity == current["activity"] and not scan.is_transition:
                current["end_s"] = scan.timestamp_s
                current["apps"] = list(
                    set(current["apps"]) |
                    {a["name"] for a in scan.detected_apps}
                )
                if scan.is_transition:
                    current["transitions"] += 1
            else:
                current["end_s"] = scan.timestamp_s
                segments.append(current)
                current = {
                    "start_s": scan.timestamp_s,
                    "end_s": scan.timestamp_s,
                    "activity": scan.activity,
                    "apps": list({a["name"] for a in scan.detected_apps}),
                    "transitions": 0,
                }

        segments.append(current)
        tag.segments = segments

        # Dominant activity = longest total duration
        from collections import Counter
        activity_durations = Counter()
        for seg in segments:
            dur = max(seg["end_s"] - seg["start_s"], sample_interval_s)
            activity_durations[seg["activity"]] += dur
        if activity_durations:
            tag.dominant_activity = activity_durations.most_common(1)[0][0]

        # Aggregate apps
        all_apps = set()
        for seg in segments:
            all_apps.update(seg["apps"])
        tag.detected_apps = sorted(all_apps)

        # Meeting detection
        tag.is_meeting = any(
            seg["activity"] == "meeting" for seg in segments
        )

    tag.scan_time_s = round(time.monotonic() - t0, 2)
    return tag
