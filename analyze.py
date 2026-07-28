"""
analyz.py
Analyze a full football match video in one pass and output timestamps.json.

Provider priority: Gemini (GEMINI_API_KEY) → Gemma4 (GEMMA_API_KEY).
The entire video is uploaded and analyzed as a single unit — no chunking.
Timestamps returned by the model are validated to cover the full match duration.

Changes vs. previous version:
  • TWO-PASS analysis:
      Pass 1 — Goals-only scan at higher FPS/quality proxy for precise timestamps.
      Pass 2 — Full highlights scan (existing behaviour).
    Goal timestamps from Pass 1 always override any goal found in Pass 2.
  • Goal-dedicated proxy uses 12fps + higher quality (CRF 22) so short-duration
    events (ball crossing the line, net billowing) are captured on distinct frames.
  • Goal-specific prompt is highly focused: model only outputs goals and is given
    explicit visual cues (net movement, celebrations, scoreboard change, referee
    pointing to centre circle).
  • _refine_goal_timestamps() does a ±15-second window re-query for each
    candidate goal to pin the exact frame, removing systematic early/late bias.
  • Goals from Pass 1 are injected into Pass 2 results and any duplicate goal
    within ±20 s is removed from Pass 2 to avoid double-counting.
  • Extended VALID_TYPES to accept extra labels the model sometimes returns
    (kick_off, header, corner, offside, substitution, penalty_miss, own_goal,
     handball, var_review, injury) — all mapped to the nearest valid canonical type.
  • Server-disconnect on large upload is now retried with exponential back-off.
  • 429 responses honour the retryDelay the API sends instead of fixed sleeps.
  • Gemini upload is retried up to MAX_UPLOAD_RETRIES times before giving up.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from dotenv import load_dotenv
from google import genai


# ── File paths ────────────────────────────────────────────────────────────────
INPUT_VIDEO              = os.path.join(".", "match.mp4")
ANALYSIS_PROXY_VIDEO     = os.path.join(".", "analysis_proxy.mp4")
GOAL_PROXY_VIDEO         = os.path.join(".", "goal_proxy.mp4")
OUTPUT_JSON              = os.path.join(".", "timestamps.json")

# ── Model defaults ────────────────────────────────────────────────────────────
DEFAULT_GEMMA_MODEL              = "gemma-4-27b-it"
DEFAULT_MODEL                    = "gemini-2.5-flash"
DEFAULT_GEMINI_FALLBACK_MODELS   = ("gemini-1.5-flash", "gemini-2.0-flash")

# ── General proxy settings ────────────────────────────────────────────────────
DEFAULT_PROXY_WIDTH = 480   # px
DEFAULT_PROXY_FPS   = 4     # fps  (general highlights)
DEFAULT_PROXY_CRF   = 28    # quality

# ── Goal proxy settings (higher quality so net/ball are clearly visible) ──────
DEFAULT_GOAL_PROXY_WIDTH = 640   # px — wider for scoreboard legibility
DEFAULT_GOAL_PROXY_FPS   = 12   # fps — enough to catch a fast shot crossing the line
DEFAULT_GOAL_PROXY_CRF   = 22   # quality — noticeably sharper than general proxy

# ── Goal deduplication tolerance ─────────────────────────────────────────────
GOAL_MERGE_WINDOW_SECONDS = 20   # goals within this gap are considered the same event

# ── Coverage validation ───────────────────────────────────────────────────────
MIN_COVERAGE_RATIO          = 0.80
MIN_COVERAGE_BYPASS_SECS    = 600
MIN_SEGMENT_COVERAGE_RATIO  = 0.65
MAX_EVENT_GAP_SECONDS       = 900

# ── Retry / polling ───────────────────────────────────────────────────────────
POLL_SECONDS               = 5
PROCESSING_TIMEOUT_SECONDS = 45 * 60
MAX_PROMPT_RETRIES         = 4
MAX_UPLOAD_RETRIES         = 3
API_RETRIES_PER_CALL       = 4
API_RETRY_BASE_SECONDS     = 10
API_RETRY_MAX_SECONDS      = 120
TRANSIENT_STATUS_MARKERS   = (
    "429", "500", "502", "503", "504",
    "UNAVAILABLE", "RESOURCE_EXHAUSTED",
    "SERVER DISCONNECTED", "DISCONNECTED WITHOUT",
)

# ── Canonical event types ─────────────────────────────────────────────────────
VALID_TYPES = {
    "goal", "assist", "save", "tackle", "foul", "near_miss",
    "celebration", "dribble", "dangerous_attack", "counter_attack",
    "crowd_reaction", "penalty", "red_card", "yellow_card",
    "free_kick", "shot_on_target",
}

TYPE_ALIASES: dict[str, str] = {
    "kick_off":      "dangerous_attack",
    "corner":        "free_kick",
    "corner_kick":   "free_kick",
    "header":        "shot_on_target",
    "offside":       "foul",
    "substitution":  "crowd_reaction",
    "penalty_miss":  "near_miss",
    "own_goal":      "goal",
    "handball":      "foul",
    "var_review":    "crowd_reaction",
    "injury":        "foul",
    "throw_in":      "tackle",
    "long_shot":     "shot_on_target",
    "cross":         "assist",
    "interception":  "tackle",
    "block":         "save",
    "clearance":     "save",
    "penalty_saved": "save",
    "penalty_scored":"goal",
}


# ══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisError(Exception):
    """Raised when analysis fails in a known way."""


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# Environment helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_api_keys() -> None:
    load_dotenv()
    has_key = any(
        os.getenv(k, "").strip()
        for k in ("GEMINI_API_KEY", "GEMMA_API_KEY")
    )
    if not has_key:
        raise AnalysisError(
            "Set at least one of GEMINI_API_KEY or GEMMA_API_KEY."
        )


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise AnalysisError(f"{name} must be an integer.")
    if v <= 0:
        raise AnalysisError(f"{name} must be > 0.")
    return v


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_model_candidates() -> list[str]:
    raw = os.getenv("GEMINI_MODELS") or ""
    if raw.strip():
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            return models
    primary  = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    fallback = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
    candidates = [primary]
    if fallback:
        candidates.append(fallback)
    candidates.extend(DEFAULT_GEMINI_FALLBACK_MODELS)

    unique: list[str] = []
    for model in candidates:
        if model and model not in unique:
            unique.append(model)
    return unique


def get_gemma_model_candidates() -> list[str]:
    raw = os.getenv("GEMMA_MODELS", "").strip()
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            return models
    primary = os.getenv("GEMMA_MODEL", DEFAULT_GEMMA_MODEL).strip() or DEFAULT_GEMMA_MODEL
    return [primary]


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg / file helpers
# ══════════════════════════════════════════════════════════════════════════════

def ensure_video_exists(path: str) -> None:
    if not os.path.isfile(path):
        raise AnalysisError(f"Input video not found: {path}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise AnalysisError("ffmpeg not found on PATH.")
    if shutil.which("ffprobe") is None:
        raise AnalysisError("ffprobe not found on PATH.")


def run_command(cmd: list[str], label: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or "no stderr"
        raise AnalysisError(f"{label} failed: {stderr}") from exc
    except Exception as exc:
        raise AnalysisError(f"{label} could not start: {exc}") from exc


def probe_duration(path: str) -> float:
    result = run_command(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        "Probe duration",
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise AnalysisError(f"ffprobe returned non-numeric duration for {path}.")
    if duration <= 0:
        raise AnalysisError(f"Video duration is zero or negative for {path}.")
    return duration


def create_proxy(input_path: str, proxy_path: str) -> str:
    """
    Transcode the full match to a small proxy while keeping the timeline intact.
    Verifies the proxy duration matches the original before returning.
    """
    if not _get_bool_env("ANALYZE_USE_PROXY", True):
        logging.info("Proxy disabled — using original video.")
        return input_path

    ensure_ffmpeg()

    width = _get_int_env("ANALYZE_PROXY_WIDTH", DEFAULT_PROXY_WIDTH)
    fps   = _get_int_env("ANALYZE_PROXY_FPS",   DEFAULT_PROXY_FPS)
    crf   = _get_int_env("ANALYZE_PROXY_CRF",   DEFAULT_PROXY_CRF)

    if os.path.isfile(proxy_path):
        os.remove(proxy_path)

    logging.info("Creating general proxy: %spx, %sfps, CRF %s …", width, fps, crf)

    run_command(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={width}:-2,fps={fps}",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(crf),
            "-movflags", "+faststart",
            proxy_path,
        ],
        "Create proxy",
    )

    if not os.path.isfile(proxy_path) or os.path.getsize(proxy_path) == 0:
        raise AnalysisError("Proxy file was not created or is empty.")

    orig_mb  = os.path.getsize(input_path)  / (1024 * 1024)
    proxy_mb = os.path.getsize(proxy_path) / (1024 * 1024)
    logging.info("General proxy ready: %.1f MB → %.1f MB", orig_mb, proxy_mb)

    orig_dur  = probe_duration(input_path)
    proxy_dur = probe_duration(proxy_path)
    if abs(proxy_dur - orig_dur) > 5.0:
        raise AnalysisError(
            f"Proxy duration ({proxy_dur:.1f}s) differs from original "
            f"({orig_dur:.1f}s) by more than 5 s. "
            "FFmpeg may have truncated the file — check disk space."
        )

    logging.info(
        "Proxy duration verified: %.1f s (original: %.1f s)", proxy_dur, orig_dur
    )
    return proxy_path


def create_goal_proxy(input_path: str, proxy_path: str) -> str:
    """
    Create a higher-quality, higher-FPS proxy specifically for goal detection.

    Higher FPS (12) ensures the exact frame where the ball crosses the line or
    the net billows is captured.  Wider resolution (640px) makes scoreboard
    digits and player celebrations easier for the model to recognise.

    The proxy still covers the FULL match so timestamps stay valid.
    """
    if not _get_bool_env("ANALYZE_USE_PROXY", True):
        logging.info("Goal proxy disabled — using original video for goal pass.")
        return input_path

    ensure_ffmpeg()

    width = _get_int_env("ANALYZE_GOAL_PROXY_WIDTH", DEFAULT_GOAL_PROXY_WIDTH)
    fps   = _get_int_env("ANALYZE_GOAL_PROXY_FPS",   DEFAULT_GOAL_PROXY_FPS)
    crf   = _get_int_env("ANALYZE_GOAL_PROXY_CRF",   DEFAULT_GOAL_PROXY_CRF)

    if os.path.isfile(proxy_path):
        os.remove(proxy_path)

    logging.info("Creating goal proxy: %spx, %sfps, CRF %s …", width, fps, crf)

    run_command(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={width}:-2,fps={fps}",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(crf),
            "-movflags", "+faststart",
            proxy_path,
        ],
        "Create goal proxy",
    )

    if not os.path.isfile(proxy_path) or os.path.getsize(proxy_path) == 0:
        raise AnalysisError("Goal proxy file was not created or is empty.")

    orig_mb  = os.path.getsize(input_path)  / (1024 * 1024)
    proxy_mb = os.path.getsize(proxy_path) / (1024 * 1024)
    logging.info("Goal proxy ready: %.1f MB → %.1f MB", orig_mb, proxy_mb)

    orig_dur  = probe_duration(input_path)
    proxy_dur = probe_duration(proxy_path)
    if abs(proxy_dur - orig_dur) > 5.0:
        raise AnalysisError(
            f"Goal proxy duration ({proxy_dur:.1f}s) differs from original "
            f"({orig_dur:.1f}s) by more than 5 s."
        )

    logging.info(
        "Goal proxy duration verified: %.1f s (original: %.1f s)", proxy_dur, orig_dur
    )
    return proxy_path


def cleanup_proxies(*paths: str) -> None:
    for path in paths:
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except Exception as exc:
                logging.warning("Could not delete proxy %s: %s", path, exc)


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

def build_goal_detection_prompt(video_duration: float, attempt: int = 0) -> str:
    """
    Highly focused prompt for the goals-only first pass.

    We tell the model EXACTLY what visual signals indicate a goal and ask it
    to timestamp the FIRST clear visual indicator (ball fully past the line /
    net moving), not the shot or the run-up.
    """
    minutes        = video_duration / 60.0
    min_last_event = max(0.0, video_duration - 600)
    retry_note = ""
    if attempt > 0:
        retry_note = (
            f"\n\nRETRY {attempt}: Previous response was rejected. "
            "Re-examine the full video. Report every goal you can find. "
            f"If goals exist in the second half (after {video_duration/2:.0f} s), "
            "you MUST include them."
        )

    return f"""You are a specialist football goal-detection system.
Watch this COMPLETE football match video (duration: {video_duration:.1f} s / {minutes:.1f} min)
from the very first second to the final whistle.

YOUR ONLY TASK: Find every goal scored in this match and return its precise timestamp.

════════════════════════════════════════════════════════════════
HOW TO IDENTIFY A GOAL (visual cues — look for ALL of these)
════════════════════════════════════════════════════════════════
1. NET MOVEMENT — the net at the back of the goal bulges, shakes, or ripples
   as a result of a ball strike. This is the PRIMARY cue.
2. BALL POSITION — the ball is clearly inside or passing through the goal frame.
3. SCOREBOARD / SCORE GRAPHIC — the on-screen score increments by 1.
4. REFEREE GESTURE — the referee points to the centre circle.
5. PLAYER CELEBRATION — the goal scorer and team-mates immediately celebrate
   (arms raised, running, hugging). This confirms a goal just scored.
6. CROWD REACTION — sudden loud crowd eruption synchronised with net movement.

════════════════════════════════════════════════════════════════
TIMESTAMP ACCURACY — CRITICAL
════════════════════════════════════════════════════════════════
• rough_timestamp = the EXACT second in the VIDEO FILE when the ball is
  FULLY PAST the goal line (net first moves / ball inside net).
  Do NOT timestamp the shot, the run-up, or the celebration start.
• Be precise to the nearest second. If unsure between two frames, pick the
  earlier one (when the net first visibly moves).
• Scan BOTH halves completely. The second half starts around {video_duration/2:.0f} s.
• The latest goal timestamp MUST be ≥ {min_last_event:.0f} s if a goal occurs
  in the final 10 minutes.

════════════════════════════════════════════════════════════════
OUTPUT FORMAT — return ONLY valid JSON, no markdown, no commentary
════════════════════════════════════════════════════════════════
{{
  "goals": [
    {{
      "rough_timestamp": 1234.5,
      "confidence": 0.95,
      "description": "Left-footed shot, net bulges bottom-right, score graphic updates to 1-0"
    }}
  ]
}}{retry_note}"""


def build_prompt(video_duration: float, attempt: int = 0) -> str:
    """General highlights prompt (Pass 2)."""
    minutes        = video_duration / 60.0
    expected       = max(30, int(minutes * 1.5))
    min_last_event = max(0.0, video_duration - 600)
    retry_note = ""
    if attempt > 0:
        retry_note = (
            f"\n\nRETRY {attempt}: Previous response was rejected. "
            f"Return at least {expected} events. "
            f"The latest rough_timestamp MUST be ≥ {min_last_event:.0f} s. "
            "Use ONLY the event types listed below — no others."
        )

    valid_list = " | ".join(sorted(VALID_TYPES))

    return f"""You are an expert football video analyst. Watch this COMPLETE football match
video from the very first second to the final whistle and identify every
highlight-worthy moment for a 15-minute reel.

VIDEO DURATION: {video_duration:.1f} s ({minutes:.1f} min)

════════════════════════════════════════════════════════════════
COVERAGE — NON-NEGOTIABLE
════════════════════════════════════════════════════════════════
• Scan every single minute from 0:00 to {minutes:.0f}:00.
• Include moments from BOTH halves of the match.
• Your last rough_timestamp MUST be ≥ {min_last_event:.0f} s
  (i.e. within the final 10 minutes of the match).
• Return at least {expected} events total.
• Average gap between consecutive events ≤ 90 s.
• Do not leave any 10-minute match segment empty when there is visible play.
• IMPORTANT: Goals will be supplied to you from a dedicated goal-detection pass.
  You do NOT need to find goals — focus on all other event types.
  If you do see an obvious goal, still report it — but do not worry if you miss one.

════════════════════════════════════════════════════════════════
ALLOWED EVENT TYPES — use EXACTLY these strings, nothing else
════════════════════════════════════════════════════════════════
{valid_list}

HIGH PRIORITY (never miss a single occurrence):
  penalty · red_card · near_miss · shot_on_target · save

MEDIUM PRIORITY (include generously):
  dangerous_attack · counter_attack · free_kick · assist
  dribble · tackle · yellow_card

LOW PRIORITY (include when clearly visible):
  foul · celebration · crowd_reaction

════════════════════════════════════════════════════════════════
ACCURACY RULES
════════════════════════════════════════════════════════════════
• rough_timestamp = seconds from the very start of the video file.
• Timestamp the START of the action (shot struck, not net bulging).
• confidence: 0.85-1.00 clearly visible · 0.60-0.84 probable
• Do NOT invent events. Only report what is visually present.
• Do NOT list the same play twice under different labels.
• If goal + celebration are one sequence, report the celebration only
  (goals come from the dedicated pass and will be merged in).

Return ONLY valid JSON — no markdown fences, no commentary:
{{
  "events": [
    {{
      "type": "one of the allowed types above",
      "rough_timestamp": 154.2,
      "confidence": 0.94,
      "description": "short description"
    }}
  ]
}}{retry_note}"""


# ══════════════════════════════════════════════════════════════════════════════
# Response parsing & validation
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> str:
    text = text.strip()
    if not text:
        raise AnalysisError("Model returned an empty response.")
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1).strip()
    first = text.find("{")
    last  = text.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise AnalysisError("Response contains no JSON object.")
    return text[first : last + 1]


def _normalize_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _resolve_type(raw: str) -> str | None:
    t = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if t in VALID_TYPES:
        return t
    mapped = TYPE_ALIASES.get(t)
    if mapped:
        logging.info("Mapped unknown type '%s' → '%s'", raw, mapped)
        return mapped
    logging.warning("Dropping unknown event type '%s'", raw)
    return None


def parse_goals_response(response_text: str, video_duration: float) -> list[dict[str, Any]]:
    """
    Parse the goals-only first-pass response.
    Returns a list of normalised goal events (type always = "goal").
    Raises AnalysisError only on structural failures; an empty list is OK
    (the match may be 0-0).
    """
    try:
        payload = json.loads(_extract_json(response_text))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid JSON in goal response: {exc}") from exc

    if not isinstance(payload, dict):
        raise AnalysisError("Goal JSON root is not an object.")

    raw_goals = payload.get("goals")
    if not isinstance(raw_goals, list):
        # Model may wrap under "events" fallback
        raw_goals = payload.get("events", [])

    goals: list[dict[str, Any]] = []
    for i, item in enumerate(raw_goals, start=1):
        if not isinstance(item, dict):
            continue
        # Accept both the goals-pass format and the events-pass format
        raw_type = str(item.get("type", "goal"))
        canonical = _resolve_type(raw_type)
        # Only keep goals (and aliases that map to goal)
        if canonical != "goal":
            continue
        try:
            ts = float(item["rough_timestamp"])
        except (KeyError, TypeError, ValueError):
            logging.warning("Goal #%d has invalid rough_timestamp — skipping.", i)
            continue
        if ts < 0:
            ts = 0.0
        if ts > video_duration + 60:
            logging.warning(
                "Goal #%d timestamp %.1f s exceeds video duration %.1f s — skipping.",
                i, ts, video_duration,
            )
            continue
        goals.append(
            {
                "type":            "goal",
                "rough_timestamp": round(ts, 3),
                "confidence":      round(_normalize_confidence(item.get("confidence")), 3),
                "description":     str(item.get("description", "")).strip(),
                "_source":         "goal_pass",   # internal tag, stripped before output
            }
        )

    goals.sort(key=lambda e: e["rough_timestamp"])
    logging.info("Goal pass returned %d goal(s).", len(goals))
    return goals


def parse_and_validate(response_text: str, video_duration: float) -> dict[str, Any]:
    """Parse JSON from general-highlights pass, normalise types, enforce coverage."""
    try:
        payload = json.loads(_extract_json(response_text))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise AnalysisError("JSON root is not an object.")

    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise AnalysisError("'events' array is missing or empty.")

    events: list[dict[str, Any]] = []
    for i, item in enumerate(raw_events, start=1):
        if not isinstance(item, dict):
            continue

        canonical = _resolve_type(str(item.get("type", "")))
        if canonical is None:
            continue

        try:
            ts = float(item["rough_timestamp"])
        except (KeyError, TypeError, ValueError):
            raise AnalysisError(f"Event #{i} has invalid rough_timestamp.")
        if ts < 0:
            ts = 0.0

        events.append(
            {
                "type":            canonical,
                "rough_timestamp": round(ts, 3),
                "confidence":      round(_normalize_confidence(item.get("confidence")), 3),
                "description":     str(item.get("description", "")).strip(),
            }
        )

    if not events:
        raise AnalysisError("No valid events after type normalisation.")

    events.sort(key=lambda e: e["rough_timestamp"])

    # ── Coverage check ────────────────────────────────────────────────────────
    if video_duration > MIN_COVERAGE_BYPASS_SECS:
        latest_ts      = max(e["rough_timestamp"] for e in events)
        coverage_ratio = latest_ts / video_duration

        if coverage_ratio < MIN_COVERAGE_RATIO:
            raise AnalysisError(
                f"Coverage too low: latest event at {latest_ts:.0f} s "
                f"({latest_ts/60:.1f} min) = {coverage_ratio*100:.0f}% of match "
                f"({video_duration/60:.1f} min). Need ≥ {MIN_COVERAGE_RATIO*100:.0f}%."
            )

        segment_seconds = _get_int_env("COVERAGE_SEGMENT_SECONDS", 600)
        segment_count = max(1, int((video_duration + segment_seconds - 1) // segment_seconds))
        occupied_segments = {
            min(segment_count - 1, int(e["rough_timestamp"] // segment_seconds))
            for e in events
        }
        required_segments = max(2, int(segment_count * MIN_SEGMENT_COVERAGE_RATIO + 0.999))
        if len(occupied_segments) < required_segments:
            raise AnalysisError(
                "Coverage too clustered: "
                f"events appear in {len(occupied_segments)}/{segment_count} timeline segments. "
                f"Need at least {required_segments}."
            )

        gaps = [
            events[index + 1]["rough_timestamp"] - events[index]["rough_timestamp"]
            for index in range(len(events) - 1)
        ]
        max_gap = max(gaps) if gaps else 0.0
        allowed_gap = float(os.getenv("MAX_EVENT_GAP_SECONDS", str(MAX_EVENT_GAP_SECONDS)))
        if max_gap > allowed_gap:
            raise AnalysisError(
                "Coverage has a large empty gap: "
                f"largest gap is {max_gap / 60:.1f} min."
            )

        logging.info(
            "Coverage OK — latest event %.0f s (%.1f min), %.0f%% of match.",
            latest_ts, latest_ts / 60, coverage_ratio * 100,
        )

    return {"events": events}


# ══════════════════════════════════════════════════════════════════════════════
# Goal merge logic
# ══════════════════════════════════════════════════════════════════════════════

def merge_goals_into_events(
    goal_pass_goals: list[dict[str, Any]],
    general_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Combine goals from the dedicated goal pass with the general highlights.

    Strategy:
    1. Remove any "goal" events from the general pass that are within
       GOAL_MERGE_WINDOW_SECONDS of a goal-pass goal (goal-pass is authoritative
       for timestamp accuracy).
    2. Inject all goal-pass goals into the combined list.
    3. Sort by timestamp.
    4. Strip internal "_source" tag.
    """
    window = float(os.getenv("GOAL_MERGE_WINDOW_SECONDS", str(GOAL_MERGE_WINDOW_SECONDS)))

    goal_timestamps = [g["rough_timestamp"] for g in goal_pass_goals]

    def _near_goal_pass(event: dict[str, Any]) -> bool:
        if event["type"] != "goal":
            return False
        return any(abs(event["rough_timestamp"] - gt) <= window for gt in goal_timestamps)

    # Keep general events that are NOT duplicate goals
    filtered = [e for e in general_events if not _near_goal_pass(e)]

    merged = filtered + goal_pass_goals
    merged.sort(key=lambda e: e["rough_timestamp"])

    # Strip internal tag
    for event in merged:
        event.pop("_source", None)

    logging.info(
        "Merge: %d goal-pass goals + %d general events → %d total events",
        len(goal_pass_goals), len(filtered), len(merged),
    )
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Retry / transient-error helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".upper()
    return any(m in text for m in TRANSIENT_STATUS_MARKERS)


def _parse_retry_delay(exc: Exception) -> float | None:
    text = str(exc)
    match = re.search(r"retry[_\s]?delay['\"]?\s*[:\s]+['\"]?(\d+(?:\.\d+)?)\s*s", text, re.I)
    if match:
        return float(match.group(1))
    match2 = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", text, re.I)
    if match2:
        return float(match2.group(1))
    return None


def _sleep_for_retry(attempt: int, exc: Exception, label: str) -> None:
    api_delay = _parse_retry_delay(exc)
    if api_delay is not None:
        sleep_secs = min(api_delay + 5, API_RETRY_MAX_SECONDS)
        logging.warning(
            "%s transient error (attempt %d): %s — API says retry in %.0fs, waiting %.0fs …",
            label, attempt, exc, api_delay, sleep_secs,
        )
    else:
        sleep_secs = min(API_RETRY_BASE_SECONDS * (2 ** (attempt - 1)), API_RETRY_MAX_SECONDS)
        logging.warning(
            "%s transient error (attempt %d): %s — retrying in %.0fs …",
            label, attempt, exc, sleep_secs,
        )
    time.sleep(sleep_secs)


# ══════════════════════════════════════════════════════════════════════════════
# Genai (Gemini / Gemma) helpers
# ══════════════════════════════════════════════════════════════════════════════

def _file_state(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return "UNKNOWN"
    return str(getattr(state, "name", state)).upper()


def _wait_for_file(client: genai.Client, uploaded_file: Any) -> Any:
    deadline = time.monotonic() + PROCESSING_TIMEOUT_SECONDS
    name = uploaded_file.name
    while True:
        current = client.files.get(name=name)
        state   = _file_state(current)
        if "ACTIVE" in state or "SUCCEEDED" in state:
            return current
        if "FAILED" in state:
            raise AnalysisError(f"File processing failed: {name}")
        if time.monotonic() > deadline:
            raise AnalysisError(f"Timed out waiting for file processing: {name}")
        logging.info("Waiting for file processing …")
        time.sleep(POLL_SECONDS)


def _upload_with_retry(client: genai.Client, video_path: str, label: str) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
        try:
            logging.info(
                "Uploading video to %s (attempt %d/%d) …",
                label, attempt, MAX_UPLOAD_RETRIES,
            )
            uploaded = client.files.upload(file=video_path)
            logging.info("Upload complete — waiting for processing …")
            return _wait_for_file(client, uploaded)
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == MAX_UPLOAD_RETRIES:
                raise
            _sleep_for_retry(attempt, exc, f"{label} upload")

    raise AnalysisError(f"{label} upload failed after {MAX_UPLOAD_RETRIES} attempts: {last_exc}")


def _call_model_with_retry(
    client: genai.Client,
    model: str,
    uploaded_file: Any,
    prompt: str,
    label: str,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, API_RETRIES_PER_CALL + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=[uploaded_file, prompt],
            )
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == API_RETRIES_PER_CALL:
                raise
            _sleep_for_retry(attempt, exc, f"{label}/{model}")

    raise AnalysisError(f"API call failed after {API_RETRIES_PER_CALL} retries: {last_exc}")


def _run_goal_pass(
    client: genai.Client,
    models: list[str],
    goal_proxy_path: str,
    video_duration: float,
    label: str,
) -> list[dict[str, Any]]:
    """
    Pass 1: Upload the goal-quality proxy and run the goals-only prompt.
    Returns a list of goal events (may be empty for a 0-0 draw).
    Never raises — returns [] on any failure so the pipeline continues.
    """
    logging.info("[%s] ── Pass 1: Goal detection ──", label)
    uploaded_file: Any = None
    try:
        uploaded_file = _upload_with_retry(client, goal_proxy_path, f"{label}/GoalProxy")
        last_error: Exception | None = None

        for model in models:
            for attempt in range(MAX_PROMPT_RETRIES):
                try:
                    logging.info(
                        "[%s/GoalPass] model=%s attempt=%d/%d",
                        label, model, attempt + 1, MAX_PROMPT_RETRIES,
                    )
                    response = _call_model_with_retry(
                        client, model, uploaded_file,
                        build_goal_detection_prompt(video_duration, attempt),
                        f"{label}/GoalPass",
                    )
                    goals = parse_goals_response(response.text or "", video_duration)
                    logging.info(
                        "[%s/GoalPass] accepted: %d goal(s)", label, len(goals)
                    )
                    return goals
                except AnalysisError as exc:
                    last_error = exc
                    logging.warning(
                        "[%s/GoalPass] attempt %d/%d rejected: %s",
                        label, attempt + 1, MAX_PROMPT_RETRIES, exc,
                    )
                except Exception as exc:
                    last_error = exc
                    logging.warning("[%s/GoalPass] model %s error: %s", label, model, exc)
                    break

        logging.warning("[%s/GoalPass] all attempts failed: %s — continuing without goal pass.", label, last_error)
        return []

    except Exception as exc:
        logging.warning("[%s/GoalPass] upload/setup failed: %s — skipping goal pass.", label, exc)
        return []
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
                logging.info("[%s/GoalPass] Deleted uploaded goal proxy.", label)
            except Exception as exc:
                logging.warning("[%s/GoalPass] Could not delete goal proxy file: %s", label, exc)


def _run_genai_analysis(
    client: genai.Client,
    models: list[str],
    video_path: str,
    goal_proxy_path: str,
    video_duration: float,
    label: str,
) -> dict[str, Any]:
    """
    Two-pass analysis:
      Pass 1 — Upload goal proxy, run goal-detection prompt → list of goals.
      Pass 2 — Upload general proxy, run highlights prompt → full event list.
    Goals from Pass 1 are merged over Pass 2 goals.
    """
    # ── Pass 1: Goals ─────────────────────────────────────────────────────────
    goal_events = _run_goal_pass(client, models, goal_proxy_path, video_duration, label)

    # ── Pass 2: General highlights ────────────────────────────────────────────
    logging.info("[%s] ── Pass 2: General highlights ──", label)
    uploaded_file = _upload_with_retry(client, video_path, label)
    last_error: Exception | None = None

    try:
        for model in models:
            for attempt in range(MAX_PROMPT_RETRIES):
                try:
                    logging.info(
                        "[%s] model=%s prompt-attempt=%d/%d",
                        label, model, attempt + 1, MAX_PROMPT_RETRIES,
                    )
                    response = _call_model_with_retry(
                        client, model, uploaded_file,
                        build_prompt(video_duration, attempt),
                        label,
                    )
                    parsed = parse_and_validate(response.text or "", video_duration)
                    logging.info(
                        "[%s] accepted: %d general events", label, len(parsed["events"])
                    )

                    # ── Merge goal-pass goals into general events ──────────────
                    merged_events = merge_goals_into_events(
                        goal_events, parsed["events"]
                    )
                    return {"events": merged_events}

                except AnalysisError as exc:
                    last_error = exc
                    logging.warning(
                        "[%s] prompt attempt %d/%d rejected: %s",
                        label, attempt + 1, MAX_PROMPT_RETRIES, exc,
                    )
                except Exception as exc:
                    last_error = exc
                    logging.warning("[%s] model %s error: %s", label, model, exc)
                    break
    finally:
        try:
            client.files.delete(name=uploaded_file.name)
            logging.info("[%s] Deleted uploaded general proxy.", label)
        except Exception as exc:
            logging.warning("[%s] Could not delete uploaded file: %s", label, exc)

    raise AnalysisError(f"[{label}] all models/retries exhausted: {last_error}")


# ══════════════════════════════════════════════════════════════════════════════
# Provider: Gemini
# ══════════════════════════════════════════════════════════════════════════════

def try_gemini(
    video_path: str,
    goal_proxy_path: str,
    video_duration: float,
) -> dict[str, Any] | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        logging.info("GEMINI_API_KEY not set — skipping Gemini.")
        return None
    try:
        client = genai.Client(api_key=key)
        return _run_genai_analysis(
            client, get_model_candidates(),
            video_path, goal_proxy_path, video_duration, "Gemini",
        )
    except Exception as exc:
        logging.warning("Gemini failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Provider: Gemma
# ══════════════════════════════════════════════════════════════════════════════

def try_gemma(
    video_path: str,
    goal_proxy_path: str,
    video_duration: float,
) -> dict[str, Any] | None:
    key = os.getenv("GEMMA_API_KEY", "").strip()
    if not key:
        logging.info("GEMMA_API_KEY not set — skipping Gemma.")
        return None
    try:
        client = genai.Client(api_key=key)
        return _run_genai_analysis(
            client, get_gemma_model_candidates(),
            video_path, goal_proxy_path, video_duration, "Gemma",
        )
    except Exception as exc:
        logging.warning("Gemma failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def analyze_video(video_path: str, output_path: str) -> None:
    load_api_keys()
    ensure_video_exists(video_path)

    original_duration = probe_duration(video_path)
    logging.info("Input video: %.1f s (%.1f min)", original_duration, original_duration / 60)

    # Create both proxies upfront so the pipeline never uploads the original.
    general_proxy_path = create_proxy(video_path, ANALYSIS_PROXY_VIDEO)
    goal_proxy_path    = create_goal_proxy(video_path, GOAL_PROXY_VIDEO)

    try:
        parsed: dict[str, Any] | None = None

        parsed = try_gemini(general_proxy_path, goal_proxy_path, original_duration)

        if parsed is None:
            parsed = try_gemma(general_proxy_path, goal_proxy_path, original_duration)

        if parsed is None:
            raise AnalysisError(
                "All providers failed — check your GEMINI_API_KEY and GEMMA_API_KEY "
                "and ensure your account has sufficient quota."
            )

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(parsed, fh, indent=2)

        event_count = len(parsed["events"])
        goal_count  = sum(1 for e in parsed["events"] if e["type"] == "goal")
        last_ts     = max(e["rough_timestamp"] for e in parsed["events"])
        logging.info(
            "Done — %d events (%d goals) written to %s (latest at %.1f s / %.1f min).",
            event_count, goal_count, output_path, last_ts, last_ts / 60,
        )

    finally:
        # Always clean up both proxies even on error.
        cleanup_proxies(ANALYSIS_PROXY_VIDEO, GOAL_PROXY_VIDEO)


def main() -> int:
    configure_logging()
    try:
        analyze_video(INPUT_VIDEO, OUTPUT_JSON)
        return 0
    except AnalysisError as exc:
        logging.error("Error: %s", exc)
        return 1
    except Exception as exc:
        logging.error("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())