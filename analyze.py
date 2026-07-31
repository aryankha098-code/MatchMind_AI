"""
analyze.py
Analyze a full football match video and output timestamps.json.

Provider priority: Gemini (GEMINI_API_KEY) -> Gemma (GEMMA_API_KEY).

── Why this version is different (v4 — chunked analysis) ─────────────────────
Three concrete accuracy bugs were found in the previous single-pass design:

  1. WRONG ASSUMPTION ABOUT FPS.
     Gemini's File API samples uploaded video at a fixed 1 frame/second by
     default, REGARDLESS of the frame rate the file was encoded at. Encoding
     a "12fps goal proxy" bought nothing — Gemini still only saw 1 frame per
     second unless the request explicitly overrides this via
     `video_metadata.fps` on the request Part. This version sets that field
     explicitly on every call (see `_video_part`).

  2. LOCALIZING A TIMESTAMP ACROSS A 90-MINUTE VIDEO IN ONE CALL IS INHERENTLY
     IMPRECISE. Long-video temporal grounding is a known weak point for
     current vision-language models — the model is effectively estimating
     "roughly how far through the video" an event was, which is why the same
     goal might get reported anywhere from 5-30s off. This version instead
     CHUNKS the match into short (default 8 min), overlapping clips and asks
     for timestamps RELATIVE TO EACH CLIP. Grounding a timestamp inside an
     8-minute clip is a much easier problem than inside a 100-minute one.
     Chunk-relative timestamps are converted back to absolute match time
     after each chunk is analyzed, and overlap regions are de-duplicated.

  3. THE MODEL HAS NO GROUND TRUTH FOR "WHAT SECOND IS THIS". This version
     burns a running HH:MM:SS clock into the top-left corner of every chunk
     proxy (ffmpeg drawtext), relative to the start of that chunk. Instead of
     estimating time from pacing/context, the model can read the exact
     second directly off the frame -- turning a hard temporal-grounding
     problem into an easy OCR-style reading problem. The bundled font at
     assets/fonts/DejaVuSans-Bold.ttf is used so this works identically on
     Windows, Docker, and any other host without relying on system fonts.

The frame-accurate CV refinement stage (event_refiner.py) is unchanged and
still runs after this script — it remains the right layer for sub-second
correction; this script's job is to get the LLM's rough timestamp close
enough (typically within a couple of seconds) for that refiner to work with.

── v5 changes — fixing missed/misidentified goals ────────────────────────────
Three more bugs, found by comparing this script against the actual Gemini
video-understanding docs, that were actively hurting goal accuracy:

  4. AUDIO WAS BEING THROWN AWAY. `create_chunk_proxy` ran ffmpeg with `-an`,
     stripping the audio track entirely before upload. Gemini processes audio
     and video as separate streams, and for goal detection the crowd roar /
     commentator reaction is often the single clearest, fastest signal that a
     goal just happened — much more reliable than trying to visually catch a
     net bulge that may be obscured by players or a quick replay cut. The
     proxy now keeps a low-bitrate mono audio track (`ANALYZE_KEEP_AUDIO=true`
     by default), and the goal prompt explicitly tells the model to listen
     for it.

  5. THE PROXY WAS TOO COMPRESSED TO SEE THE THING IT WAS BEING ASKED TO
     FIND. 640px width at CRF 23 blurs out net movement and ball position for
     shots on the far side of the pitch — exactly the detail a goal call
     depends on. Bumped to 960px / CRF 18.

  6. FLASH WAS THE DEFAULT MODEL FOR A PASS WHERE ACCURACY MATTERS MOST.
     The goal pass now tries a Pro-tier model first (see v6 below), falling
     back through the same flash chain as the general pass only if Pro
     fails or isn't available.

── v6 changes — squeezing more accuracy out of what v5 already fixed ────────
  7. RESOLUTION CEILING. Gemini tiles frames into 768x768 tiles regardless
     of source resolution — past ~1536px (2x2 tiles) extra pixels stop
     buying real detail. Proxy width bumped 960 -> 1536, the actual ceiling
     rather than an arbitrary number.
  8. GOAL-PASS SAMPLE FPS. 2.0 -> 5.0 fps for the goal pass only (general
     pass stays at 1.0) — catches fast net-ripple moments 2fps could miss.
  9. GOAL PASS MODEL SPLIT. Goal pass now tries `gemini-3.1-pro-preview`
     first, falling back through the same flash waterfall as the general
     pass. NOTE: Pro-tier models moved behind billing (~April 2026) — on a
     free-tier key this call fails immediately (not a quota error) and
     falls through automatically; the pipeline won't break, you just won't
     get Pro-tier accuracy without billing enabled.
 10. DETERMINISM. temperature=0 on every call — same match in, same events
     out, run to run. Doesn't narrow what footage the model can handle;
     temperature only affects response consistency, not generality.
 11. OVERLAP WIDENED 20s -> 30s. More margin so a goal sequence near a chunk
     boundary isn't split awkwardly across both proxies. Note this only
     affects DETECTION — the final highlight clip is cut from the original
     match.mp4 by generate_highlights.py using its own pre/post-roll padding
     around the refined exact timestamp, so this never truncates output.
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
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ── File paths ────────────────────────────────────────────────────────────────
INPUT_VIDEO   = os.path.join(".", "match.mp4")
OUTPUT_JSON   = os.path.join(".", "timestamps.json")
CHUNK_WORKDIR = os.path.join(".", "_analyze_chunks")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BUNDLED_FONT = os.path.join(SCRIPT_DIR, "assets", "fonts", "DejaVuSans-Bold.ttf")

# ── Model defaults ────────────────────────────────────────────────────────────
# Waterfall: try the newest/best model first; on a hard failure for that model
# (not just a transient quota error — those retry in place first, see
# _is_transient / _sleep_for_retry) fall through to the next one down.
#
# Status check as of the last time this list was verified against Google's
# docs (2026-07-29) — model availability in this generation shifts fast, so
# re-check https://ai.google.dev/gemini-api/docs/models before assuming this
# is still current:
#   gemini-3.6-flash     - current GA flash-tier model, released 2026-07-21.
#   gemini-3.5-flash     - previous flash-tier GA model, still callable.
#   gemini-3.1-flash-lite- there is no plain "gemini-3.1-flash"; flash-lite is
#                          the closest match in that generation.
#   gemini-2.5-flash     - officially deprecated (shutdown 2026-10-16), and
#                          already returning intermittent 404s in some regions
#                          well before that date — treat as unreliable, not gone.
#   gemini-2.0-flash     - ALREADY SHUT DOWN (2026-06-01). Kept only as an
#                          inert last resort; will fail fast (404) every time.
#   gemini-1.5-flash     - ALREADY SHUT DOWN. Same as above — dead weight but
#                          harmless, since a hard failure just falls through
#                          to the next candidate instead of retrying forever.
DEFAULT_GEMMA_MODEL              = "gemma-4-27b-it"
DEFAULT_MODEL                    = "gemini-3.6-flash"
DEFAULT_GEMINI_FALLBACK_MODELS   = (
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)

# Goal pass gets a Pro-tier model first — it's the one call where a wrong
# answer costs you a missed highlight, so it's worth the extra cost/latency.
# NOTE: as of ~April 2026 Google moved Pro models behind billing, off the
# free tier entirely. Without billing enabled this call fails immediately
# (not a quota error) and falls through to the flash chain below — the
# pipeline still won't break, you just won't get the Pro-tier accuracy
# unless billing is on. Falls back through the SAME flash chain as the
# general pass (goal accuracy > general-event accuracy, never the reverse).
DEFAULT_GOAL_MODEL = "gemini-3.1-pro-preview"

# ── Chunking settings ──────────────────────────────────────────────────────────
# Shorter chunks = better timestamp grounding, more API calls (cost/time).
# 8 min with 20s overlap is a reasonable default for a ~90-100 min match.
DEFAULT_CHUNK_SECONDS         = 480   # 8 minutes
DEFAULT_CHUNK_OVERLAP_SECONDS = 30    # was 20 — more margin so a goal near a
                                       # chunk boundary isn't split awkwardly
                                       # across both proxies

# ── Chunk proxy encoding (shared by both goal + general passes) ───────────────
# 640px/CRF23 was too lossy to reliably show net movement / ball-across-line
# on far-side action — bumped resolution up and CRF down. This costs more
# upload bandwidth and slightly more inference time, but the proxy is what
# the model actually sees, so it directly gates detection accuracy.
# Gemini tiles frames into 768x768 tiles regardless of source resolution —
# past 2 tiles wide (~1536px) extra pixels stop buying real detail for
# broadcast-style footage. 1536 is the actual ceiling, not an arbitrary bump.
DEFAULT_CHUNK_PROXY_WIDTH = 1536  # px — at the real Gemini tiling ceiling
DEFAULT_CHUNK_PROXY_FPS   = 15    # encoded fps (smooth burned-in clock text)
DEFAULT_CHUNK_PROXY_CRF   = 18    # lower = higher quality (23 was too lossy)

# ── Gemini internal sampling rate (THIS is what actually controls what the
#    model sees — independent of the encoded proxy fps above). ────────────────
DEFAULT_GOAL_SAMPLE_FPS    = 5.0   # was 2.0 — catches fast net-ripple moments
DEFAULT_GENERAL_SAMPLE_FPS = 1.0   # Gemini's own default, set explicitly anyway

# Deterministic output: same match in -> same events out, run to run.
# (Does not narrow what kind of football footage the model can handle —
# temperature only affects response consistency, not generality.)
DEFAULT_TEMPERATURE = 0.0

# Per-part media resolution for the goal pass only (Gemini 3 models only —
# gated by model name in _video_part; older fallback models silently skip it).
GOAL_MEDIA_RESOLUTION = types.MediaResolution.MEDIA_RESOLUTION_HIGH

# ── De-duplication across overlapping chunk boundaries ─────────────────────────
GOAL_MERGE_WINDOW_SECONDS    = 20   # goals within this gap = same event
GENERIC_MERGE_WINDOW_SECONDS = 8    # other events within this gap = same event

# ── Retry / polling ───────────────────────────────────────────────────────────
POLL_SECONDS               = 5
PROCESSING_TIMEOUT_SECONDS = 20 * 60
MAX_PROMPT_RETRIES         = 3
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


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        raise AnalysisError(f"{name} must be a number.")
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


def get_goal_model_candidates() -> list[str]:
    """Goal pass: try the Pro model first, then fall back through the same
    flash waterfall the general pass uses. Reuses get_model_candidates()
    instead of maintaining a second parallel list."""
    goal_model = os.getenv("GEMINI_GOAL_MODEL", DEFAULT_GOAL_MODEL).strip() or DEFAULT_GOAL_MODEL
    rest = [m for m in get_model_candidates() if m != goal_model]
    return [goal_model] + rest


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


# ══════════════════════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkSpec:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def compute_chunks(
    duration: float, chunk_seconds: float, overlap_seconds: float
) -> list[ChunkSpec]:
    """
    Split [0, duration] into overlapping, roughly EQUAL-length chunks.
    Overlap lets the same real event get seen by two chunks near a boundary;
    dedupe_events() collapses that back into a single event afterwards, so
    nothing is lost or doubled.

    Why equal-length instead of "fixed size + whatever's left over":
    The previous version always cut fixed `chunk_seconds`-long pieces and
    left the remainder as its own trailing chunk, only folding that
    remainder into the previous chunk if it was under 25% of chunk_seconds.
    That meant, e.g., a 10-minute match (600s) at the 480s default produced
    an 8-minute chunk + a lopsided 2.3-minute chunk — not wrong, but an
    unnecessary, unbalanced split for a video that's barely longer than one
    chunk to begin with. Solving for an even chunk length given the desired
    overlap removes that lopsidedness entirely, for any match length.
    """
    if duration <= chunk_seconds:
        return [ChunkSpec(0, 0.0, duration)]

    step_target = chunk_seconds - overlap_seconds
    if step_target <= 0:
        raise AnalysisError("Chunk overlap must be smaller than chunk duration.")

    # How many chunks of ~chunk_seconds (net of overlap) does this need?
    n = max(1, -(-(duration - overlap_seconds) // step_target))  # ceil division
    n = int(n)

    if n == 1:
        return [ChunkSpec(0, 0.0, duration)]

    # Solve for the equal chunk length L such that n chunks of length L,
    # each stepping forward by (L - overlap_seconds), exactly span `duration`:
    #   (n - 1) * (L - overlap_seconds) + L = duration
    L = (duration + (n - 1) * overlap_seconds) / n
    step = L - overlap_seconds

    chunks: list[ChunkSpec] = []
    start = 0.0
    for index in range(n):
        end = duration if index == n - 1 else start + L
        chunks.append(ChunkSpec(index, start, min(end, duration)))
        start += step

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Burned-in timecode overlay
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_font() -> str | None:
    if os.path.isfile(BUNDLED_FONT):
        return BUNDLED_FONT
    logging.warning(
        "Bundled font not found at %s — burned-in timecode will be disabled.",
        BUNDLED_FONT,
    )
    return None


def _escape_ffmpeg_path(path: str) -> str:
    """Escape a filesystem path for safe use inside an ffmpeg filtergraph
    option value (colons separate filter options, so 'C:\\...' on Windows
    must become 'C\\:/...')."""
    p = path.replace("\\", "/")
    p = p.replace(":", "\\:")
    return p


def _drawtext_filter(font_path: str) -> str:
    escaped = _escape_ffmpeg_path(os.path.abspath(font_path))
    return (
        f"drawtext=fontfile='{escaped}':"
        "text='%{pts\\:hms}':"
        "x=24:y=24:fontsize=40:fontcolor=white:"
        "box=1:boxcolor=black@0.55:boxborderw=12"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Chunk proxy creation
# ══════════════════════════════════════════════════════════════════════════════

def create_chunk_proxy(
    input_path: str,
    chunk: ChunkSpec,
    out_path: str,
    font_path: str | None,
) -> None:
    """
    Cut+re-encode a single chunk of the match, starting its own timeline at 0.
    -ss placed BEFORE -i is fast-seek, but ffmpeg performs accurate
    (frame-exact) seeking automatically whenever the output is re-encoded, so
    the chunk boundaries stay precise.
    """
    ensure_ffmpeg()

    width = _get_int_env("ANALYZE_CHUNK_PROXY_WIDTH", DEFAULT_CHUNK_PROXY_WIDTH)
    fps   = _get_int_env("ANALYZE_CHUNK_PROXY_FPS",   DEFAULT_CHUNK_PROXY_FPS)
    crf   = _get_int_env("ANALYZE_CHUNK_PROXY_CRF",   DEFAULT_CHUNK_PROXY_CRF)
    burn  = _get_bool_env("ANALYZE_BURN_TIMECODE", True)

    vf = f"scale={width}:-2,fps={fps}"
    if burn and font_path:
        vf += "," + _drawtext_filter(font_path)

    if os.path.isfile(out_path):
        os.remove(out_path)

    keep_audio = _get_bool_env("ANALYZE_KEEP_AUDIO", True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{chunk.start:.3f}",
        "-i", input_path,
        "-t", f"{chunk.duration:.3f}",
        "-vf", vf,
    ]
    if keep_audio:
        # KEEP AUDIO. Gemini processes audio and video as separate streams —
        # stripping audio (previously "-an") throws away the crowd-roar
        # spike and commentator reaction, which are often the single
        # clearest signal that a goal just happened. Downmix to mono/low
        # bitrate since we only need it as a corroborating cue, not for
        # playback quality.
        cmd += ["-c:a", "aac", "-ac", "1", "-b:a", "64k"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-movflags", "+faststart",
        out_path,
    ]

    run_command(cmd, f"Create chunk {chunk.index} proxy")

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise AnalysisError(f"Chunk {chunk.index} proxy was not created or is empty.")


# ══════════════════════════════════════════════════════════════════════════════
# Prompts (chunk-relative)
# ══════════════════════════════════════════════════════════════════════════════

def build_goal_detection_prompt(chunk_duration: float, attempt: int = 0) -> str:
    retry_note = ""
    if attempt > 0:
        retry_note = (
            f"\n\nRETRY {attempt}: Previous response was rejected. "
            "Re-watch the ENTIRE clip from 0:00 to the end and re-check the "
            "burned-in clock before answering."
        )

    return f"""You are a specialist football goal-detection system.
This is a SHORT CLIP cut from a longer match. It is {chunk_duration:.1f} seconds
long. Watch it from the very first frame to the last.

A running on-screen clock is burned into the TOP-LEFT corner of the video,
format HH:MM:SS, showing the exact elapsed time WITHIN THIS CLIP (it starts
at 00:00:00 at the first frame). Read the clock directly rather than
estimating time from pacing or context -- it is ground truth.

YOUR ONLY TASK: Find every goal scored in THIS CLIP and report its precise
timestamp in seconds from the START OF THIS CLIP (0 to {chunk_duration:.0f}).
It is completely fine to return an empty "goals" list if no goal occurs here.

════════════════════════════════════════════════════════════════
HOW TO IDENTIFY A GOAL (use BOTH audio and visual cues)
════════════════════════════════════════════════════════════════
This clip has audio. Listen to it as carefully as you watch the video —
a sudden crowd roar or the commentator's excited reaction (e.g. shouting
the scorer's name, "GOAL", a pitch/volume spike) is often the clearest,
fastest signal that a goal just happened, and should raise your confidence
even if the exact instant the ball crosses the line is briefly obscured by
a camera cut or replay.

VISUAL cues:
1. NET MOVEMENT — the net bulges, shakes, or ripples from a ball strike.
   This is the PRIMARY visual cue.
2. BALL POSITION — the ball is clearly inside or passing through the goal.
3. SCOREBOARD / SCORE GRAPHIC — the on-screen score increments by 1.
4. REFEREE GESTURE — the referee points to the centre circle.
5. PLAYER CELEBRATION — scorer and team-mates immediately celebrate.
6. CROWD REACTION (visual) — sudden crowd eruption synchronised with net movement.

AUDIO cues:
7. CROWD ROAR — a sudden, sustained volume spike in crowd noise.
8. COMMENTATOR REACTION — excited tone, raised pitch/volume, or the
   scorer's name being called out immediately after a shot.

A goal is confirmed when visual and audio cues agree. If audio strongly
suggests a goal but the ball-crossing-the-line moment itself is not
clearly visible (e.g. blocked by players, quick cut to celebration),
still report it — set the timestamp to the moment the net moves or the
shot is struck (whichever is visible) and lower confidence accordingly
rather than omitting the goal entirely.

════════════════════════════════════════════════════════════════
TIMESTAMP ACCURACY — CRITICAL
════════════════════════════════════════════════════════════════
• rough_timestamp = the value read off the burned-in clock at the moment the
  ball is FULLY PAST the goal line (net first moves / ball inside net).
• Do NOT timestamp the shot, the run-up, or the celebration start.
• Cross-check your answer against the on-screen clock -- do not guess.

════════════════════════════════════════════════════════════════
OUTPUT FORMAT — return ONLY valid JSON, no markdown, no commentary
════════════════════════════════════════════════════════════════
{{
  "goals": [
    {{
      "rough_timestamp": 123.5,
      "confidence": 0.95,
      "description": "Left-footed shot, net bulges bottom-right, score graphic updates to 1-0"
    }}
  ]
}}{retry_note}"""


def build_prompt(chunk_duration: float, attempt: int = 0) -> str:
    """General highlights prompt, chunk-relative."""
    expected_hint = max(1, round(chunk_duration / 60 * 1.2))
    retry_note = ""
    if attempt > 0:
        retry_note = (
            f"\n\nRETRY {attempt}: Previous response was invalid or empty JSON. "
            "Re-check the burned-in clock and re-submit using ONLY the event "
            "types listed below."
        )

    valid_list = " | ".join(sorted(VALID_TYPES))

    return f"""You are an expert football video analyst. Watch this SHORT CLIP
({chunk_duration:.1f} seconds, cut from a longer match) from its first frame
to its last and identify every highlight-worthy moment for a TV-style reel.

A running on-screen clock is burned into the TOP-LEFT corner, format
HH:MM:SS, showing elapsed time WITHIN THIS CLIP (starts at 00:00:00 at the
first frame). Read timestamps directly off this clock.

════════════════════════════════════════════════════════════════
COVERAGE
════════════════════════════════════════════════════════════════
• Scan the entire clip, start to finish.
• Roughly {expected_hint} or more events is typical for a clip this length if
  the play is active — but report what is ACTUALLY there. A quiet clip
  (e.g. midfield build-up with no shots) can legitimately have very few
  events. Never invent events to hit a quota.
• IMPORTANT: Goals are supplied by a dedicated goal-detection pass over this
  same clip. You do NOT need to find goals — focus on all other event types.
  If you do see an obvious goal, still report it; the pipeline will
  de-duplicate it against the goal pass automatically.

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
• rough_timestamp = seconds from the start of THIS CLIP, read off the
  burned-in clock — not your own sense of pacing.
• Timestamp the START of the action (shot struck, not net bulging).
• confidence: 0.85-1.00 clearly visible · 0.60-0.84 probable
• Do NOT invent events. Only report what is visually present.
• Do NOT list the same play twice under different labels.
• If goal + celebration are one sequence, report the celebration only.

Return ONLY valid JSON — no markdown fences, no commentary. An empty
"events" list is a valid answer for a quiet clip:
{{
  "events": [
    {{
      "type": "one of the allowed types above",
      "rough_timestamp": 54.2,
      "confidence": 0.94,
      "description": "short description"
    }}
  ]
}}{retry_note}"""


# ══════════════════════════════════════════════════════════════════════════════
# Response parsing & validation (chunk-relative bounds)
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
        logging.info("Mapped unknown type '%s' -> '%s'", raw, mapped)
        return mapped
    logging.warning("Dropping unknown event type '%s'", raw)
    return None


def parse_goals_response(response_text: str, chunk_duration: float) -> list[dict[str, Any]]:
    """Parse the goals-only pass for one chunk. Empty list is a valid result."""
    try:
        payload = json.loads(_extract_json(response_text))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid JSON in goal response: {exc}") from exc

    if not isinstance(payload, dict):
        raise AnalysisError("Goal JSON root is not an object.")

    raw_goals = payload.get("goals")
    if not isinstance(raw_goals, list):
        raw_goals = payload.get("events", [])
        if not isinstance(raw_goals, list):
            raise AnalysisError("'goals' array is missing.")

    goals: list[dict[str, Any]] = []
    for i, item in enumerate(raw_goals, start=1):
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("type", "goal"))
        canonical = _resolve_type(raw_type)
        if canonical != "goal":
            continue
        try:
            ts = float(item["rough_timestamp"])
        except (KeyError, TypeError, ValueError):
            logging.warning("Goal #%d has invalid rough_timestamp — skipping.", i)
            continue
        if ts < 0:
            ts = 0.0
        if ts > chunk_duration + 15:
            logging.warning(
                "Goal #%d timestamp %.1f s exceeds chunk duration %.1f s — skipping.",
                i, ts, chunk_duration,
            )
            continue
        ts = min(ts, chunk_duration)
        goals.append(
            {
                "type":            "goal",
                "rough_timestamp": round(ts, 3),
                "confidence":      round(_normalize_confidence(item.get("confidence")), 3),
                "description":     str(item.get("description", "")).strip(),
                "_source":         "goal_pass",
            }
        )

    goals.sort(key=lambda e: e["rough_timestamp"])
    return goals


def parse_and_validate(response_text: str, chunk_duration: float) -> dict[str, Any]:
    """Parse JSON from the general-highlights pass for one chunk."""
    try:
        payload = json.loads(_extract_json(response_text))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise AnalysisError("JSON root is not an object.")

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise AnalysisError("'events' array is missing.")

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
        ts = min(ts, chunk_duration + 15)

        events.append(
            {
                "type":            canonical,
                "rough_timestamp": round(ts, 3),
                "confidence":      round(_normalize_confidence(item.get("confidence")), 3),
                "description":     str(item.get("description", "")).strip(),
            }
        )

    events.sort(key=lambda e: e["rough_timestamp"])
    # An empty list is a legitimate answer for a quiet chunk — not an error.
    return {"events": events}


# ══════════════════════════════════════════════════════════════════════════════
# Goal merge (within a single chunk) + cross-chunk de-duplication
# ══════════════════════════════════════════════════════════════════════════════

def merge_goals_into_events(
    goal_pass_goals: list[dict[str, Any]],
    general_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Goal-pass goals are authoritative for timestamp accuracy within a chunk."""
    window = GOAL_MERGE_WINDOW_SECONDS
    goal_timestamps = [g["rough_timestamp"] for g in goal_pass_goals]

    def _near_goal_pass(event: dict[str, Any]) -> bool:
        if event["type"] != "goal":
            return False
        return any(abs(event["rough_timestamp"] - gt) <= window for gt in goal_timestamps)

    filtered = [e for e in general_events if not _near_goal_pass(e)]
    merged = filtered + goal_pass_goals
    merged.sort(key=lambda e: e["rough_timestamp"])
    for event in merged:
        event.pop("_source", None)
    return merged


def dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Collapse duplicate events that appear in the overlap region between two
    adjacent chunks. Events are already in absolute match-time seconds here.

    Same-chunk events are NEVER merged, even if close in time: the model
    already reported them as two separate things in one call, so collapsing
    them on timestamp-proximity alone risks silently dropping a real event
    (e.g. two genuinely distinct goals scored within the merge window of
    each other). Only cross-chunk pairs — the actual overlap-region
    duplicates this function exists to clean up — are candidates for merge.
    """
    events = sorted(events, key=lambda e: e["rough_timestamp"])
    deduped: list[dict[str, Any]] = []

    for event in events:
        window = GOAL_MERGE_WINDOW_SECONDS if event["type"] == "goal" else GENERIC_MERGE_WINDOW_SECONDS
        merged = False
        for existing in reversed(deduped):
            if event["rough_timestamp"] - existing["rough_timestamp"] > max(
                GOAL_MERGE_WINDOW_SECONDS, GENERIC_MERGE_WINDOW_SECONDS
            ):
                break
            if existing["type"] != event["type"]:
                continue
            if existing["_chunk"] == event["_chunk"]:
                continue
            if abs(event["rough_timestamp"] - existing["rough_timestamp"]) <= window:
                if event["confidence"] > existing["confidence"]:
                    existing.update(event)
                merged = True
                break
        if not merged:
            deduped.append(dict(event))

    deduped.sort(key=lambda e: e["rough_timestamp"])
    for event in deduped:
        event.pop("_chunk", None)
    return deduped


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
        time.sleep(POLL_SECONDS)


def _upload_with_retry(client: genai.Client, video_path: str, label: str) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
        try:
            uploaded = client.files.upload(file=video_path)
            return _wait_for_file(client, uploaded)
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == MAX_UPLOAD_RETRIES:
                raise
            _sleep_for_retry(attempt, exc, f"{label} upload")

    raise AnalysisError(f"{label} upload failed after {MAX_UPLOAD_RETRIES} attempts: {last_exc}")


def _video_part(uploaded_file: Any, fps: float | None, media_resolution: Any = None) -> Any:
    """
    Build the request Part explicitly so we can set video_metadata.fps.
    Gemini's File API samples at a fixed 1 FPS by default REGARDLESS of the
    encoded proxy's frame rate — this override is the only way to actually
    change what the model samples.

    media_resolution is a Gemini-3-only per-part override (see
    GOAL_MEDIA_RESOLUTION); pass None for older/fallback models.
    """
    file_uri  = getattr(uploaded_file, "uri", None)
    mime_type = getattr(uploaded_file, "mime_type", None) or "video/mp4"
    if not file_uri:
        # Fallback for SDK versions where passing the File object directly works.
        return uploaded_file
    return types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=fps) if fps else None,
        media_resolution=media_resolution,
    )


def _call_model_with_retry(
    client: genai.Client,
    model: str,
    uploaded_file: Any,
    fps: float,
    prompt: str,
    label: str,
    media_resolution: Any = None,
) -> Any:
    # Gemini-3-only feature — only send it to models in that family.
    part_resolution = media_resolution if model.startswith("gemini-3") else None
    config = types.GenerateContentConfig(temperature=DEFAULT_TEMPERATURE)

    last_exc: Exception | None = None
    for attempt in range(1, API_RETRIES_PER_CALL + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=[_video_part(uploaded_file, fps, part_resolution), prompt],
                config=config,
            )
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == API_RETRIES_PER_CALL:
                raise
            _sleep_for_retry(attempt, exc, f"{label}/{model}")

    raise AnalysisError(f"API call failed after {API_RETRIES_PER_CALL} retries: {last_exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-chunk analysis
# ══════════════════════════════════════════════════════════════════════════════

def _run_goal_pass_for_chunk(
    client: genai.Client,
    models: list[str],
    uploaded_file: Any,
    chunk: ChunkSpec,
    label: str,
) -> list[dict[str, Any]]:
    goal_fps = _get_float_env("ANALYZE_GOAL_SAMPLE_FPS", DEFAULT_GOAL_SAMPLE_FPS)
    last_error: Exception | None = None
    for model in models:
        for attempt in range(MAX_PROMPT_RETRIES):
            try:
                response = _call_model_with_retry(
                    client, model, uploaded_file, goal_fps,
                    build_goal_detection_prompt(chunk.duration, attempt),
                    f"{label}/chunk{chunk.index}/Goal",
                    media_resolution=GOAL_MEDIA_RESOLUTION,
                )
                return parse_goals_response(response.text or "", chunk.duration)
            except AnalysisError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
                break
    logging.warning(
        "[%s/chunk%d/Goal] all attempts failed: %s — no goals for this chunk.",
        label, chunk.index, last_error,
    )
    return []


def _run_general_pass_for_chunk(
    client: genai.Client,
    models: list[str],
    uploaded_file: Any,
    chunk: ChunkSpec,
    label: str,
) -> list[dict[str, Any]]:
    general_fps = _get_float_env("ANALYZE_GENERAL_SAMPLE_FPS", DEFAULT_GENERAL_SAMPLE_FPS)
    last_error: Exception | None = None
    for model in models:
        for attempt in range(MAX_PROMPT_RETRIES):
            try:
                response = _call_model_with_retry(
                    client, model, uploaded_file, general_fps,
                    build_prompt(chunk.duration, attempt),
                    f"{label}/chunk{chunk.index}",
                )
                return parse_and_validate(response.text or "", chunk.duration)["events"]
            except AnalysisError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
                break
    logging.warning(
        "[%s/chunk%d] all attempts failed: %s — skipping general events for this chunk.",
        label, chunk.index, last_error,
    )
    return []


def _process_chunk(
    client: genai.Client,
    goal_models: list[str],
    general_models: list[str],
    input_video_path: str,
    chunk: ChunkSpec,
    work_dir: str,
    font_path: str | None,
    label: str,
) -> list[dict[str, Any]]:
    logging.info(
        "[%s] ── Chunk %d: %.0fs-%.0fs (%.0fs) ──",
        label, chunk.index, chunk.start, chunk.end, chunk.duration,
    )

    proxy_path = os.path.join(work_dir, f"chunk_{chunk.index:03d}.mp4")
    create_chunk_proxy(input_video_path, chunk, proxy_path, font_path)

    uploaded_file: Any = None
    try:
        uploaded_file = _upload_with_retry(client, proxy_path, f"{label}/chunk{chunk.index}")

        goal_events    = _run_goal_pass_for_chunk(client, goal_models, uploaded_file, chunk, label)
        general_events = _run_general_pass_for_chunk(client, general_models, uploaded_file, chunk, label)
        merged = merge_goals_into_events(goal_events, general_events)

        for event in merged:
            event["rough_timestamp"] = round(event["rough_timestamp"] + chunk.start, 3)
            event["_chunk"] = chunk.index

        logging.info(
            "[%s/chunk%d] %d event(s) (%d goal(s))",
            label, chunk.index, len(merged),
            sum(1 for e in merged if e["type"] == "goal"),
        )
        return merged
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as exc:
                logging.warning("[%s/chunk%d] Could not delete uploaded file: %s", label, chunk.index, exc)
        if os.path.isfile(proxy_path):
            try:
                os.remove(proxy_path)
            except Exception as exc:
                logging.warning("[%s/chunk%d] Could not delete chunk proxy: %s", label, chunk.index, exc)


def _run_genai_analysis(
    client: genai.Client,
    goal_models: list[str],
    general_models: list[str],
    input_video_path: str,
    video_duration: float,
    label: str,
) -> dict[str, Any]:
    chunk_seconds    = _get_int_env("ANALYZE_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS)
    overlap_seconds  = _get_int_env("ANALYZE_CHUNK_OVERLAP_SECONDS", DEFAULT_CHUNK_OVERLAP_SECONDS)
    chunks = compute_chunks(video_duration, chunk_seconds, overlap_seconds)
    logging.info(
        "[%s] Split %.0fs match into %d chunk(s) of ~%ds (overlap %ds).",
        label, video_duration, len(chunks), chunk_seconds, overlap_seconds,
    )

    font_path = _resolve_font()
    work_dir = os.path.join(CHUNK_WORKDIR, label.lower())
    os.makedirs(work_dir, exist_ok=True)

    all_events: list[dict[str, Any]] = []
    try:
        for chunk in chunks:
            chunk_events = _process_chunk(
                client, goal_models, general_models, input_video_path, chunk, work_dir, font_path, label
            )
            all_events.extend(chunk_events)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if not all_events:
        raise AnalysisError(f"[{label}] No events found in any chunk.")

    deduped = dedupe_events(all_events)
    logging.info(
        "[%s] %d raw events across all chunks -> %d after de-duplication.",
        label, len(all_events), len(deduped),
    )
    return {"events": deduped}


# ══════════════════════════════════════════════════════════════════════════════
# Provider: Gemini
# ══════════════════════════════════════════════════════════════════════════════

def try_gemini(input_video_path: str, video_duration: float) -> dict[str, Any] | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        logging.info("GEMINI_API_KEY not set — skipping Gemini.")
        return None
    try:
        client = genai.Client(api_key=key)
        return _run_genai_analysis(
            client, get_goal_model_candidates(), get_model_candidates(),
            input_video_path, video_duration, "Gemini",
        )
    except Exception as exc:
        logging.warning("Gemini failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Provider: Gemma
# ══════════════════════════════════════════════════════════════════════════════

def try_gemma(input_video_path: str, video_duration: float) -> dict[str, Any] | None:
    key = os.getenv("GEMMA_API_KEY", "").strip()
    if not key:
        logging.info("GEMMA_API_KEY not set — skipping Gemma.")
        return None
    try:
        client = genai.Client(api_key=key)
        gemma_models = get_gemma_model_candidates()
        return _run_genai_analysis(
            client, gemma_models, gemma_models,  # no Pro tier in Gemma — one list, both passes
            input_video_path, video_duration, "Gemma",
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
    ensure_ffmpeg()

    duration = probe_duration(video_path)
    logging.info("Input video: %.1f s (%.1f min)", duration, duration / 60)

    shutil.rmtree(CHUNK_WORKDIR, ignore_errors=True)
    os.makedirs(CHUNK_WORKDIR, exist_ok=True)

    try:
        parsed: dict[str, Any] | None = None

        parsed = try_gemini(video_path, duration)
        if parsed is None:
            parsed = try_gemma(video_path, duration)

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
        shutil.rmtree(CHUNK_WORKDIR, ignore_errors=True)


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
