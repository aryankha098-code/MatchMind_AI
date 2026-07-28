""" 
event_refiner.py
Stage 2: refine Gemini's rough football events to frame-accurate events.

This module is intentionally deterministic and Docker-friendly: it uses OpenCV,
NumPy, and FFmpeg only. It does not pretend Gemini timestamps are exact. Each
rough timestamp is searched frame-by-frame in a +-10 second window, scored with
visual motion, ball-like object motion, scoreboard-region change, and optional
audio reaction energy.

Goal-detection changes (v3 — tight anchor):
  • PREVIOUS APPROACH (v2) searched up to 13 s before rough_timestamp for a
    "shot-strike spike" preceding the audio/scoreboard reaction. In practice
    this let the chosen frame drift 2-5 s EARLIER than the real goal moment
    (e.g. goal at 4:50, effect applied at 4:39-4:48 — the true goal second was
    never inside the output clip at all). That approach is REMOVED.
  • NEW APPROACH: Gemini's rough_timestamp for goals is usually already close
    (within 1-3 s) to the real goal moment, since the model is instructed to
    timestamp "ball fully past the line" in the goal-detection prompt. So the
    refiner's job here is SUB-SECOND correction, not a wide backward search.
  • GOAL_SEARCH_TIGHT_RADIUS_SECONDS (default 3.0 s) defines a narrow window
    centered on rough_timestamp. Within that window we look for the strongest
    local motion spike (ball/net impact) using GOAL_SPIKE_ZSCORE.
  • GOAL_MAX_DRIFT_SECONDS (default 2.5 s) is a HARD CLAMP: whatever frame is
    chosen, if it is more than this many seconds away from rough_timestamp, we
    snap back to rough_timestamp itself. This guarantees the slow-mo/zoom
    replay window built downstream (8 s pre / 2 s post of effect_start_frame)
    always contains the real goal moment, even in the worst case.
  • If no clear spike is found in the tight window, falls back directly to
    rough_timestamp's own frame (no large backward jump).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv


INPUT_VIDEO = os.path.join(".", "match.mp4")
COARSE_JSON = os.path.join(".", "timestamps.json")
OUTPUT_JSON = os.path.join(".", "refined_events.json")

SEARCH_RADIUS_SECONDS = 10.0

# Goals: NARROW, symmetric search window centered on rough_timestamp.
# Gemini's goal timestamp is usually already close to correct (it is told to
# mark "ball fully past the line"), so we only do small local correction.
GOAL_SEARCH_TIGHT_RADIUS_SECONDS = 3.0   # seconds each side of rough_timestamp

# Hard clamp: chosen goal frame can NEVER be more than this far from
# rough_timestamp. Guarantees the goal moment is always inside the
# downstream 8s-pre/2s-post replay window built in generate_highlighs.py.
GOAL_MAX_DRIFT_SECONDS = 2.5

# Minimum z-score a frame-diff spike must have to be treated as a shot strike.
GOAL_SPIKE_ZSCORE = 1.2

# Window over which the ball position is sampled for the moving zoom track.
# Matches the 8s-pre / 2s-post replay window used in generate_highlighs.py so
# the zoom can follow the ball across the entire replay, not just the goal
# frame itself.
GOAL_REPLAY_PRE_SECONDS_FOR_TRACK = 8.0
GOAL_REPLAY_POST_SECONDS_FOR_TRACK = 2.0

GOAL_SLOWMO_OUTPUT_SECONDS = 9.0
GOAL_SLOWMO_SPEED = 0.5
GOAL_EFFECT_DURATION_SECONDS = GOAL_SLOWMO_OUTPUT_SECONDS * GOAL_SLOWMO_SPEED
ANALYSIS_WIDTH = 640
MIN_CONFIDENCE = 0.35
COARSE_DUPLICATE_SECONDS = 9.0
DEFAULT_HIGH_PRECISION_TYPES = {"goal", "penalty", "save", "near_miss", "shot_on_target"}
PROTECTED_EVENT_TYPES = {"goal", "penalty", "red_card"}

VALID_TYPES = {
    "goal",
    "assist",
    "save",
    "tackle",
    "foul",
    "near_miss",
    "celebration",
    "dribble",
    "dangerous_attack",
    "counter_attack",
    "crowd_reaction",
    "penalty",
    "red_card",
    "yellow_card",
    "free_kick",
    "shot_on_target",
}

EVENT_PRIORITY = {
    "goal": 100,
    "penalty": 95,
    "red_card": 92,
    "shot_on_target": 91,
    "near_miss": 90,
    "save": 88,
    "free_kick": 82,
    "yellow_card": 73,
    "assist": 84,
    "dangerous_attack": 78,
    "counter_attack": 76,
    "celebration": 70,
    "dribble": 68,
    "tackle": 58,
    "foul": 45,
    "crowd_reaction": 40,
}


class RefinementError(Exception):
    """Raised when frame refinement fails."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    has_audio: bool


@dataclass
class FrameSignal:
    frame_number: int
    timestamp: float
    motion: float
    ball_motion: float
    scoreboard_change: float
    audio: float
    subject_x: float
    subject_y: float
    cluster_score: float = 0.0
    cluster_x: float = 0.5
    cluster_y: float = 0.5


def configure_logging() -> None:
    try:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    except Exception as exc:
        print(f"Logging setup failed: {exc}")


def ensure_inputs() -> None:
    try:
        if not os.path.isfile(INPUT_VIDEO):
            raise RefinementError(f"Missing input video: {INPUT_VIDEO}")
        if not os.path.isfile(COARSE_JSON):
            raise RefinementError(f"Missing coarse event JSON: {COARSE_JSON}")
        if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
            raise RefinementError("FFmpeg and FFprobe must be available on PATH.")
    except RefinementError:
        raise
    except Exception as exc:
        raise RefinementError(f"Could not validate inputs: {exc}") from exc


def run_command(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "No stderr output."
        raise RefinementError(f"{label} failed: {stderr}") from exc
    except Exception as exc:
        raise RefinementError(f"{label} could not start: {exc}") from exc


def probe_video(video_path: str) -> VideoInfo:
    try:
        result = run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                video_path,
            ],
            "Video probe",
        )
        metadata = json.loads(result.stdout)
        streams = metadata.get("streams", [])
        video_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            raise RefinementError("No video stream found.")

        fps = parse_ratio(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
        if fps <= 0:
            raise RefinementError("Could not determine video FPS.")

        duration = float(metadata.get("format", {}).get("duration", 0))
        frame_count = int(float(video_stream.get("nb_frames") or duration * fps))
        width = int(video_stream.get("width"))
        height = int(video_stream.get("height"))
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        return VideoInfo(fps, frame_count, duration, width, height, has_audio)
    except RefinementError:
        raise
    except Exception as exc:
        raise RefinementError(f"Could not probe video metadata: {exc}") from exc


def parse_ratio(value: Any) -> float:
    try:
        text = str(value)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        return float(text)
    except Exception:
        return 0.0


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    return int(round(timestamp * fps))


def frame_to_timestamp(frame_number: int, fps: float) -> float:
    return frame_number / fps


def get_high_precision_types() -> set[str]:
    try:
        raw_value = os.getenv("REFINE_HIGH_PRECISION_TYPES", "").strip()
        if not raw_value:
            return set(DEFAULT_HIGH_PRECISION_TYPES)

        selected = {
            item.strip().lower()
            for item in raw_value.split(",")
            if item.strip().lower()
        }
        invalid = selected - VALID_TYPES
        if invalid:
            raise RefinementError(
                "REFINE_HIGH_PRECISION_TYPES contains unsupported types: "
                + ", ".join(sorted(invalid))
            )
        return selected
    except RefinementError:
        raise
    except Exception as exc:
        raise RefinementError(f"Could not read REFINE_HIGH_PRECISION_TYPES: {exc}") from exc


def read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8-sig") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            raise RefinementError(f"{path} must contain a JSON object.")
        return payload
    except json.JSONDecodeError as exc:
        raise RefinementError(f"Invalid JSON in {path}: {exc}") from exc
    except RefinementError:
        raise
    except Exception as exc:
        raise RefinementError(f"Could not read {path}: {exc}") from exc


def normalize_coarse_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        if isinstance(payload.get("events"), list):
            raw_events = payload["events"]
            events = []
            for index, item in enumerate(raw_events, start=1):
                event_type = str(item.get("type", "")).lower()
                if event_type not in VALID_TYPES:
                    continue
                rough_timestamp = float(
                    item.get("rough_timestamp", item.get("exact_timestamp", item.get("start_time", 0)))
                )
                events.append(
                    {
                        "type": event_type,
                        "rough_timestamp": rough_timestamp,
                        "confidence": float(item.get("confidence", 0.5)),
                        "description": str(item.get("description", "")),
                        "source_index": index,
                    }
                )
            return events

        if isinstance(payload.get("moments"), list):
            return [
                {
                    "type": str(item.get("type", "")).lower(),
                    "rough_timestamp": float(item.get("peak_time", item.get("start_time", 0))),
                    "confidence": float(item.get("confidence", item.get("excitement_score", 5))) / 10,
                    "description": str(item.get("description", "")),
                    "source_index": index,
                }
                for index, item in enumerate(payload["moments"], start=1)
                if str(item.get("type", "")).lower() in VALID_TYPES
            ]

        if isinstance(payload.get("highlights"), list):
            return [
                {
                    "type": str(item.get("event", "")).lower(),
                    "rough_timestamp": timestamp_to_seconds(str(item.get("start", 0))),
                    "confidence": 0.5,
                    "description": str(item.get("description", "")),
                    "source_index": index,
                }
                for index, item in enumerate(payload["highlights"], start=1)
                if str(item.get("event", "")).lower() in VALID_TYPES
            ]

        raise RefinementError("No supported event format found in timestamps.json.")
    except RefinementError:
        raise
    except Exception as exc:
        raise RefinementError(f"Could not normalize coarse events: {exc}") from exc


def event_priority(event_type: str) -> int:
    return EVENT_PRIORITY.get(event_type, 50)


def merge_descriptions(events: list[dict[str, Any]]) -> str:
    descriptions: list[str] = []
    for event in events:
        text = str(event.get("description", "")).strip()
        if text and text not in descriptions:
            descriptions.append(text)
    return " | ".join(descriptions[:3])


def coarse_event_score(event: dict[str, Any]) -> float:
    confidence = max(0.0, min(1.0, float(event.get("confidence", 0.5))))
    return event_priority(str(event.get("type", ""))) + confidence


def is_protected_event(event: dict[str, Any]) -> bool:
    return str(event.get("type", "")).lower() in PROTECTED_EVENT_TYPES


def choose_cluster_winner(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    protected = [event for event in cluster if is_protected_event(event)]
    if protected:
        return max(protected, key=coarse_event_score)
    return max(cluster, key=coarse_event_score)


def compress_coarse_duplicates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        if not events:
            return []

        ordered = sorted(events, key=lambda item: item["rough_timestamp"])
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = [ordered[0]]

        for event in ordered[1:]:
            previous = current[-1]
            near_in_time = (
                float(event["rough_timestamp"]) - float(previous["rough_timestamp"])
                <= COARSE_DUPLICATE_SECONDS
            )
            same_type_nearby = (
                event["type"] == previous["type"]
                and float(event["rough_timestamp"]) - float(previous["rough_timestamp"])
                <= COARSE_DUPLICATE_SECONDS * 1.6
            )
            protected_cluster = any(is_protected_event(item) for item in current) and (
                float(event["rough_timestamp"]) - float(current[0]["rough_timestamp"])
                <= COARSE_DUPLICATE_SECONDS * 2
            )

            if near_in_time or same_type_nearby or protected_cluster:
                current.append(event)
            else:
                clusters.append(current)
                current = [event]

        clusters.append(current)

        compressed: list[dict[str, Any]] = []
        duplicate_count = 0
        for cluster in clusters:
            winner = choose_cluster_winner(cluster)
            merged = dict(winner)
            merged["description"] = merge_descriptions(cluster) or winner.get("description", "")
            merged["merged_coarse_events"] = [
                {
                    "type": item["type"],
                    "rough_timestamp": round(float(item["rough_timestamp"]), 3),
                    "confidence": round(float(item.get("confidence", 0.5)), 3),
                }
                for item in cluster
            ]
            compressed.append(merged)
            duplicate_count += max(0, len(cluster) - 1)

        if duplicate_count:
            logging.info(
                "Removed %d obvious duplicate coarse detections before refinement.",
                duplicate_count,
            )
        return compressed
    except Exception as exc:
        raise RefinementError(f"Could not compress coarse duplicates: {exc}") from exc


def timestamp_to_seconds(value: str) -> float:
    try:
        if ":" not in value:
            return float(value)
        parts = [float(part) for part in value.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        raise ValueError(value)
    except Exception as exc:
        raise RefinementError(f"Invalid timestamp: {value}") from exc


def normalize(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    minimum = float(array.min())
    maximum = float(array.max())
    if math.isclose(maximum, minimum):
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def smooth(values: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    try:
        if values.size < kernel_size:
            return values
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        return np.convolve(values, kernel, mode="same")
    except Exception:
        return values


def extract_audio_rms(video_path: str, start_time: float, duration: float, fps: float) -> np.ndarray:
    try:
        if duration <= 0:
            return np.zeros(0, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        try:
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_time:.6f}",
                    "-t",
                    f"{duration:.6f}",
                    "-i",
                    video_path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    temp_audio_path,
                ],
                "Audio extraction",
            )
            with wave.open(temp_audio_path, "rb") as wav_file:
                samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
                sample_rate = wav_file.getframerate()
            if samples.size == 0:
                return np.zeros(int(round(duration * fps)), dtype=np.float32)
            samples_per_frame = max(1, int(round(sample_rate / fps)))
            frame_count = int(round(duration * fps))
            rms = np.zeros(frame_count, dtype=np.float32)
            for index in range(frame_count):
                chunk = samples[index * samples_per_frame : (index + 1) * samples_per_frame]
                if chunk.size:
                    rms[index] = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            return normalize(rms).astype(np.float32)
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    except Exception:
        return np.zeros(int(round(duration * fps)), dtype=np.float32)


def resize_for_analysis(frame: np.ndarray) -> np.ndarray:
    try:
        height, width = frame.shape[:2]
        if width <= ANALYSIS_WIDTH:
            return frame
        scale = ANALYSIS_WIDTH / width
        return cv2.resize(frame, (ANALYSIS_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
    except Exception as exc:
        raise RefinementError(f"Could not resize frame: {exc}") from exc


def ball_candidate(frame: np.ndarray) -> tuple[float, float, float]:
    try:
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 80, 255]))
        yellow = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([40, 255, 255]))
        mask = cv2.bitwise_or(white, yellow)
        mask[: int(height * 0.12), :] = 0
        mask[int(height * 0.92) :, :] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = 0.0
        best_x = 0.5
        best_y = 0.5
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8 or area > 450:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            aspect = min(w, h) / max(w, h)
            score = float(circularity * aspect * min(1.0, area / 80))
            if score > best_score:
                best_score = score
                best_x = (x + w / 2) / width
                best_y = (y + h / 2) / height
        return best_score, best_x, best_y
    except Exception:
        return 0.0, 0.5, 0.5


def action_cluster_center(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
) -> tuple[float, float, float]:
    """
    Find where the on-pitch action is concentrated in this frame, using motion
    density rather than color-blob matching for the ball.

    Broadcast-distance youth/amateur football footage makes a small ball very
    hard to track reliably by color (white shirts, socks, field lines, and
    scoreboard graphics all produce false positives). Player movement is a
    much more robust signal: wherever players are moving and clustered
    together is where the play — and therefore the goal action — actually is.

    Returns (confidence, x, y) in normalized [0,1] coordinates, restricted to
    the pitch region (excludes scoreboard strip and crowd/background).
    """
    try:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if previous_gray is None:
            return 0.0, 0.5, 0.5

        diff = cv2.absdiff(gray, previous_gray)
        _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)

        # Restrict to the pitch area: exclude top scoreboard strip and the
        # very bottom (near-camera grass with little useful signal).
        pitch_top = int(height * 0.18)
        pitch_bottom = int(height * 0.95)
        motion_mask[:pitch_top, :] = 0
        motion_mask[pitch_bottom:, :] = 0

        # Dilate to merge nearby player blobs into clusters.
        kernel = np.ones((9, 9), np.uint8)
        merged = cv2.dilate(motion_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.5, 0.5

        # Pick the largest connected motion cluster — this is where the
        # players/ball are actively contesting play.
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 40:
            return 0.0, 0.5, 0.5

        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w / 2, y + h / 2
        else:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

        confidence = float(min(1.0, area / (width * height * 0.05)))
        return confidence, cx / width, cy / height
    except Exception:
        return 0.0, 0.5, 0.5


def analyze_window(
    video_path: str,
    video_info: VideoInfo,
    start_frame: int,
    end_frame: int,
) -> list[FrameSignal]:
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RefinementError(f"Could not open video with OpenCV: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        signals: list[FrameSignal] = []
        previous_gray: np.ndarray | None = None
        previous_ball: tuple[float, float] | None = None
        audio = extract_audio_rms(
            video_path,
            frame_to_timestamp(start_frame, video_info.fps),
            frame_to_timestamp(end_frame - start_frame + 1, video_info.fps),
            video_info.fps,
        )

        frame_number = start_frame
        while frame_number <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            small = resize_for_analysis(frame)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if previous_gray is None:
                motion = 0.0
                scoreboard_change = 0.0
            else:
                diff = cv2.absdiff(gray, previous_gray)
                motion = float(np.mean(diff))
                top = max(1, int(gray.shape[0] * 0.16))
                lower_start = int(gray.shape[0] * 0.58)
                lower_end = max(lower_start + 1, int(gray.shape[0] * 0.82))
                scoreboard_strip = float(np.mean(diff[:top, :]))
                broadcast_overlay = float(np.mean(diff[lower_start:lower_end, :])) * 0.65
                scoreboard_change = max(scoreboard_strip, broadcast_overlay)

            ball_score, ball_x, ball_y = ball_candidate(small)
            # Robust complement to the color-blob ball detector: a motion-
            # cluster centroid, used for the moving zoom track since it does
            # not depend on the ball being color-distinguishable at distance.
            cluster_score, cluster_x, cluster_y = action_cluster_center(small, previous_gray)
            if previous_ball is None:
                ball_motion = 0.0
            else:
                ball_motion = math.hypot(ball_x - previous_ball[0], ball_y - previous_ball[1])
            previous_ball = (ball_x, ball_y) if ball_score > 0 else previous_ball

            audio_index = frame_number - start_frame
            audio_score = float(audio[audio_index]) if audio_index < audio.size else 0.0
            signals.append(
                FrameSignal(
                    frame_number=frame_number,
                    timestamp=frame_to_timestamp(frame_number, video_info.fps),
                    motion=motion,
                    ball_motion=ball_motion,
                    scoreboard_change=scoreboard_change,
                    audio=audio_score,
                    subject_x=ball_x,
                    subject_y=ball_y,
                    cluster_score=cluster_score,
                    cluster_x=cluster_x,
                    cluster_y=cluster_y,
                )
            )
            previous_gray = gray
            frame_number += 1

        capture.release()
        if not signals:
            raise RefinementError("OpenCV produced no frame signals for search window.")
        return signals
    except RefinementError:
        raise
    except Exception as exc:
        raise RefinementError(f"Could not analyze frame window: {exc}") from exc


def event_weights(event_type: str) -> dict[str, float]:
    if event_type in {"goal", "penalty"}:
        # Fast shots are often missed by the ball-blob tracker, but the
        # frame-diff (motion) spike at the moment of contact is reliable.
        # Reduce ball_motion weight; raise motion weight accordingly.
        return {"motion": 0.55, "ball": 0.25, "scoreboard": 0.05, "audio": 0.15}
    if event_type == "save":
        return {"motion": 0.30, "ball": 0.45, "scoreboard": 0.05, "audio": 0.20}
    if event_type == "tackle":
        return {"motion": 0.55, "ball": 0.25, "scoreboard": 0.00, "audio": 0.20}
    if event_type == "dribble":
        return {"motion": 0.40, "ball": 0.45, "scoreboard": 0.00, "audio": 0.15}
    if event_type in {"near_miss", "shot_on_target", "free_kick"}:
        return {"motion": 0.25, "ball": 0.45, "scoreboard": 0.05, "audio": 0.25}
    if event_type == "celebration":
        return {"motion": 0.35, "ball": 0.05, "scoreboard": 0.15, "audio": 0.45}
    if event_type in {"dangerous_attack", "counter_attack", "assist"}:
        return {"motion": 0.35, "ball": 0.45, "scoreboard": 0.05, "audio": 0.15}
    if event_type == "crowd_reaction":
        return {"motion": 0.20, "ball": 0.00, "scoreboard": 0.20, "audio": 0.60}
    if event_type in {"red_card", "yellow_card"}:
        return {"motion": 0.25, "ball": 0.05, "scoreboard": 0.25, "audio": 0.45}
    return {"motion": 0.45, "ball": 0.25, "scoreboard": 0.05, "audio": 0.25}


def _find_goal_frame(
    signals: list[FrameSignal],
    fps: float,
    rough_frame: int,
) -> tuple[int, str]:
    """
    Tight-anchor goal-frame finder.

    Gemini's rough_timestamp for goals is usually already close to the real
    goal moment (within 1-3 s), since the goal-detection prompt explicitly
    asks for "ball fully past the line". So instead of searching widely
    backward (which previously caused 2-5 s of early drift), we:

    1. Look only within GOAL_SEARCH_TIGHT_RADIUS_SECONDS of rough_timestamp
       for the strongest local motion (frame-diff) spike — the ball/net
       impact frame.
    2. If a spike clears GOAL_SPIKE_ZSCORE, use it.
    3. Otherwise fall back directly to rough_timestamp's own frame — no big
       jump in either direction.
    4. HARD CLAMP: whatever is chosen, snap back to rough_frame if it strays
       more than GOAL_MAX_DRIFT_SECONDS away. This guarantees correctness
       even if the spike search picks a misleading local peak.

    Returns (signal_index, method_used).
    """
    n = len(signals)
    if n == 0:
        raise RefinementError("No signals available for goal frame search.")

    frame_numbers = np.array([s.frame_number for s in signals], dtype=np.int64)

    # Index in `signals` closest to rough_frame — used as our anchor and as
    # the ultimate fallback.
    anchor_idx = int(np.argmin(np.abs(frame_numbers - rough_frame)))

    motion_raw = np.array([s.motion for s in signals], dtype=np.float32)
    motion_n = normalize(motion_raw.tolist())

    tight_frames = int(round(GOAL_SEARCH_TIGHT_RADIUS_SECONDS * fps))
    tight_start = max(0, anchor_idx - tight_frames)
    tight_end = min(n, anchor_idx + tight_frames + 1)

    best_idx = anchor_idx
    method = "rough_timestamp_anchor"

    if tight_end > tight_start + 1:
        window_motion = motion_n[tight_start:tight_end]
        mean_m = float(window_motion.mean())
        std_m = float(window_motion.std()) if window_motion.std() > 0 else 1e-6
        peak_in_window = int(np.argmax(window_motion))
        peak_val = float(window_motion[peak_in_window])
        z_score = (peak_val - mean_m) / std_m

        if z_score >= GOAL_SPIKE_ZSCORE:
            candidate_idx = tight_start + peak_in_window
            best_idx = candidate_idx
            method = "local_strike_spike"

    # ── HARD CLAMP: never drift more than GOAL_MAX_DRIFT_SECONDS from Gemini's
    #    own rough timestamp. This is the safety net that prevents the goal
    #    moment from ever falling outside the downstream replay window.
    max_drift_frames = int(round(GOAL_MAX_DRIFT_SECONDS * fps))
    if abs(signals[best_idx].frame_number - rough_frame) > max_drift_frames:
        logging.warning(
            "Goal candidate frame drifted %.2f s from rough_timestamp — "
            "clamping back to rough timestamp.",
            abs(signals[best_idx].frame_number - rough_frame) / fps,
        )
        best_idx = anchor_idx
        method = "clamped_to_rough_timestamp"

    logging.info(
        "Goal frame selected: %.3f s (rough was %.3f s, Δ=%.2f s, method=%s)",
        signals[best_idx].timestamp,
        rough_frame / fps,
        signals[best_idx].timestamp - (rough_frame / fps),
        method,
    )
    return best_idx, method


def choose_exact_frame(
    event: dict[str, Any],
    signals: list[FrameSignal],
    fps: float,
) -> tuple[FrameSignal, float]:
    try:
        weights = event_weights(event["type"])
        motion = smooth(normalize([signal.motion for signal in signals]))
        ball = smooth(normalize([signal.ball_motion for signal in signals]))
        scoreboard = smooth(normalize([signal.scoreboard_change for signal in signals]))
        audio = smooth(normalize([signal.audio for signal in signals]))
        rough_frame = timestamp_to_frame(event["rough_timestamp"], fps)

        scores = (
            motion * weights["motion"]
            + ball * weights["ball"]
            + scoreboard * weights["scoreboard"]
            + audio * weights["audio"]
        )

        distances = np.asarray(
            [abs(signal.frame_number - rough_frame) / max(1, fps * SEARCH_RADIUS_SECONDS) for signal in signals],
            dtype=np.float32,
        )
        proximity = 1.0 - np.clip(distances, 0.0, 1.0)
        scores = scores * 0.85 + proximity * 0.15

        if event["type"] == "goal":
            best_index, method = _find_goal_frame(signals, fps, rough_frame)
            logging.info(
                "Goal refined to %.3f s via %s (rough was %.3f s, Δ=%.2f s)",
                signals[best_index].timestamp,
                method,
                event["rough_timestamp"],
                signals[best_index].timestamp - event["rough_timestamp"],
            )

        elif event["type"] in {"celebration", "crowd_reaction"}:
            threshold = max(float(scores.max()) * 0.72, float(scores.mean() + scores.std()))
            candidates = np.where(scores >= threshold)[0]
            best_index = int(candidates[0]) if candidates.size else int(np.argmax(scores))
        else:
            best_index = int(np.argmax(scores))

        confidence = max(MIN_CONFIDENCE, min(0.99, float(scores[best_index])))
        return signals[best_index], confidence
    except Exception as exc:
        raise RefinementError(f"Could not choose exact frame: {exc}") from exc


def _build_goal_subject_track(
    video_path: str,
    video_info: VideoInfo,
    effect_start_frame: int,
    fps: float,
) -> list[dict[str, float]]:
    """
    Sample the action-cluster position across the FULL goal replay window (8s
    before to 2s after effect_start_frame) so the zoom can follow play toward
    the goal instead of staying frozen on wherever the action started (e.g. a
    corner kick).

    Uses the motion-cluster centroid (action_cluster_center) rather than the
    color-blob ball detector: on broadcast-distance amateur/youth footage the
    ball is often just a few pixels and gets confused with white shirts,
    socks, field lines, and scoreboard graphics. Player/ball motion density
    is far more reliable for finding where the play actually is.

    The track is heavily smoothed and is guaranteed to end near the position
    detected AT the goal frame itself (effect_start_frame), so the zoom
    always finishes on the goal regardless of how noisy earlier samples are.

    Returns a list of {"t": seconds_from_window_start, "x": .., "y": ..}
    sorted by time. Falls back to a single centered point if tracking fails.
    """
    try:
        pre_frames = int(round(GOAL_REPLAY_PRE_SECONDS_FOR_TRACK * fps))
        post_frames = int(round(GOAL_REPLAY_POST_SECONDS_FOR_TRACK * fps))
        track_start = max(0, effect_start_frame - pre_frames)
        track_end = min(video_info.frame_count - 1, effect_start_frame + post_frames)

        if track_end <= track_start:
            return [{"t": 0.0, "x": 0.5, "y": 0.5}]

        track_signals = analyze_window(video_path, video_info, track_start, track_end)
        if not track_signals:
            return [{"t": 0.0, "x": 0.5, "y": 0.5}]

        # Find the cluster position at (or nearest to) the actual goal frame —
        # this is our anchor / ground truth for "where the goal is".
        goal_signal_idx = int(
            np.argmin([abs(s.frame_number - effect_start_frame) for s in track_signals])
        )

        # Collect raw cluster samples, only trusting frames with a reasonable
        # confidence score; carry forward the last trusted position otherwise
        # so the path doesn't jump to frame-center on a single bad frame.
        raw_x = np.array([s.cluster_x for s in track_signals], dtype=np.float32)
        raw_y = np.array([s.cluster_y for s in track_signals], dtype=np.float32)
        raw_conf = np.array([s.cluster_score for s in track_signals], dtype=np.float32)

        CONFIDENCE_FLOOR = 0.08
        cleaned_x = raw_x.copy()
        cleaned_y = raw_y.copy()
        last_x, last_y = raw_x[0], raw_y[0]
        for i in range(len(track_signals)):
            if raw_conf[i] >= CONFIDENCE_FLOOR:
                last_x, last_y = raw_x[i], raw_y[i]
            cleaned_x[i] = last_x
            cleaned_y[i] = last_y

        # Heavy smoothing so the zoom pans rather than jitters frame-to-frame.
        smooth_kernel = max(3, int(round(fps * 0.6)))
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        smoothed_x = smooth(cleaned_x, kernel_size=smooth_kernel)
        smoothed_y = smooth(cleaned_y, kernel_size=smooth_kernel)

        # Anchor the END of the track to the goal frame's own detected
        # position, blending smoothly over the final ~1.5s so the zoom is
        # guaranteed to land on the actual goal regardless of upstream noise.
        anchor_x = float(raw_x[goal_signal_idx]) if raw_conf[goal_signal_idx] >= CONFIDENCE_FLOOR else float(smoothed_x[goal_signal_idx])
        anchor_y = float(raw_y[goal_signal_idx]) if raw_conf[goal_signal_idx] >= CONFIDENCE_FLOOR else float(smoothed_y[goal_signal_idx])

        blend_frames = int(round(1.5 * fps))
        blend_start = max(0, goal_signal_idx - blend_frames)
        for i in range(blend_start, len(track_signals)):
            denom = max(1, len(track_signals) - 1 - blend_start)
            weight = (i - blend_start) / denom
            weight = min(1.0, weight)
            smoothed_x[i] = smoothed_x[i] * (1 - weight) + anchor_x * weight
            smoothed_y[i] = smoothed_y[i] * (1 - weight) + anchor_y * weight

        # Sample roughly every 0.3s to keep the track light while still
        # smooth enough for a moving zoom keyframe path.
        sample_stride_frames = max(1, int(round(0.3 * fps)))

        points: list[dict[str, float]] = []
        for i, signal in enumerate(track_signals):
            offset = (signal.frame_number - track_start) % sample_stride_frames
            is_boundary = i in (0, len(track_signals) - 1) or signal.frame_number in (track_start, track_end)
            is_goal_frame = i == goal_signal_idx
            if offset != 0 and not is_boundary and not is_goal_frame:
                continue
            x = min(0.85, max(0.15, float(smoothed_x[i])))
            y = min(0.85, max(0.15, float(smoothed_y[i])))
            points.append(
                {
                    "t": round((signal.frame_number - track_start) / fps, 3),
                    "x": round(x, 4),
                    "y": round(y, 4),
                }
            )

        if not points:
            return [{"t": 0.0, "x": 0.5, "y": 0.5}]
        return points
    except Exception as exc:
        logging.warning("Could not build goal subject track: %s — using single point.", exc)
        return [{"t": 0.0, "x": 0.5, "y": 0.5}]


def refine_event(event: dict[str, Any], video_info: VideoInfo) -> dict[str, Any]:
    try:
        rough_frame = timestamp_to_frame(event["rough_timestamp"], video_info.fps)

        if event["type"] == "goal":
            # Small symmetric window: just enough margin around rough_timestamp
            # for the tight-anchor search (±3s) plus the hard clamp (±2.5s),
            # with a little extra padding for the audio/scoreboard signals
            # used elsewhere. No more large backward-biased search.
            window_radius_seconds = GOAL_SEARCH_TIGHT_RADIUS_SECONDS + 2.0
            radius_frames = int(round(window_radius_seconds * video_info.fps))
            start_frame  = max(0, rough_frame - radius_frames)
            end_frame    = min(video_info.frame_count - 1, rough_frame + radius_frames)
            logging.info(
                "Goal search window: %.2f s – %.2f s (rough %.2f s)",
                frame_to_timestamp(start_frame, video_info.fps),
                frame_to_timestamp(end_frame,   video_info.fps),
                event["rough_timestamp"],
            )
        else:
            radius_frames = int(round(SEARCH_RADIUS_SECONDS * video_info.fps))
            start_frame   = max(0, rough_frame - radius_frames)
            end_frame     = min(video_info.frame_count - 1, rough_frame + radius_frames)

        signals = analyze_window(INPUT_VIDEO, video_info, start_frame, end_frame)
        best_signal, confidence = choose_exact_frame(event, signals, video_info.fps)
        effect_start_frame = best_signal.frame_number
        effect_duration = GOAL_EFFECT_DURATION_SECONDS if event["type"] in {"goal", "penalty"} else 0.0
        effect_end_frame = effect_start_frame + int(round(effect_duration * video_info.fps))

        # The color-blob ball detector is unreliable at broadcast distance
        # (small ball, white shirts/socks/lines causing false positives).
        # When its confidence is low, prefer the more robust motion-cluster
        # centroid for framing the zoom, instead of defaulting to a possibly
        # wrong ball position or a dead-center 0.5/0.5 fallback.
        BALL_CONFIDENCE_FLOOR = 0.15
        if best_signal.cluster_score >= 0.08:
            subject_x = best_signal.cluster_x
            subject_y = best_signal.cluster_y
        else:
            subject_x = best_signal.subject_x
            subject_y = best_signal.subject_y

        result = {
            "type": event["type"],
            "rough_timestamp": round(event["rough_timestamp"], 3),
            "exact_timestamp": round(frame_to_timestamp(effect_start_frame, video_info.fps), 6),
            "frame_number": int(effect_start_frame),
            "effect_start_frame": int(effect_start_frame),
            "effect_end_frame": int(effect_end_frame),
            "effect_duration": effect_duration,
            "confidence": round(confidence, 3),
            "subject_center": {
                "x": round(float(subject_x), 4),
                "y": round(float(subject_y), 4),
            },
            "description": event.get("description", ""),
            "coarse_confidence": round(float(event.get("confidence", 0.5)), 3),
            "refinement_mode": "opencv",
        }

        if event["type"] == "goal":
            # Additive field: a short track of action-cluster positions
            # spanning the full replay window, so the zoom can follow play
            # from the strike point (e.g. a corner) into the goal, instead of
            # staying frozen on subject_center the whole time. Existing
            # consumers that only read subject_center are unaffected.
            result["subject_track"] = _build_goal_subject_track(
                INPUT_VIDEO, video_info, effect_start_frame, video_info.fps
            )

        return result
    except Exception as exc:
        raise RefinementError(
            f"Could not refine {event.get('type')} at {event.get('rough_timestamp')}s: {exc}"
        ) from exc


def approximate_event(event: dict[str, Any], video_info: VideoInfo) -> dict[str, Any]:
    try:
        frame_number = max(
            0,
            min(video_info.frame_count - 1, timestamp_to_frame(event["rough_timestamp"], video_info.fps)),
        )
        return {
            "type": event["type"],
            "rough_timestamp": round(event["rough_timestamp"], 3),
            "exact_timestamp": round(frame_to_timestamp(frame_number, video_info.fps), 6),
            "frame_number": int(frame_number),
            "effect_start_frame": int(frame_number),
            "effect_end_frame": int(frame_number),
            "effect_duration": 0.0,
            "confidence": round(max(MIN_CONFIDENCE, float(event.get("confidence", 0.5))), 3),
            "subject_center": {"x": 0.5, "y": 0.5},
            "description": event.get("description", ""),
            "coarse_confidence": round(float(event.get("confidence", 0.5)), 3),
            "refinement_mode": "approximate",
        }
    except Exception as exc:
        raise RefinementError(
            f"Could not approximate {event.get('type')} at {event.get('rough_timestamp')}s: {exc}"
        ) from exc


def refine_events() -> None:
    try:
        load_dotenv()
        ensure_inputs()
        video_info = probe_video(INPUT_VIDEO)
        coarse_events = compress_coarse_duplicates(normalize_coarse_events(read_json(COARSE_JSON)))
        if not coarse_events:
            raise RefinementError("No valid coarse events to refine.")

        high_precision_types = get_high_precision_types()
        logging.info(
            "OpenCV high-precision refinement enabled for: %s",
            ", ".join(sorted(high_precision_types)) or "none",
        )

        refined: list[dict[str, Any]] = []
        for index, event in enumerate(coarse_events, start=1):
            try:
                if event["type"] not in high_precision_types:
                    logging.info(
                        "Fast timestamp pass %s %d/%d around %.3fs...",
                        event["type"],
                        index,
                        len(coarse_events),
                        event["rough_timestamp"],
                    )
                    refined.append(approximate_event(event, video_info))
                    continue

                logging.info(
                    "Refining %s %d/%d around %.3fs...",
                    event["type"],
                    index,
                    len(coarse_events),
                    event["rough_timestamp"],
                )
                refined.append(refine_event(event, video_info))
            except Exception as exc:
                logging.error("Skipping event %d: %s", index, exc)

        if not refined:
            raise RefinementError("No events were refined successfully.")

        refined.sort(key=lambda item: item["frame_number"])
        payload = {
            "video": {
                "fps": video_info.fps,
                "frame_count": video_info.frame_count,
                "duration": video_info.duration,
                "width": video_info.width,
                "height": video_info.height,
            },
            "events": refined,
        }
        with open(OUTPUT_JSON, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)

        logging.info("Saved %d refined events to %s", len(refined), OUTPUT_JSON)
    except RefinementError:
        raise
    except Exception as exc:
        raise RefinementError(f"Event refinement failed: {exc}") from exc


def main() -> int:
    try:
        configure_logging()
        refine_events()
        return 0
    except RefinementError as exc:
        logging.error("Error: %s", exc)
        return 1
    except Exception as exc:
        logging.error("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())