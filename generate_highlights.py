"""
generate_highlighs.py
Stage 3: generate highlights from frame-refined event timings.

The input to this stage is refined_events.json, produced by event_refiner.py.
Effects are aligned from exact frame numbers, not loose timestamps.

Fix applied:
  • clip_ranges() for "goal" events previously misused event_context_seconds(),
    which returns (pre, post) = (8.0, 2.0) for goals. The old code used
    pre_frames for BOTH the clip_start subtraction AND mistakenly fed the same
    pre-roll value into the variable meant to hold post_frames, producing a
    clip window that was symmetric around effect_start (±8s) instead of
    asymmetric (-8s / +2s). This meant the replay/slowmo/zoom segment — which
    is built from this exact clip_start/clip_end window — was centered on the
    8-seconds-before point rather than the actual goal frame, so the visible
    effect played out mostly BEFORE the ball crossed the line and cut off
    right around the real goal moment.
  • Now clip_ranges() for goals explicitly uses GOAL_REPLAY_PRE_SECONDS (8s)
    before effect_start and GOAL_REPLAY_POST_SECONDS (2s) after effect_start,
    with effect_start being the exact frame-accurate goal frame produced by
    event_refiner.py. This window is what gets duplicated, slowed to 0.5x,
    and zoomed in clip_filter_complex(), so the goal moment now sits correctly
    inside the replay rather than at its edge or outside it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dotenv import load_dotenv


INPUT_VIDEO = os.path.join(".", "match.mp4")
REFINED_JSON = os.path.join(".", "refined_events.json")
OUTPUT_VIDEO = os.path.join(".", "highlights.mp4")
INCLUDED_MOMENTS_JSON = os.path.join(".", "included_moments.json")
CLIPS_DIR = os.path.join(".", "clips")
CONCAT_FILE = os.path.join(CLIPS_DIR, "concat_list.txt")

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30
VIDEO_BITRATE = "4000k"
AUDIO_BITRATE = "192k"
MIN_SUCCESSFUL_CLIPS = 3

TARGET_MIN_SECONDS = 12 * 60
TARGET_MAX_SECONDS = 15 * 60
TARGET_PREFERRED_SECONDS = 14.5 * 60
DEFAULT_PRE_ROLL_SECONDS = 7.0
DEFAULT_POST_ROLL_SECONDS = 5.0
GOAL_SLOWMO_OUTPUT_SECONDS = 9.0
SLOWMO_SPEED = 0.5
GOAL_EFFECT_SECONDS = GOAL_SLOWMO_OUTPUT_SECONDS * SLOWMO_SPEED

GOAL_REPLAY_PRE_SECONDS = 7.0
GOAL_REPLAY_POST_SECONDS = 4.0

SAVE_EFFECT_SECONDS = 3.5
TRANSITION_SECONDS = 0.2
MAX_WORKERS = 2
MAX_GOAL_MOMENT_SECONDS = 42.0
MAX_STANDARD_MOMENT_SECONDS = 28.0
EXPANDED_GOAL_MOMENT_SECONDS = 75.0
EXPANDED_STANDARD_MOMENT_SECONDS = 60.0
TIMELINE_SEGMENT_SECONDS = 600
MIN_EVENT_POOL_COVERAGE_RATIO = 0.65
MIN_SELECTED_SEGMENT_RATIO = 0.5
CRITICAL_EVENT_TYPES = {"goal", "penalty", "red_card"}
EVENT_BASE_SCORES = {
    "goal": 100,
    "penalty": 96,
    "red_card": 94,
    "shot_on_target": 91,
    "assist": 88,
    "near_miss": 90,
    "save": 85,
    "free_kick": 82,
    "yellow_card": 73,
    "dangerous_attack": 80,
    "counter_attack": 78,
    "dribble": 70,
    "celebration": 68,
    "crowd_reaction": 62,
    "tackle": 60,
    "foul": 45,
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


class HighlightGenerationError(Exception):
    """Raised when highlight generation fails."""


def configure_logging() -> None:
    try:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    except Exception as exc:
        print(f"Logging setup failed: {exc}")


def ensure_inputs() -> None:
    try:
        if not os.path.isfile(INPUT_VIDEO):
            raise HighlightGenerationError(f"Missing input video: {INPUT_VIDEO}")
        if not os.path.isfile(REFINED_JSON):
            raise HighlightGenerationError(
                f"Missing {REFINED_JSON}. Run event_refiner.py after analyze.py."
            )
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                raise HighlightGenerationError(f"{tool} was not found on PATH.")
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Could not validate inputs: {exc}") from exc


def run_command(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "No stderr output."
        raise HighlightGenerationError(f"{label} failed: {stderr}") from exc
    except Exception as exc:
        raise HighlightGenerationError(f"{label} could not start: {exc}") from exc


def read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8-sig") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            raise HighlightGenerationError(f"{path} must contain a JSON object.")
        return payload
    except json.JSONDecodeError as exc:
        raise HighlightGenerationError(f"Invalid JSON in {path}: {exc}") from exc
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Could not read {path}: {exc}") from exc


def load_refined_events() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = read_json(REFINED_JSON)
        video = payload.get("video")
        events = payload.get("events")
        if not isinstance(video, dict) or not isinstance(events, list) or not events:
            raise HighlightGenerationError("refined_events.json must contain video and events.")
        fps = float(video["fps"])
        frame_count = int(video["frame_count"])
        normalized = []
        for index, event in enumerate(events, start=1):
            exact_frame = int(event["frame_number"])
            effect_start = int(event.get("effect_start_frame", exact_frame))
            event_type = str(event["type"]).lower()
            effect_seconds = source_effect_seconds(event_type)
            effect_end = effect_start + int(round(effect_seconds * fps))
            normalized.append(
                {
                    "index": index,
                    "type": event_type,
                    "frame_number": exact_frame,
                    "effect_start_frame": effect_start,
                    "effect_end_frame": effect_end,
                    "exact_timestamp": float(event["exact_timestamp"]),
                    "confidence": float(event.get("confidence", 0.5)),
                    "description": str(event.get("description", "")),
                    "subject_center": event.get("subject_center", {"x": 0.5, "y": 0.5}),
                    "effect_duration": float(event.get("effect_duration", 0.0)),
                    "refinement_mode": event.get("refinement_mode"),
                }
            )
        normalized.sort(key=lambda item: item["frame_number"])
        return video, normalized
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Could not load refined events: {exc}") from exc


def frame_to_timestamp(frame_number: int, fps: float) -> float:
    return frame_number / fps


def prepare_clips_directory() -> None:
    try:
        if os.path.isdir(CLIPS_DIR):
            shutil.rmtree(CLIPS_DIR)
        os.makedirs(CLIPS_DIR, exist_ok=True)
    except Exception as exc:
        raise HighlightGenerationError(f"Could not prepare clips directory: {exc}") from exc


def escape_drawtext(text: str) -> str:
    try:
        return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    except Exception as exc:
        raise HighlightGenerationError(f"Could not escape drawtext: {exc}") from exc



def base_video_filter() -> str:
    return (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        "eq=contrast=1.04:saturation=1.08:brightness=0.01,"
        "unsharp=5:5:0.55:3:3:0.25,"
        f"setsar=1,fps={TARGET_FPS},format=yuv420p"
    )


def source_effect_seconds(event_type: str) -> float:
    if event_type == "goal":
        return goal_replay_source_seconds()
    return 0.0


def goal_slowmo_output_seconds() -> float:
    try:
        raw_value = os.getenv("GOAL_SLOWMO_OUTPUT_SECONDS", str(GOAL_SLOWMO_OUTPUT_SECONDS))
        return max(7.0, min(10.0, float(raw_value)))
    except (TypeError, ValueError):
        return GOAL_SLOWMO_OUTPUT_SECONDS


def goal_source_effect_seconds() -> float:
    return goal_slowmo_output_seconds() * SLOWMO_SPEED


def goal_replay_source_seconds() -> float:
    return GOAL_REPLAY_PRE_SECONDS + GOAL_REPLAY_POST_SECONDS


def event_context_seconds(event_type: str) -> tuple[float, float]:
    context = {
        "goal": (GOAL_REPLAY_PRE_SECONDS, GOAL_REPLAY_POST_SECONDS),
        "penalty": (10.0, 10.0),
        "red_card": (6.0, 8.0),
        "yellow_card": (4.0, 5.0),
        "save": (6.0, 7.0),
        "near_miss": (7.0, 6.0),
        "shot_on_target": (7.0, 6.0),
        "free_kick": (8.0, 6.0),
        "assist": (7.0, 5.0),
        "dangerous_attack": (9.0, 5.0),
        "counter_attack": (8.0, 5.0),
        "dribble": (5.0, 4.0),
        "celebration": (2.0, 8.0),
        "tackle": (4.0, 4.0),
        "foul": (3.0, 5.0),
        "crowd_reaction": (2.0, 5.0),
    }
    return context.get(event_type, (DEFAULT_PRE_ROLL_SECONDS, DEFAULT_POST_ROLL_SECONDS))


def effect_config(event_type: str) -> dict[str, Any]:
    if event_type == "goal":
        return {"slowmo": True, "zoom": True, "zoom_max": 1.28}
    return {"slowmo": False, "zoom": False, "zoom_max": 1.0}


def audio_tempo_filter(speed: float) -> str:
    try:
        filters: list[str] = []
        remaining = speed
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        filters.append(f"atempo={remaining:.6f}")
        return ",".join(filters)
    except Exception as exc:
        raise HighlightGenerationError(f"Could not build audio tempo filter: {exc}") from exc


def _build_moving_center_expr(track: list[dict[str, float]], axis: str, fps: float) -> str:
    """
    Build an ffmpeg expression that linearly interpolates between track points
    over time (in output frames `on`), for use as zoompan's x or y center.

    track entries are {"t": seconds_from_clip_start, "x": .., "y": ..}.
    Falls back to a constant if the track has fewer than 2 usable points.
    """
    points = [(float(p["t"]), float(p[axis])) for p in track if "t" in p and axis in p]
    points.sort(key=lambda item: item[0])

    if len(points) < 2:
        value = points[0][1] if points else 0.5
        return f"{value:.4f}"

    # Build nested if() chain: for on between frame(t_i) and frame(t_{i+1}),
    # linearly interpolate between value_i and value_{i+1}.
    expr = f"{points[-1][1]:.4f}"
    for index in range(len(points) - 2, -1, -1):
        t0, v0 = points[index]
        t1, v1 = points[index + 1]
        frame0 = t0 * fps
        frame1 = t1 * fps
        if frame1 <= frame0:
            continue
        # linear interpolation: v0 + (v1-v0) * (on-frame0) / (frame1-frame0)
        interp = (
            f"({v0:.4f}+({v1:.4f}-{v0:.4f})*"
            f"(on-{frame0:.2f})/({frame1:.2f}-{frame0:.2f}))"
        )
        expr = f"if(lte(on,{frame1:.2f}),{interp},{expr})"
    return expr


def zoompan_filter(event: dict[str, Any], zoom_max: float, fps: float = TARGET_FPS) -> str:
    try:
        ramp_frames = max(1, int(round(2 * TARGET_FPS)))
        zoom_expr = f"if(lte(on,{ramp_frames}),1+({zoom_max}-1)*on/{ramp_frames},{zoom_max})"

        track = event.get("subject_track")
        if isinstance(track, list) and len(track) >= 2:
            # Moving zoom: follow the ball across the replay (e.g. from a
            # corner strike into the goal) instead of staying fixed on the
            # point where the action started.
            cx_expr = _build_moving_center_expr(track, "x", TARGET_FPS)
            cy_expr = _build_moving_center_expr(track, "y", TARGET_FPS)
            x_expr = f"iw*({cx_expr})-(iw/zoom/2)"
            y_expr = f"ih*({cy_expr})-(ih/zoom/2)"
        else:
            # Fallback: original static-center behaviour, unchanged.
            center = event.get("subject_center") or {}
            cx = min(0.9, max(0.1, float(center.get("x", 0.5))))
            cy = min(0.9, max(0.1, float(center.get("y", 0.5))))
            x_expr = f"iw*{cx}-(iw/zoom/2)"
            y_expr = f"ih*{cy}-(ih/zoom/2)"

        # zoompan keeps the tracked subject around the crop center while ramping.
        return (
            "zoompan="
            f"z='{zoom_expr}':"
            f"x='{x_expr}':"
            f"y='{y_expr}':"
            f"d=1:s={TARGET_WIDTH}x{TARGET_HEIGHT}:fps={TARGET_FPS}"
        )
    except Exception as exc:
        raise HighlightGenerationError(f"Could not build zoompan filter: {exc}") from exc


def clip_ranges(event: dict[str, Any], fps: float, frame_count: int) -> dict[str, int]:
    """
    Compute the source clip window [clip_start, clip_end] (in frames) and the
    "effect" sub-window inside it (used for goal slow-mo replay markers).

    For goals: effect_start_frame IS the exact frame-accurate goal moment
    produced by event_refiner.py. The window must be exactly
    GOAL_REPLAY_PRE_SECONDS (8s) BEFORE that frame and GOAL_REPLAY_POST_SECONDS
    (2s) AFTER it — this entire window gets duplicated, slowed to SLOWMO_SPEED,
    and zoomed in clip_filter_complex(), so the goal frame must sit correctly
    inside this window (at roughly the 8s mark from the window start), not at
    an edge or outside it.
    """
    try:
        effect_start = int(event["effect_start_frame"])

        if event["type"] == "goal":
            goal_pre_frames = int(round(GOAL_REPLAY_PRE_SECONDS * fps))
            goal_post_frames = int(round(GOAL_REPLAY_POST_SECONDS * fps))
            clip_start = max(0, effect_start - goal_pre_frames)
            clip_end = min(frame_count, effect_start + goal_post_frames)
            effect_end = clip_end
        else:
            pre_seconds, post_seconds = event_context_seconds(event["type"])
            pre_frames = int(round(pre_seconds * fps))
            post_frames = int(round(post_seconds * fps))
            effect_seconds = source_effect_seconds(event["type"])
            exact_effect_frames = int(round(effect_seconds * fps))
            effect_end = min(frame_count, effect_start + exact_effect_frames)
            if exact_effect_frames <= 0:
                effect_end = effect_start
            clip_start = max(0, effect_start - pre_frames)
            clip_end = min(frame_count, effect_end + post_frames)

            if "clip_start_frame" in event:
                clip_start = max(0, int(event["clip_start_frame"]))
            if "clip_end_frame" in event:
                clip_end = min(frame_count, int(event["clip_end_frame"]))

        if clip_end <= clip_start:
            raise HighlightGenerationError("Clip end frame must be after start frame.")
        return {
            "clip_start": clip_start,
            "effect_start": effect_start,
            "effect_end": effect_end,
            "clip_end": clip_end,
        }
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Could not compute clip ranges: {exc}") from exc


def clip_filter_complex(
    event: dict[str, Any],
    video: dict[str, Any],
    has_audio: bool,
) -> tuple[str, list[str]]:
    try:
        fps = float(video["fps"])
        frame_count = int(video["frame_count"])
        ranges = clip_ranges(event, fps, frame_count)

        clip_start = ranges["clip_start"]
        clip_end = ranges["clip_end"]
        source_duration = max(0.0, (clip_end - clip_start) / fps)

        if event["type"] == "goal":
            config = effect_config(event["type"])
            replay_duration = source_duration / SLOWMO_SPEED
            output_duration = source_duration + replay_duration
            fade_out_start = max(0.0, output_duration - 0.3)
            replay_text = escape_drawtext("REPLAY")
            replay_drawtext = (
                "drawtext="
                f"text='{replay_text}':x=60:y=60:fontsize=58:fontcolor=white:"
                "box=1:boxcolor=black@0.45:boxborderw=18"
            )
            replay_filters = [
                base_video_filter(),
                zoompan_filter(event, float(config["zoom_max"])),
                f"setpts=(PTS-STARTPTS)/{SLOWMO_SPEED:.6f}",
                replay_drawtext,
            ]
            video_filter = (
                "[0:v]split=2[vorigsrc][vreplaysrc];"
                f"[vorigsrc]trim=start_frame={clip_start}:end_frame={clip_end},"
                f"setpts=PTS-STARTPTS,{base_video_filter()}[v0];"
                f"[vreplaysrc]trim=start_frame={clip_start}:end_frame={clip_end},"
                f"setpts=PTS-STARTPTS,{','.join(replay_filters)}[v1];"
                "[v0][v1]concat=n=2:v=1:a=0,"
                f"fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out_start:.6f}:d=0.3[v]"
            )

            if not has_audio:
                return video_filter, ["-map", "[v]"]

            clip_start_time = frame_to_timestamp(clip_start, fps)
            clip_end_time = frame_to_timestamp(clip_end, fps)
            audio_filter = (
                "[0:a]asplit=2[aorigsrc][areplaysrc];"
                f"[aorigsrc]atrim=start={clip_start_time:.6f}:end={clip_end_time:.6f},"
                "asetpts=PTS-STARTPTS[a0];"
                f"[areplaysrc]atrim=start={clip_start_time:.6f}:end={clip_end_time:.6f},"
                f"asetpts=PTS-STARTPTS,{audio_tempo_filter(SLOWMO_SPEED)}[a1];"
                "[a0][a1]concat=n=2:v=0:a=1,"
                "aresample=48000,aformat=channel_layouts=stereo[a]"
            )
            return f"{video_filter};{audio_filter}", ["-map", "[v]", "-map", "[a]"]

        output_duration = source_duration
        fade_out_start = max(0.0, output_duration - 0.25)
        video_filter = (
            "[0:v]"
            f"trim=start_frame={clip_start}:end_frame={clip_end},"
            f"setpts=PTS-STARTPTS,{base_video_filter()},"
            f"fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out_start:.6f}:d=0.25[v]"
        )
        if not has_audio:
            return video_filter, ["-map", "[v]"]

        clip_start_time = frame_to_timestamp(clip_start, fps)
        clip_end_time = frame_to_timestamp(clip_end, fps)
        audio_filter = (
            f"[0:a]atrim=start={clip_start_time:.6f}:end={clip_end_time:.6f},"
            "asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.25,"
            f"afade=t=out:st={fade_out_start:.6f}:d=0.25,"
            "aresample=48000,aformat=channel_layouts=stereo[a]"
        )
        return f"{video_filter};{audio_filter}", ["-map", "[v]", "-map", "[a]"]
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Could not build frame-accurate filter: {exc}") from exc

def probe_has_audio() -> bool:
    try:
        result = run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                INPUT_VIDEO,
            ],
            "Audio probe",
        )
        metadata = json.loads(result.stdout)
        return any(stream.get("codec_type") == "audio" for stream in metadata.get("streams", []))
    except Exception:
        return False


def extract_clip(event: dict[str, Any], video: dict[str, Any], has_audio: bool) -> str:
    try:
        output_path = os.path.join(CLIPS_DIR, f"clip_{event['index']:03d}.mp4")
        filter_complex, map_args = clip_filter_complex(event, video, has_audio)
        if event["type"] == "goal":
            logging.info(
                "Inserting %.1fs replay at %.1fx speed with zoom for goal at %.3fs.",
                goal_replay_source_seconds(),
                SLOWMO_SPEED,
                float(event["exact_timestamp"]),
            )
        command = [
            "ffmpeg",
            "-y",
            "-i",
            INPUT_VIDEO,
            "-filter_complex",
            filter_complex,
            *map_args,
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-b:v",
            VIDEO_BITRATE,
            "-maxrate",
            VIDEO_BITRATE,
            "-bufsize",
            "8000k",
            "-r",
            str(TARGET_FPS),
            "-c:a",
            "aac",
            "-b:a",
            AUDIO_BITRATE,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-threads",
            "0",
            output_path,
        ]
        run_command(command, f"Extract frame-accurate clip #{event['index']}")
        return output_path
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Could not extract clip #{event.get('index')}: {exc}") from exc


def create_transition_clip(index: int, has_audio: bool) -> str:
    try:
        output_path = os.path.join(CLIPS_DIR, f"transition_{index:03d}.mp4")
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-t",
            f"{TRANSITION_SECONDS:.6f}",
            "-i",
            f"color=c=black:s={TARGET_WIDTH}x{TARGET_HEIGHT}:r={TARGET_FPS}",
        ]
        if has_audio:
            command.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{TRANSITION_SECONDS:.6f}",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-shortest",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                ]
            )
        else:
            command.extend(["-map", "0:v:0"])
        command.extend(
            [
                "-c:v",
                "libx264",
                "-b:v",
                VIDEO_BITRATE,
                "-r",
                str(TARGET_FPS),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                AUDIO_BITRATE,
                "-ar",
                "48000",
                "-ac",
                "2",
                output_path,
            ]
        )
        run_command(command, f"Create transition #{index}")
        return output_path
    except Exception as exc:
        raise HighlightGenerationError(f"Could not create transition #{index}: {exc}") from exc


def concat_path_line(path: str) -> str:
    normalized = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
    return f"file '{normalized}'"


def write_concat_file(paths: list[str]) -> None:
    try:
        with open(CONCAT_FILE, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(concat_path_line(path) for path in paths))
            file_obj.write("\n")
    except Exception as exc:
        raise HighlightGenerationError(f"Could not write concat file: {exc}") from exc


def merge_clips(paths: list[str]) -> None:
    try:
        write_concat_file(paths)
        run_command(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                CONCAT_FILE,
                "-c",
                "copy",
                OUTPUT_VIDEO,
            ],
            "Merge final highlights",
        )
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Could not merge clips: {exc}") from exc


def verify_output() -> None:
    try:
        if not os.path.isfile(OUTPUT_VIDEO):
            raise HighlightGenerationError("highlights.mp4 was not created.")
        if os.path.getsize(OUTPUT_VIDEO) <= 1_000_000:
            raise HighlightGenerationError("highlights.mp4 is smaller than 1MB.")
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Could not verify output: {exc}") from exc


def write_included_events(events: list[dict[str, Any]]) -> None:
    try:
        payload = {
            "events": [
                {
                    "type": event["type"],
                    "exact_timestamp": event["exact_timestamp"],
                    "frame_number": event["frame_number"],
                    "effect_duration": source_effect_seconds(event["type"]),
                    "slowmo_output_duration": (
                        goal_replay_source_seconds() / SLOWMO_SPEED
                        if event["type"] == "goal"
                        else None
                    ),
                    "merged_event_count": event.get("merged_event_count", 1),
                    "merged_event_types": event.get("merged_event_types", [event["type"]]),
                    "refinement_mode": event.get("refinement_mode"),
                    "score": event.get("score"),
                    "confidence": event["confidence"],
                    "description": event["description"],
                }
                for event in events
            ]
        }
        with open(INCLUDED_MOMENTS_JSON, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)
    except Exception as exc:
        raise HighlightGenerationError(f"Could not write included events: {exc}") from exc


def cleanup() -> None:
    try:
        if os.path.isdir(CLIPS_DIR):
            shutil.rmtree(CLIPS_DIR)
    except Exception as exc:
        logging.warning("Could not clean clips directory: %s", exc)


def build_concat_sequence(clip_paths: list[str], has_audio: bool) -> list[str]:
    sequence: list[str] = []
    for index, clip_path in enumerate(clip_paths, start=1):
        sequence.append(clip_path)
        if index < len(clip_paths):
            sequence.append(create_transition_clip(index, has_audio))
    return sequence


def event_score(event: dict[str, Any]) -> float:
    base = EVENT_BASE_SCORES.get(event["type"], 50)
    confidence_bonus = max(0.0, min(1.0, event.get("confidence", 0.5))) * 15
    description = event.get("description", "").lower()
    text_bonus = 0
    for keyword in ("great", "dangerous", "clear chance", "shot", "cross", "counter", "amazing"):
        if keyword in description:
            text_bonus += 2
    return round(base + confidence_bonus + text_bonus, 3)


def event_priority(event: dict[str, Any]) -> int:
    return EVENT_PRIORITY.get(event["type"], 50)


def timeline_segment_seconds() -> int:
    try:
        return max(60, int(os.getenv("COVERAGE_SEGMENT_SECONDS", str(TIMELINE_SEGMENT_SECONDS))))
    except (TypeError, ValueError):
        return TIMELINE_SEGMENT_SECONDS


def timeline_segment_count(video: dict[str, Any]) -> int:
    duration = float(video.get("duration", 0))
    if duration <= 0:
        frame_count = int(video.get("frame_count", 0))
        fps = max(1.0, float(video.get("fps", 1)))
        duration = frame_count / fps
    return max(1, int(math.ceil(duration / timeline_segment_seconds())))


def event_segment(event: dict[str, Any], video: dict[str, Any]) -> int:
    segment_count = timeline_segment_count(video)
    return min(
        segment_count - 1,
        max(0, int(float(event.get("exact_timestamp", 0.0)) // timeline_segment_seconds())),
    )


def validate_event_pool_coverage(events: list[dict[str, Any]], video: dict[str, Any]) -> None:
    """Fail fast when the analysis only found events from the opening part."""
    if not events:
        raise HighlightGenerationError("No events are available for highlight generation.")

    duration = float(video.get("duration", 0))
    if duration < timeline_segment_seconds() * 2:
        return

    latest = max(float(event.get("exact_timestamp", 0.0)) for event in events)
    coverage_ratio = latest / duration
    if coverage_ratio < MIN_EVENT_POOL_COVERAGE_RATIO:
        raise HighlightGenerationError(
            "Event timeline only covers the opening part of the match: "
            f"latest event is at {latest / 60:.1f} min of {duration / 60:.1f} min. "
            "Run analyze.py again so Gemini/OpenAI returns moments from the full match."
        )


def range_overlap_seconds(first: dict[str, int], second: dict[str, int], fps: float) -> float:
    overlap = min(first["clip_end"], second["clip_end"]) - max(first["clip_start"], second["clip_start"])
    return max(0.0, overlap / fps)


def ranges_overlap_significantly(first: dict[str, int], second: dict[str, int], fps: float) -> bool:
    overlap = range_overlap_seconds(first, second, fps)
    if overlap <= 0:
        return False
    first_duration = max(1.0, (first["clip_end"] - first["clip_start"]) / fps)
    second_duration = max(1.0, (second["clip_end"] - second["clip_start"]) / fps)
    first_center = (first["clip_start"] + first["clip_end"]) / 2
    second_center = (second["clip_start"] + second["clip_end"]) / 2
    center_distance = abs(first_center - second_center) / fps
    overlap_ratio = overlap / min(first_duration, second_duration)
    return overlap_ratio >= 0.90 or (overlap >= 15.0 and center_distance <= 5.0)


def merge_description_text(events: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for event in events:
        description = str(event.get("description", "")).strip()
        if description and description not in texts:
            texts.append(description)
    return " | ".join(texts[:4])


def cap_merged_range(
    start_frame: int,
    end_frame: int,
    anchor_frame: int,
    fps: float,
    max_seconds: float,
) -> tuple[int, int]:
    max_frames = int(round(max_seconds * fps))
    if end_frame - start_frame <= max_frames:
        return start_frame, end_frame

    before_seconds = 14.0 if max_seconds == MAX_GOAL_MOMENT_SECONDS else 9.0
    capped_start = max(start_frame, anchor_frame - int(round(before_seconds * fps)))
    capped_end = min(end_frame, capped_start + max_frames)
    if capped_end - capped_start < max_frames:
        capped_start = max(start_frame, capped_end - max_frames)
    return capped_start, capped_end


def merge_event_cluster(cluster: list[dict[str, Any]], video: dict[str, Any]) -> dict[str, Any]:
    fps = float(video["fps"])
    frame_count = int(video["frame_count"])
    ranges = [clip_ranges(event, fps, frame_count) for event in cluster]
    winner = max(
        cluster,
        key=lambda event: (event_priority(event), event_score(event), event.get("confidence", 0.5)),
    )
    merged = dict(winner)
    merged.pop("_range", None)
    clip_start = min(item["clip_start"] for item in ranges)
    clip_end = max(item["clip_end"] for item in ranges)

    if winner["type"] == "goal":
        # Never let merging stretch a goal's clip window away from the exact
        # goal frame; always re-anchor to the canonical 8s-pre/2s-post window
        # around the winner's own effect_start_frame.
        goal_pre_frames = int(round(GOAL_REPLAY_PRE_SECONDS * fps))
        goal_post_frames = int(round(GOAL_REPLAY_POST_SECONDS * fps))
        anchor = int(winner["effect_start_frame"])
        clip_start = max(0, min(clip_start, anchor - goal_pre_frames))
        clip_end = min(frame_count, max(clip_end, anchor + goal_post_frames))
        max_seconds = MAX_GOAL_MOMENT_SECONDS
        clip_start, clip_end = cap_merged_range(clip_start, clip_end, anchor, fps, max_seconds)
        # Guarantee the canonical window survives capping.
        clip_start = min(clip_start, max(0, anchor - goal_pre_frames))
        clip_end = max(clip_end, min(frame_count, anchor + goal_post_frames))
    else:
        max_seconds = MAX_STANDARD_MOMENT_SECONDS
        clip_start, clip_end = cap_merged_range(
            clip_start,
            clip_end,
            int(winner["frame_number"]),
            fps,
            max_seconds,
        )

    merged["clip_start_frame"] = clip_start
    merged["clip_end_frame"] = clip_end
    merged["merged_event_count"] = len(cluster)
    merged["merged_event_types"] = sorted({event["type"] for event in cluster}, key=lambda kind: -EVENT_PRIORITY.get(kind, 0))
    merged["description"] = merge_description_text(cluster) or winner.get("description", "")
    return merged


def merge_overlapping_events(events: list[dict[str, Any]], video: dict[str, Any]) -> list[dict[str, Any]]:
    """Build unique broadcast moments so the same play appears once."""
    try:
        fps = float(video["fps"])
        frame_count = int(video["frame_count"])
        prepared = []
        for event in events:
            ranges = clip_ranges(event, fps, frame_count)
            prepared.append({**event, "_range": ranges})
        prepared.sort(key=lambda event: event["_range"]["clip_start"])

        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_range: dict[str, int] | None = None

        for event in prepared:
            event_range = event["_range"]
            if not current:
                current = [event]
                current_range = dict(event_range)
                continue

            assert current_range is not None
            goal_in_cluster = any(item["type"] == "goal" for item in current)
            near_goal_aftermath = goal_in_cluster and (
                event["frame_number"] - min(item["frame_number"] for item in current)
                <= int(round(24.0 * fps))
            )
            overlaps_existing_event = any(
                ranges_overlap_significantly(item["_range"], event_range, fps)
                for item in current
            )

            if overlaps_existing_event or near_goal_aftermath:
                current.append(event)
                current_range["clip_start"] = min(current_range["clip_start"], event_range["clip_start"])
                current_range["clip_end"] = max(current_range["clip_end"], event_range["clip_end"])
            else:
                clusters.append(current)
                current = [event]
                current_range = dict(event_range)

        if current:
            clusters.append(current)

        merged = [merge_event_cluster(cluster, video) for cluster in clusters]
        merged.sort(key=lambda event: event["clip_start_frame"])
        removed = len(events) - len(merged)
        if removed > 0:
            logging.info("Merged %d overlapping duplicate detections into unique moments.", removed)
        return merged
    except Exception as exc:
        raise HighlightGenerationError(f"Could not merge overlapping events: {exc}") from exc


def estimate_event_duration(event: dict[str, Any], fps: float, frame_count: int) -> float:
    ranges = clip_ranges(event, fps, frame_count)
    source_duration = (ranges["clip_end"] - ranges["clip_start"]) / fps
    if event["type"] == "goal":
        return source_duration + (source_duration / SLOWMO_SPEED)
    return source_duration


def select_events_for_duration(events: list[dict[str, Any]], video: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        fps = float(video["fps"])
        frame_count = int(video["frame_count"])

        def overlaps_selected(candidate: dict[str, Any], chosen: list[dict[str, Any]]) -> bool:
            candidate_range = clip_ranges(candidate, fps, frame_count)
            return any(
                ranges_overlap_significantly(candidate_range, clip_ranges(item, fps, frame_count), fps)
                for item in chosen
            )

        enriched = []
        for event in events:
            item = {**event}
            item["score"] = event_score(event)
            try:
                item["estimated_duration"] = estimate_event_duration(item, fps, frame_count)
                enriched.append(item)
            except Exception as exc:
                logging.warning("Skipping unselectable event %s: %s", event.get("index"), exc)

        enriched.sort(key=lambda event: event["score"], reverse=True)
        selected: list[dict[str, Any]] = []
        total = 0.0
        type_counts: dict[str, int] = {}

        for event in sorted(
            (item for item in enriched if item["type"] in CRITICAL_EVENT_TYPES),
            key=lambda item: (item["frame_number"], -item["score"]),
        ):
            if event in selected:
                continue
            if overlaps_selected(event, selected):
                selected = [
                    item
                    for item in selected
                    if not ranges_overlap_significantly(
                        clip_ranges(event, fps, frame_count),
                        clip_ranges(item, fps, frame_count),
                        fps,
                    )
                    or item["type"] in CRITICAL_EVENT_TYPES
                ]
                total = sum(item["estimated_duration"] for item in selected)
            selected.append(event)
            total += event["estimated_duration"]
            type_counts[event["type"]] = type_counts.get(event["type"], 0) + 1

        segment_count = timeline_segment_count(video)
        min_segments = max(1, int(math.ceil(segment_count * MIN_SELECTED_SEGMENT_RATIO)))
        segments_seen: set[int] = set()
        by_segment: dict[int, list[dict[str, Any]]] = {}
        for event in enriched:
            by_segment.setdefault(event_segment(event, video), []).append(event)

        for segment in sorted(by_segment):
            if len(segments_seen) >= min_segments and total >= TARGET_MIN_SECONDS * 0.5:
                break
            for event in sorted(by_segment[segment], key=lambda item: item["score"], reverse=True):
                if event in selected or overlaps_selected(event, selected):
                    continue
                if total + event["estimated_duration"] > TARGET_MAX_SECONDS and total >= TARGET_MIN_SECONDS:
                    continue
                selected.append(event)
                total += event["estimated_duration"]
                type_counts[event["type"]] = type_counts.get(event["type"], 0) + 1
                segments_seen.add(segment)
                break

        for event in enriched:
            event_type = event["type"]
            if event in selected:
                continue
            if overlaps_selected(event, selected):
                continue
            if event_type in {"crowd_reaction", "celebration"} and type_counts.get(event_type, 0) >= 4:
                continue
            if total >= TARGET_PREFERRED_SECONDS and event["score"] < 80:
                continue
            if total + event["estimated_duration"] > TARGET_MAX_SECONDS and total >= TARGET_MIN_SECONDS:
                continue
            selected.append(event)
            total += event["estimated_duration"]
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            if total >= TARGET_PREFERRED_SECONDS:
                break

        if total < TARGET_MIN_SECONDS:
            for event in enriched:
                if event in selected:
                    continue
                if overlaps_selected(event, selected):
                    continue
                if total + event["estimated_duration"] > TARGET_MAX_SECONDS:
                    continue
                selected.append(event)
                total += event["estimated_duration"]
                if total >= TARGET_MIN_SECONDS:
                    break

        selected.sort(key=lambda event: event["frame_number"])
        selected_segments = sorted({event_segment(event, video) for event in selected})
        logging.info(
            "Selected %d events across %d/%d timeline segments for %.1f minutes of estimated highlights.",
            len(selected),
            len(selected_segments),
            segment_count,
            total / 60,
        )
        return selected
    except Exception as exc:
        raise HighlightGenerationError(f"Could not select highlight events: {exc}") from exc


def total_estimated_duration(events: list[dict[str, Any]], video: dict[str, Any]) -> float:
    fps = float(video["fps"])
    frame_count = int(video["frame_count"])
    return sum(estimate_event_duration(event, fps, frame_count) for event in events)


def expand_selected_events_to_target(events: list[dict[str, Any]], video: dict[str, Any]) -> list[dict[str, Any]]:
    """Widen selected clips to reach target duration without duplicating moments."""
    try:
        if not events:
            return []

        fps = float(video["fps"])
        frame_count = int(video["frame_count"])
        expanded = [dict(event) for event in sorted(events, key=lambda item: item["frame_number"])]
        current_total = total_estimated_duration(expanded, video)
        target_total = min(TARGET_MAX_SECONDS, TARGET_PREFERRED_SECONDS)
        if current_total >= TARGET_MIN_SECONDS:
            return expanded

        deficit_seconds = target_total - current_total
        if deficit_seconds <= 0:
            return expanded

        for pass_index in range(3):
            if deficit_seconds <= 1.0:
                break
            changed = False
            for index, event in enumerate(expanded):
                if deficit_seconds <= 1.0:
                    break

                ranges = clip_ranges(event, fps, frame_count)
                left_bound = (
                    0
                    if index == 0
                    else int((expanded[index - 1]["frame_number"] + event["frame_number"]) / 2)
                )
                right_bound = (
                    frame_count
                    if index == len(expanded) - 1
                    else int((event["frame_number"] + expanded[index + 1]["frame_number"]) / 2)
                )
                max_seconds = (
                    EXPANDED_GOAL_MOMENT_SECONDS
                    if event["type"] == "goal"
                    else EXPANDED_STANDARD_MOMENT_SECONDS
                )
                current_source_frames = ranges["clip_end"] - ranges["clip_start"]
                max_source_frames = int(round(max_seconds * fps))
                capacity_frames = max(0, max_source_frames - current_source_frames)
                available_left = max(0, ranges["clip_start"] - left_bound)
                available_right = max(0, right_bound - ranges["clip_end"])
                available_frames = min(capacity_frames, available_left + available_right)
                if available_frames <= 0:
                    continue

                remaining_events = max(1, len(expanded) - index)
                desired_frames = int(round((deficit_seconds / remaining_events) * fps))
                grow_frames = max(1, min(desired_frames, available_frames))
                left_add = min(available_left, grow_frames // 2)
                right_add = min(available_right, grow_frames - left_add)
                if right_add < grow_frames - left_add and available_left > left_add:
                    left_add += min(available_left - left_add, grow_frames - left_add - right_add)

                if left_add <= 0 and right_add <= 0:
                    continue

                # Goals keep their canonical 8s/2s replay window untouched; do
                # not let duration-expansion logic push the goal frame off-center.
                if event["type"] == "goal":
                    continue

                event["clip_start_frame"] = ranges["clip_start"] - left_add
                event["clip_end_frame"] = ranges["clip_end"] + right_add
                gained = (left_add + right_add) / fps
                deficit_seconds -= gained
                changed = True

            if not changed:
                break

        expanded_total = total_estimated_duration(expanded, video)
        logging.info(
            "Expanded selected moment context from %.1f to %.1f minutes.",
            current_total / 60,
            expanded_total / 60,
        )
        return expanded
    except Exception as exc:
        raise HighlightGenerationError(f"Could not expand selected event context: {exc}") from exc


def extract_clips_parallel(events: list[dict[str, Any]], video: dict[str, Any], has_audio: bool) -> list[str]:
    try:
        max_workers = max(1, int(os.getenv("HIGHLIGHT_MAX_WORKERS", str(MAX_WORKERS))))
        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(extract_clip, event, video, has_audio): event
                for event in events
            }
            for future in as_completed(future_map):
                event = future_map[future]
                try:
                    results[event["index"]] = future.result()
                except Exception as exc:
                    logging.error("Skipping event %s: %s", event.get("index"), exc)
        return [results[event["index"]] for event in events if event["index"] in results]
    except Exception as exc:
        raise HighlightGenerationError(f"Parallel clip extraction failed: {exc}") from exc


def generate_highlights() -> None:
    try:
        load_dotenv()
        ensure_inputs()
        video, events = load_refined_events()
        validate_event_pool_coverage(events, video)
        unique_events = merge_overlapping_events(events, video)
        selected_events = expand_selected_events_to_target(
            select_events_for_duration(unique_events, video),
            video,
        )
        has_audio = probe_has_audio()
        prepare_clips_directory()

        logging.info("Extracting clips with parallel workers...")
        clip_paths = extract_clips_parallel(selected_events, video, has_audio)

        if len(clip_paths) < MIN_SUCCESSFUL_CLIPS:
            raise HighlightGenerationError(
                f"Only {len(clip_paths)} clips extracted successfully; need at least {MIN_SUCCESSFUL_CLIPS}."
            )

        logging.info("Merging final video...")
        merge_clips(build_concat_sequence(clip_paths, has_audio))
        verify_output()
        write_included_events(selected_events)
        logging.info("Done.")
    except HighlightGenerationError:
        raise
    except Exception as exc:
        raise HighlightGenerationError(f"Highlight generation failed: {exc}") from exc
    finally:
        cleanup()


def main() -> int:
    try:
        configure_logging()
        generate_highlights()
        return 0
    except HighlightGenerationError as exc:
        logging.error("Error: %s", exc)
        return 1
    except Exception as exc:
        logging.error("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())