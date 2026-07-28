"""
server.py
FastAPI entry point for the video highlight pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import traceback
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, UploadFile


app = FastAPI()

JOB_STATUS: dict[str, Any] = {
    "status": "idle",
    "step": None,
    "message": "Waiting for a video.",
    "original_filename": None,
}


def set_status(status: str, step: str | None, message: str, original_filename: str | None = None) -> None:
    try:
        JOB_STATUS["status"] = status
        JOB_STATUS["step"] = step
        JOB_STATUS["message"] = message
        if original_filename is not None:
            JOB_STATUS["original_filename"] = original_filename
    except Exception as exc:
        print(f"Could not update job status: {exc}")


def run_python_script(script_name: str, step: str, extra_args: list[str] | None = None) -> None:
    try:
        command = ["python", script_name]
        if extra_args:
            command.extend(extra_args)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{step} step failed with exit code {exc.returncode}") from exc
    except Exception as exc:
        raise RuntimeError(f"{step} step could not start: {exc}") from exc


def notify_failure(original_filename: str, step: str, error_message: str) -> None:
    try:
        run_python_script(
            "upload_and_notify.py",
            "failure notification",
            ["failure", original_filename, step, error_message],
        )
    except Exception as exc:
        print(f"Could not send failure notification: {exc}")


def run_generate_with_retry() -> None:
    try:
        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                print(f"Generate attempt {attempt}/2")
                run_python_script("generate_highlights.py", "generate")
                return
            except Exception as exc:
                last_error = exc
                print(f"Generate attempt {attempt} failed: {exc}")
        raise RuntimeError(f"generate step failed after retry: {last_error}")
    except Exception:
        raise


def run_pipeline(original_filename: str) -> None:
    try:
        print("Starting pipeline...")

        set_status("processing", "analyze", "Analyzing video with Gemini.", original_filename)
        run_python_script("analyze.py", "analyze")

        set_status("processing", "refine", "Refining event frames with OpenCV.", original_filename)
        run_python_script("event_refiner.py", "refine")

        set_status("processing", "generate", "Generating highlight reel.", original_filename)
        run_generate_with_retry()

        set_status("processing", "upload", "Uploading highlights and sending notification.", original_filename)
        run_python_script("upload_and_notify.py", "upload", ["success", original_filename])

        set_status("done", "complete", "Highlights generated, uploaded, and notification sent.", original_filename)
        print("Pipeline done!")
    except Exception as exc:
        error_message = f"{exc}\n{traceback.format_exc()}"
        failed_step = JOB_STATUS.get("step") or "unknown"
        set_status("error", str(failed_step), str(exc), original_filename)
        notify_failure(original_filename, str(failed_step), error_message)
        print(f"Pipeline failed during {failed_step}: {exc}")


@app.get("/status")
async def status() -> dict[str, Any]:
    try:
        return JOB_STATUS
    except Exception as exc:
        return {"status": "error", "step": "status", "message": str(exc)}


@app.post("/process")
async def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, str]:
    try:
        original_filename = file.filename or "uploaded-video.mp4"
        video_path = os.path.join(".", "match.mp4")
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("Video received")
        set_status("processing", "queued", "Video received. Pipeline queued.", original_filename)
        background_tasks.add_task(run_pipeline, original_filename)
        return {"status": "started"}
    except Exception as exc:
        set_status("error", "upload", f"Could not receive uploaded video: {exc}")
        return {"status": "error"}
