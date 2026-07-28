"""
upload_and_notify.py
Upload highlights.mp4 to Google Drive and send Gmail notifications."""

from __future__ import annotations

import json
import os
import smtplib
import subprocess
import sys
import time
from email.message import EmailMessage
from typing import Any

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


OUTPUT_VIDEO = os.path.join(".", "highlights.mp4")
TIMESTAMPS_JSON = os.path.join(".", "timestamps.json")
INCLUDED_MOMENTS_JSON = os.path.join(".", "included_moments.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]


class NotifyError(Exception):
    """Raised when upload or notification fails."""


def require_env(name: str) -> str:
    try:
        value = os.getenv(name)
        if not value:
            raise NotifyError(f"Missing required environment variable: {name}")
        return value
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Could not read environment variable {name}: {exc}") from exc


def read_json(path: str) -> dict[str, Any]:
    try:
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8-sig") as file_obj:
            payload = json.load(file_obj)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        raise NotifyError(f"Could not read {path}: {exc}") from exc


def get_video_duration_minutes() -> float:
    try:
        if not os.path.isfile(OUTPUT_VIDEO):
            return 0.0
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                OUTPUT_VIDEO,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return round(float(result.stdout.strip()) / 60, 1)
    except Exception as exc:
        raise NotifyError(f"Could not read output video info: {exc}") from exc



def build_oauth_credentials() -> Credentials:
    try:
        client_secret_path = require_env("GOOGLE_OAUTH_CLIENT_SECRET_JSON")
        token_path = os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "token.json")

        credentials = None
        if os.path.isfile(token_path):
            credentials = Credentials.from_authorized_user_file(token_path, SCOPES)

        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        elif not credentials or not credentials.valid:
            if not os.path.isfile(client_secret_path):
                raise NotifyError(
                    f"Missing OAuth client secret file: {client_secret_path}. "
                    "Download a Desktop OAuth client JSON from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            print("Opening browser for Google Drive permission...")
            credentials = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(credentials.to_json())
        return credentials
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Could not create Google OAuth credentials: {exc}") from exc



def build_drive_service():
    try:
        credentials = build_oauth_credentials()
        return build("drive", "v3", credentials=credentials)
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Could not create Google Drive service: {exc}") from exc


def validate_drive_folder(service: Any, folder_id: str) -> None:
    try:
        service.files().get(
            fileId=folder_id,
            fields="id, name, mimeType",
            supportsAllDrives=True,
        ).execute()
    except Exception as exc:
        raise NotifyError(
            "Google Drive folder was not found or the selected Google Drive credentials cannot access it. "
            "Check GDRIVE_OUTPUT_FOLDER_ID. For OAuth, log in with the Google account "
            "that owns or can edit that folder. Original error: "
            f"{exc}"
        ) from exc


def upload_to_drive() -> str:
    try:
        if not os.path.isfile(OUTPUT_VIDEO):
            raise NotifyError(f"Missing output video: {OUTPUT_VIDEO}")

        folder_id = require_env("GDRIVE_OUTPUT_FOLDER_ID")
        service = build_drive_service()
        validate_drive_folder(service, folder_id)
        media = MediaFileUpload(OUTPUT_VIDEO, mimetype="video/mp4", resumable=True)
        metadata = {"name": "highlights.mp4", "parents": [folder_id]}
        uploaded = (
            service.files()
            .create(
                body=metadata,
                media_body=media,
                fields="id, webViewLink",
                supportsAllDrives=True,
            )
            .execute()
        )
        file_id = uploaded["id"]
        service.permissions().create(
            fileId=file_id,
            body={"role": "reader", "type": "anyone"},
            supportsAllDrives=True,
        ).execute()
        return f"https://drive.google.com/file/d/{file_id}/view"
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Google Drive upload failed: {exc}") from exc


def upload_to_drive_with_retry() -> str:
    try:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                print(f"Drive upload attempt {attempt}/3")
                return upload_to_drive()
            except Exception as exc:
                last_error = exc
                print(f"Drive upload attempt {attempt} failed: {exc}")
                if attempt < 3:
                    time.sleep(10)
        raise NotifyError(f"Drive upload failed after 3 attempts: {last_error}")
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Drive retry logic failed: {exc}") from exc


def moments_summary() -> str:
    try:
        payload = read_json(INCLUDED_MOMENTS_JSON) or read_json(TIMESTAMPS_JSON)
        moments = payload.get("events") or payload.get("moments", [])
        if not isinstance(moments, list) or not moments:
            return "No moment list was available."
        lines = []
        for moment in moments:
            timestamp = moment.get("exact_timestamp", moment.get("start_time", moment.get("rough_timestamp", "?")))
            lines.append(
                f"- {moment.get('type', 'moment')} at {timestamp}s: "
                f"{moment.get('description', '')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        raise NotifyError(f"Could not build moments summary: {exc}") from exc


def send_email(subject: str, body: str) -> None:
    try:
        sender = require_env("GMAIL_SENDER")
        password = require_env("GMAIL_APP_PASSWORD")
        recipient = require_env("NOTIFY_EMAIL")

        message = EmailMessage()
        message["From"] = sender
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(message)
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Could not send Gmail notification: {exc}") from exc


def send_success_notification(original_filename: str, drive_link: str) -> None:
    try:
        duration_minutes = get_video_duration_minutes()
        subject = f" Highlights Ready ” {original_filename} ” {duration_minutes} mins"
        body = (
            f"Your highlight video is ready.\n\n"
            f"Original file: {original_filename}\n"
            f"Duration: {duration_minutes} minutes\n"
            f"Drive link: {drive_link}\n\n"
            f"Moments included:\n{moments_summary()}\n"
        )
        send_email(subject, body)
    except Exception as exc:
        raise NotifyError(f"Could not send success notification: {exc}") from exc


def send_failure_notification(original_filename: str, step: str, error_message: str) -> None:
    try:
        subject = f" Highlights Failed ” {original_filename}"
        body = (
            f"The highlight pipeline failed.\n\n"
            f"Original file: {original_filename}\n"
            f"Failed step: {step}\n\n"
            f"Error:\n{error_message}\n"
        )
        send_email(subject, body)
    except Exception as exc:
        raise NotifyError(f"Could not send failure notification: {exc}") from exc


def run_success(original_filename: str) -> None:
    try:
        drive_link = upload_to_drive_with_retry()
        send_success_notification(original_filename, drive_link)
        print(f"Uploaded highlights: {drive_link}")
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Success notification flow failed: {exc}") from exc


def main() -> int:
    try:
        load_dotenv()
        if len(sys.argv) < 3:
            raise NotifyError("Usage: upload_and_notify.py success <filename> OR failure <filename> <step> <error>")

        mode = sys.argv[1]
        original_filename = sys.argv[2]
        if mode == "success":
            run_success(original_filename)
        elif mode == "failure":
            if len(sys.argv) < 5:
                raise NotifyError("Failure mode requires filename, step, and error message.")
            send_failure_notification(original_filename, sys.argv[3], sys.argv[4])
        else:
            raise NotifyError(f"Unknown mode: {mode}")
        return 0
    except NotifyError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())



