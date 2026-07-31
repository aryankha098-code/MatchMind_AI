"""
upload_and_notify.py
Upload highlights.mp4 to Google Drive and YouTube, and send Gmail
notifications -- all using ONE OAuth client (oauth_client_secret.json) and
ONE cached token (token.json). No SMTP, no Gmail app password.

Why this version is different:
  - OLD: Drive used OAuth, but email used smtplib + GMAIL_SENDER +
    GMAIL_APP_PASSWORD (a second, separate credential the client had to
    generate by hand in their Google Account security settings), and there
    was no YouTube upload at all despite the README describing one.
  - NEW: Drive, YouTube, and Gmail all authenticate with the SAME
    credentials object built from oauth_client_secret.json. Email is sent
    via the Gmail API (users.messages.send) as the account that completed
    the OAuth consent -- so there is nothing beyond oauth_client_secret.json
    for the client to obtain manually.

IMPORTANT: the OAuth scopes below are wider than the old Drive-only version
(drive + youtube.upload + gmail.send). If token.json was created by an
older version of this script, it will NOT carry the new scopes. Delete
token.json once so the next run asks for consent again -- this code
detects an insufficient-scope failure and tells you to do exactly that
rather than failing with a confusing Google API error.
"""

from __future__ import annotations

import base64
import json
import os
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
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload


OUTPUT_VIDEO = os.path.join(".", "highlights.mp4")
TIMESTAMPS_JSON = os.path.join(".", "timestamps.json")
INCLUDED_MOMENTS_JSON = os.path.join(".", "included_moments.json")
UPLOAD_RESULTS_JSON = os.path.join(".", "upload_results.json")

# One combined scope set for Drive + YouTube + Gmail -- one consent, one
# token.json. If you ever add/remove a scope here, delete token.json once.
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/gmail.send",
]

DEFAULT_YOUTUBE_PRIVACY_STATUS = "unlisted"
DEFAULT_YOUTUBE_UPLOAD_TIMEOUT_SECONDS = 1800
DEFAULT_YOUTUBE_UPLOAD_CHUNK_MB = 8


class NotifyError(Exception):
    """Raised when upload or notification fails."""


# ══════════════════════════════════════════════════════════════════════════
# Env helpers
# ══════════════════════════════════════════════════════════════════════════

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


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise NotifyError(f"{name} must be an integer.")


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
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                OUTPUT_VIDEO,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return round(float(result.stdout.strip()) / 60, 1)
    except Exception as exc:
        raise NotifyError(f"Could not read output video info: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════
# OAuth (shared across Drive, YouTube, Gmail)
# ══════════════════════════════════════════════════════════════════════════

def build_oauth_credentials() -> Credentials:
    """
    Builds/loads a single Credentials object good for Drive + YouTube +
    Gmail. On a brand-new machine (no token.json yet) this opens a
    local-server OAuth flow ONCE; the client logs into their Google account,
    clicks Allow, and token.json is cached from then on -- no manual file
    to fetch beyond oauth_client_secret.json, which you (the developer)
    provide as part of the deployment, not something the client generates.
    """
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
                    "This file is provided by the developer as part of the "
                    "Docker image/deployment -- it is not something the "
                    "end user creates."
                )
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            print(
                "First-time setup: open the URL below (or the browser window "
                "that just opened) and sign in with the Google account you "
                "want Drive uploads, YouTube uploads, and email notifications "
                "to use. This only happens once."
            )
            credentials = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(credentials.to_json())
        return credentials
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Could not create Google OAuth credentials: {exc}") from exc


def build_service(name: str, version: str):
    try:
        credentials = build_oauth_credentials()
        return build(name, version, credentials=credentials)
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Could not create Google {name} service: {exc}") from exc


def _is_insufficient_scope_error(exc: Exception) -> bool:
    text = str(exc).upper()
    return "INSUFFICIENT" in text and "SCOPE" in text


def _reraise_with_scope_hint(exc: Exception, token_path: str) -> None:
    if _is_insufficient_scope_error(exc):
        raise NotifyError(
            "Google rejected this request because the cached token.json "
            f"({token_path}) does not have the scopes this version of the "
            "script needs. Delete token.json and run the pipeline again -- "
            "it will prompt for a fresh one-time consent covering Drive, "
            "YouTube, and Gmail together."
        ) from exc
    raise exc


# ══════════════════════════════════════════════════════════════════════════
# Google Drive upload
# ══════════════════════════════════════════════════════════════════════════

def validate_drive_folder(service: Any, folder_id: str) -> None:
    try:
        service.files().get(
            fileId=folder_id,
            fields="id, name, mimeType",
            supportsAllDrives=True,
        ).execute()
    except Exception as exc:
        raise NotifyError(
            "Google Drive folder was not found or the OAuth account cannot "
            "access it. Check GDRIVE_OUTPUT_FOLDER_ID. Original error: "
            f"{exc}"
        ) from exc


def upload_to_drive() -> str:
    try:
        if not os.path.isfile(OUTPUT_VIDEO):
            raise NotifyError(f"Missing output video: {OUTPUT_VIDEO}")

        folder_id = require_env("GDRIVE_OUTPUT_FOLDER_ID")
        service = build_service("drive", "v3")
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
    except HttpError as exc:
        _reraise_with_scope_hint(exc, os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "token.json"))
        raise NotifyError(f"Google Drive upload failed: {exc}") from exc
    except Exception as exc:
        raise NotifyError(f"Google Drive upload failed: {exc}") from exc


def upload_to_drive_with_retry() -> str | None:
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
    print(f"Drive upload failed after 3 attempts: {last_error}")
    return None


# ══════════════════════════════════════════════════════════════════════════
# YouTube upload
# ══════════════════════════════════════════════════════════════════════════

def _youtube_metadata(original_filename: str) -> dict[str, Any]:
    title = f"Match Highlights - {original_filename}"[:100]
    return {
        "snippet": {
            "title": title,
            "description": "Auto-generated highlight reel.",
            "categoryId": "17",  # Sports
        },
        "status": {
            "privacyStatus": os.getenv(
                "YOUTUBE_PRIVACY_STATUS", DEFAULT_YOUTUBE_PRIVACY_STATUS
            ),
            "selfDeclaredMadeForKids": False,
        },
    }


def _upload_video_resumable(service: Any, body: dict[str, Any], file_path: str) -> dict[str, Any]:
    chunk_mb = _get_int_env("YOUTUBE_UPLOAD_CHUNK_MB", DEFAULT_YOUTUBE_UPLOAD_CHUNK_MB)
    timeout_seconds = _get_int_env(
        "YOUTUBE_UPLOAD_TIMEOUT_SECONDS", DEFAULT_YOUTUBE_UPLOAD_TIMEOUT_SECONDS
    )
    media = MediaFileUpload(
        file_path,
        mimetype="video/mp4",
        chunksize=max(1, chunk_mb) * 1024 * 1024,
        resumable=True,
    )
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    deadline = time.monotonic() + timeout_seconds
    while response is None:
        if time.monotonic() > deadline:
            raise NotifyError(
                f"YouTube upload did not finish within {timeout_seconds}s "
                "(YOUTUBE_UPLOAD_TIMEOUT_SECONDS)."
            )
        status, response = request.next_chunk()
        if status:
            print(f"YouTube upload progress: {int(status.progress() * 100)}%")
    return response


def upload_to_youtube(original_filename: str) -> str:
    try:
        if not os.path.isfile(OUTPUT_VIDEO):
            raise NotifyError(f"Missing output video: {OUTPUT_VIDEO}")

        service = build_service("youtube", "v3")
        body = _youtube_metadata(original_filename)
        response = _upload_video_resumable(service, body, OUTPUT_VIDEO)
        video_id = response["id"]
        return f"https://youtu.be/{video_id}"
    except NotifyError:
        raise
    except HttpError as exc:
        _reraise_with_scope_hint(exc, os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "token.json"))
        raise NotifyError(f"YouTube upload failed: {exc}") from exc
    except Exception as exc:
        raise NotifyError(f"YouTube upload failed: {exc}") from exc


def upload_to_youtube_with_retry(original_filename: str) -> str | None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            print(f"YouTube upload attempt {attempt}/3")
            return upload_to_youtube(original_filename)
        except Exception as exc:
            last_error = exc
            print(f"YouTube upload attempt {attempt} failed: {exc}")
            if attempt < 3:
                time.sleep(10)
    print(f"YouTube upload failed after 3 attempts: {last_error}")
    return None


# ══════════════════════════════════════════════════════════════════════════
# Gmail API notification (replaces smtplib + GMAIL_APP_PASSWORD entirely)
# ══════════════════════════════════════════════════════════════════════════

def moments_summary() -> str:
    try:
        payload = read_json(INCLUDED_MOMENTS_JSON) or read_json(TIMESTAMPS_JSON)
        moments = payload.get("events") or payload.get("moments", [])
        if not isinstance(moments, list) or not moments:
            return "No moment list was available."
        lines = []
        for moment in moments:
            timestamp = moment.get(
                "exact_timestamp", moment.get("start_time", moment.get("rough_timestamp", "?"))
            )
            lines.append(
                f"- {moment.get('type', 'moment')} at {timestamp}s: "
                f"{moment.get('description', '')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        raise NotifyError(f"Could not build moments summary: {exc}") from exc


def send_email(subject: str, body: str) -> None:
    """
    Sends via the Gmail API as the account that completed the OAuth
    consent (userId="me") -- no sender password, no SMTP server, and the
    "from" address is whichever Google account was authorized, so there is
    nothing extra to configure.
    """
    try:
        recipient = require_env("NOTIFY_EMAIL")
        service = build_service("gmail", "v1")

        message = EmailMessage()
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
    except NotifyError:
        raise
    except HttpError as exc:
        _reraise_with_scope_hint(exc, os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "token.json"))
        raise NotifyError(f"Could not send Gmail notification: {exc}") from exc
    except Exception as exc:
        raise NotifyError(f"Could not send Gmail notification: {exc}") from exc


def _format_link_lines(links: list[dict[str, str]]) -> str:
    if not links:
        return "No uploads succeeded."
    return "\n".join(f"- {item['label']}: {item['url']}" for item in links)


def send_success_notification(original_filename: str, links: list[dict[str, str]]) -> None:
    try:
        duration_minutes = get_video_duration_minutes()
        subject = f"Highlights Ready - {original_filename} - {duration_minutes} mins"
        body = (
            f"Your highlight video is ready.\n\n"
            f"Original file: {original_filename}\n"
            f"Duration: {duration_minutes} minutes\n\n"
            f"Links:\n{_format_link_lines(links)}\n\n"
            f"Moments included:\n{moments_summary()}\n"
        )
        send_email(subject, body)
    except Exception as exc:
        raise NotifyError(f"Could not send success notification: {exc}") from exc


def send_failure_notification(original_filename: str, step: str, error_message: str) -> None:
    try:
        subject = f"Highlights Failed - {original_filename}"
        body = (
            f"The highlight pipeline failed.\n\n"
            f"Original file: {original_filename}\n"
            f"Failed step: {step}\n\n"
            f"Error:\n{error_message}\n"
        )
        send_email(subject, body)
    except Exception as exc:
        raise NotifyError(f"Could not send failure notification: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════

def write_upload_results(links: list[dict[str, str]]) -> None:
    try:
        with open(UPLOAD_RESULTS_JSON, "w", encoding="utf-8") as file_obj:
            json.dump({"result_links": links}, file_obj, indent=2)
    except Exception as exc:
        print(f"Could not write {UPLOAD_RESULTS_JSON}: {exc}")


def run_success(original_filename: str) -> None:
    try:
        links: list[dict[str, str]] = []

        drive_link = upload_to_drive_with_retry()
        if drive_link:
            links.append({"label": "Highlights (Drive)", "url": drive_link})

        youtube_link = upload_to_youtube_with_retry(original_filename)
        if youtube_link:
            links.append({"label": "Highlights (YouTube)", "url": youtube_link})

        write_upload_results(links)

        if not links:
            raise NotifyError("Both Drive and YouTube uploads failed.")

        send_success_notification(original_filename, links)
        for item in links:
            print(f"Uploaded: {item['label']} -> {item['url']}")
    except NotifyError:
        raise
    except Exception as exc:
        raise NotifyError(f"Success notification flow failed: {exc}") from exc


def main() -> int:
    try:
        load_dotenv()
        if len(sys.argv) < 3:
            raise NotifyError(
                "Usage: upload_and_notify.py success <filename> OR "
                "failure <filename> <step> <error message>"
            )

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