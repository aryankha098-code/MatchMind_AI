# AI Football Video Highlights

A backend pipeline that takes a raw football match video, analyzes it with Gemini (Gemma as fallback), generates a TV-style highlight reel with FFmpeg (slow-motion + zoom on goals), uploads the highlights to Google Drive and YouTube, and emails the result links.

It's a pure API — there is no web upload page. It's meant to be driven by a script, `curl`/Postman, or an automation tool such as n8n.

## How It Works

| Stage | Script | Reads | Writes |
|---|---|---|---|
| API server | `server.py` | uploaded file | `match.mp4`, in-memory job status |
| 1. Analyze | `analyze.py` | `match.mp4` | `timestamps.json` |
| 2. Refine | `event_refiner.py` | `match.mp4`, `timestamps.json` | `refined_events.json` |
| 3. Generate | `generate_highlights.py` | `match.mp4`, `refined_events.json` | `highlights.mp4`, `included_moments.json` |
| 4. Upload & notify | `upload_and_notify.py` | `highlights.mp4`, `included_moments.json` | `upload_results.json`, an email |

1. `POST /process` with the video file attached — `server.py` saves it as `match.mp4` and immediately returns `{"status": "started"}`, then runs the four stages below in the background.
2. `analyze.py` splits the match into overlapping chunks, burns a running clock into each one, and asks Gemini to find every highlight-worthy moment (goals get a dedicated, higher-accuracy detection pass). Writes `timestamps.json`.
3. `event_refiner.py` uses OpenCV (motion, ball tracking, scoreboard change, audio) to correct the important events (goals, penalties, saves, near-misses, shots on target) to an exact video frame. Writes `refined_events.json`.
4. `generate_highlights.py` selects the best ~12–15 minutes of events, cuts a clip for each with FFmpeg (slow-motion replay + tracking zoom for goals), and concatenates everything into `highlights.mp4`. Writes `included_moments.json`.
5. `upload_and_notify.py` uploads `highlights.mp4` to Google Drive and YouTube, then emails the links via the Gmail API. Writes `upload_results.json`.
6. Poll `GET /status` at any time to see which stage is running, or whether the job finished (`done`) or failed (`error`).

> **Note:** only `highlights.mp4` is uploaded — the original full match (`match.mp4`) is not currently published anywhere by this pipeline.

## Local Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install FFmpeg and ffprobe, then confirm both are on `PATH`:

```bash
ffmpeg -version
ffprobe -version
```

Copy `env.example` to `.env` and fill in your keys (see [Configuration](#configuration) below):

```bash
cp env.example .env
```

Run the server:

```bash
uvicorn server:app --reload
```

Trigger a job (there is no browser UI — use a tool that can send a multipart POST):

```bash
curl -X POST http://localhost:8000/process -F "file=@match.mp4"
curl http://localhost:8000/status
```

## Google Cloud Setup

This project authenticates to **Drive, YouTube, and Gmail all through one OAuth client and one cached token** — there's no separate Gmail app password or SMTP config.

1. Create/select a project in [Google Cloud Console](https://console.cloud.google.com/).
2. Enable **YouTube Data API v3** and **Google Drive API**.
3. Configure the OAuth consent screen (External user type is fine for testing) and add the Google account you'll use as a **test user** while the app is in "Testing" mode.
4. Create an **OAuth Client ID** with application type **Desktop app**, and download the JSON.
5. Rename it to `oauth_client_secret.json` and place it in the project root.
6. Get a **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey).

On first run, `upload_and_notify.py` opens a one-time browser login covering all three scopes below and caches the result to `token.json` — you won't be asked again unless the scopes change.

```text
https://www.googleapis.com/auth/drive
https://www.googleapis.com/auth/youtube.upload
https://www.googleapis.com/auth/gmail.send
```

If `token.json` was created by an older version of this project (Drive-only, or before Gmail API sending was added), delete it and run the pipeline again so it can re-consent with all three scopes at once.

## Google Drive Setup

Create or choose a Drive folder for the finished highlight uploads. Copy the folder ID from its URL:

```text
https://drive.google.com/drive/folders/<this-part-is-the-folder-id>
```

```text
GDRIVE_OUTPUT_FOLDER_ID=your_folder_id
```

Uploaded files are set to "anyone with the link can view."

## YouTube Quota

The YouTube Data API's default daily quota is 10,000 units; a video upload costs roughly 1,600 units. Since this pipeline uploads **one** video per run (`highlights.mp4` only), that's about **6 full pipeline runs per day** on the default quota. Drive uploads don't use YouTube quota.

## Configuration

All settings are environment variables loaded from `.env` (see `env.example` for the full, current list). The most important ones:

```text
GEMINI_API_KEY=                                   # required (or GEMMA_API_KEY as fallback)
GEMMA_API_KEY=

GOOGLE_OAUTH_CLIENT_SECRET_JSON=oauth_client_secret.json
GOOGLE_OAUTH_TOKEN_JSON=token.json

GDRIVE_OUTPUT_FOLDER_ID=                          # required
NOTIFY_EMAIL=                                     # required

YOUTUBE_PRIVACY_STATUS=unlisted
YOUTUBE_UPLOAD_TIMEOUT_SECONDS=1800
YOUTUBE_UPLOAD_CHUNK_MB=8
```

Video-analysis tuning (see `analyze.py`'s module docstring for the full rationale behind these defaults):

```text
ANALYZE_CHUNK_SECONDS=480          # length of each chunk analyzed per Gemini call (8 min)
ANALYZE_CHUNK_OVERLAP_SECONDS=30   # overlap between consecutive chunks
ANALYZE_CHUNK_PROXY_WIDTH=1536     # Gemini's real tiling ceiling — no benefit going higher
ANALYZE_CHUNK_PROXY_FPS=15         # encoded fps, for a smooth burned-in clock
ANALYZE_CHUNK_PROXY_CRF=18
ANALYZE_BURN_TIMECODE=true         # burns a running clock into each chunk so the model reads
                                    # the exact second instead of estimating it
ANALYZE_KEEP_AUDIO=true            # crowd roar / commentary is a strong goal signal
ANALYZE_GOAL_SAMPLE_FPS=5.0        # higher sample rate for the dedicated goal-detection pass
ANALYZE_GENERAL_SAMPLE_FPS=1.0
GEMINI_MODELS=gemini-3.6-flash,gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash,gemini-2.0-flash,gemini-1.5-flash
GEMINI_GOAL_MODEL=gemini-3.1-pro-preview   # tried first for goals only; requires billing enabled,
                                            # falls back to GEMINI_MODELS automatically if unavailable
```

Frame-refinement and highlight-generation tuning:

```text
REFINE_HIGH_PRECISION_TYPES=goal,penalty,save,near_miss,shot_on_target
GOAL_SLOWMO_OUTPUT_SECONDS=9       # clamped to 7-10
HIGHLIGHT_MAX_WORKERS=2            # parallel FFmpeg workers extracting clips
```

## API

Start a job:

```text
POST /process
```

Multipart form data, file field name:

```text
file
```

Poll job status:

```text
GET /status
```

```json
{
  "status": "processing",
  "step": "generate",
  "message": "Generating highlight reel.",
  "original_filename": "match.mp4"
}
```

`status` is one of `idle`, `processing`, `done`, `error`. `step` is one of `queued`, `analyze`, `refine`, `generate`, `upload`, `complete`.

Result links (once uploaded) are written to `upload_results.json` and included in the notification email:

```json
{
  "result_links": [
    { "label": "Highlights (Drive)", "url": "https://drive.google.com/file/d/.../view" },
    { "label": "Highlights (YouTube)", "url": "https://youtu.be/..." }
  ]
}
```

## Docker

Build:

```bash
docker build -t football-highlights-api .
```

Run, with credentials and env mounted in:

```bash
docker run --name football-highlights-api \
  --env-file .env \
  -p 8000:8000 \
  -v "$(pwd)/oauth_client_secret.json:/app/oauth_client_secret.json:ro" \
  -v "$(pwd)/token.json:/app/token.json" \
  football-highlights-api
```

(PowerShell: replace `\` line continuations with `` ` `` and `$(pwd)` with `${PWD}`.)

## Output Files

Runtime files are created in the app's working directory as the pipeline runs:

```text
match.mp4
timestamps.json
refined_events.json
included_moments.json
highlights.mp4
upload_results.json
```

## Security

**Never commit `.env`, `oauth_client_secret.json`, or `token.json`** — all three are already excluded via `.gitignore`/`.dockerignore`, and that should stay that way. If any of them is ever exposed (e.g. shared in a support ticket, chat, or public repo), treat it as compromised: regenerate the OAuth client in Google Cloud Console, issue a fresh `oauth_client_secret.json`, and rotate the Gemini API key.
