# AI Football Video Highlights

FastAPI service that accepts a football match video, analyzes it with Gemini, generates a TV-style highlight reel with FFmpeg, uploads the result to Google Drive, and emails a notification.

## Flow

1. `POST /process` receives a video from n8n and saves it as `match.mp4`.
2. `server.py` starts the pipeline in a background task and returns `{"status": "started"}` immediately.
3. `analyze.py` uploads the video to Gemini and writes coarse candidate events to `timestamps.json`.
4. `event_refiner.py` scans rough timestamp windows frame-by-frame with OpenCV and writes `refined_events.json`.
5. `generate_highlights.py` creates frame-aligned `highlights.mp4` and `included_moments.json`.
6. `upload_and_notify.py` uploads the video to Google Drive and sends success or failure email.

## API

Start a job:

```text
POST /process
```

Poll job status:

```text
GET /status
```

Statuses are `idle`, `processing`, `done`, or `error`.

## Environment

```text
GEMINI_API_KEY=
OPENAI_API_KEY=
GOOGLE_SERVICE_ACCOUNT_JSON=service_account.json
GDRIVE_OUTPUT_FOLDER_ID=
GMAIL_SENDER=
GMAIL_APP_PASSWORD=
NOTIFY_EMAIL=
```

Do not include `.env` in Docker images or commits.

## Output

The generator targets broadcast-friendly output:

- `1920x1080`
- `30fps`
- H.264 video at `4000k`
- AAC audio at `192k`
- black clip fades
- short broadcast-style transitions
- lower-third event labels
- event scoring for goals, assists, near misses, saves, dangerous attacks, dribbles, tackles, celebrations, and crowd reactions
- target duration selection for roughly 12-15 minutes when enough quality events exist
- duplicate-event merging so one play appears only once
- goal and save premium effects: frame-aligned `0.5x` slow motion plus smooth action-centered zoom
- lighter zoom-only effects for near misses, dribbles, tackles, and celebrations
- goal effects start at the refined event frame and target about 9 seconds of slow-motion output
- goal clips include build-up, shot, aftermath, and celebration context when nearby detections overlap

## Faster Analysis

`analyze.py` keeps `match.mp4` as the source, but uploads a small temporary proxy to Gemini by default:

```text
ANALYZE_USE_PROXY=true
ANALYZE_PROXY_WIDTH=854
ANALYZE_PROXY_FPS=4
ANALYZE_PROXY_CRF=28
ANALYZE_CHUNKED=true
ANALYZE_CHUNK_SECONDS=600
ANALYZE_CHUNK_OVERLAP_SECONDS=5
```

The proxy is split into 10-minute analysis chunks before upload. Each chunk is analyzed separately, then timestamps are offset back into the full-match timeline. This prevents long videos from producing events only from the opening minutes.

`event_refiner.py` is optimized to run expensive OpenCV frame scanning only for the event types that need precise effect timing:

```text
REFINE_HIGH_PRECISION_TYPES=goal,penalty,save,near_miss,shot_on_target
```

Other events keep Gemini/OpenAI's rough timestamp converted to a frame number, which makes the backend faster while preserving accuracy for goals, saves, and near misses.

The analyzer and generator reject event timelines that only cover the opening part of the video. For full-match coverage, events are checked in 10-minute segments:

```text
COVERAGE_SEGMENT_SECONDS=600
```

If Gemini returns a temporary high-demand error such as `503 UNAVAILABLE`, `analyze.py` retries automatically with backoff. You can also configure fallback models:

```text
GEMINI_MODELS=gemini-3.5-flash,gemini-2.5-flash
```

If Gemini still fails and `OPENAI_API_KEY` is set, `analyze.py` falls back to OpenAI vision analysis by sampling timestamped storyboard frames:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
OPENAI_FALLBACK_MAX_FRAMES=120
OPENAI_FALLBACK_FRAME_WIDTH=512
```

The generator first merges overlapping detections into unique broadcast moments, selects strong moments from across the match timeline, fills the remaining duration by score, then restores chronological order.

## Local Run

```powershell
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

FFmpeg and FFprobe must be installed and available on `PATH`.
