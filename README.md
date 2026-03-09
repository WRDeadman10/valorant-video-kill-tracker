# VALORANT Video Analyzer

Production-style Python project to extract structured match data from VALORANT gameplay videos.

This project supports:
- Single video analysis (`.mp4` input)
- Folder analysis (scans only that folder by default)
- Per-video kill extraction with timestamps
- Per-kill details (`site_name`, `weapon_used`, banner/score debug)

---

## 1. What The Project Extracts

For each video:
- `agent`
- `kills`
- `kill_events` (timestamp list)
- `kill_event_details`:
  - `site_name`
  - `weapon_used`
  - kill-feed/banner confidence fields

For folder mode:
- One fresh JSON generated for the input folder
- Top-level `agent_name`
- Per-video array elements with kill breakdown

---

## 2. Detection Pipeline (High Level)

1. Sample frames from video using FFmpeg.
2. Use fixed ROIs (regions of interest):
   - map name (top-left)
   - kill feed (top-right)
   - HUD agent icon (bottom-center)
3. Detect map via OCR on map ROI.
4. Detect agent using template matching on HUD icon ROI.
5. Detect kill events using kill-feed logic:
   - Match kill-feed row template (`agent_name_v2.png`)
   - OCR username from candidate row
   - Require non-red row background
   - Cluster repeated detections of the same feed entry
6. Export JSON.

---

## 3. Project Structure

```text
valorant-video-analyzer/
|
|-- main.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- analyzer/
|   |-- __init__.py
|   |-- video_loader.py
|   |-- frame_sampler.py
|   |-- roi_detector.py
|   |-- map_detector.py
|   |-- agent_detector.py
|   |-- kill_feed_detector.py
|   `-- json_exporter.py
|
|-- assets/
|   |-- templates/
|   |   |-- agent_name_v2.png
|   |   `-- ...
|   `-- agent_icons/
|       |-- *.webp
|       `-- ...
|
`-- output/
```

---

## 4. Requirements

- Python 3.11 recommended
- FFmpeg installed and available in `PATH`
- Tesseract OCR installed (or `TESSERACT_CMD` set)

Python dependencies:
- `opencv-python`
- `numpy`
- `pytesseract`
- `ffmpeg-python`
- `tqdm`

Install:

```bash
pip install -r requirements.txt
```

---

## 5. FFmpeg and Tesseract Setup

### Windows

Install FFmpeg:

```powershell
winget install Gyan.FFmpeg
```

Install Tesseract:

```powershell
winget install tesseract-ocr.tesseract
```

If Tesseract is not in `PATH`, set:

```powershell
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

The app also auto-checks this default location on Windows:

`C:\Program Files\Tesseract-OCR\tesseract.exe`

---

## 6. Asset Requirements

### 6.1 Kill-feed template

Required:
- `assets/templates/agent_name_v2.png`

Fallback:
- `assets/templates/agent_name.png`

Compatibility fallback path is also supported for typo folders:
- `assets/templets/agent_name_v2.png`

### 6.2 Agent icon templates

Primary path:
- `assets/agent_icons/*.webp`

Compatibility fallback paths:
- `assets/agents/agetns_icon/*`
- `assets/agents/agent_icons/*`

Note:
- Kill-feed left icon matching is best-effort.
- If confidence is low, agent remains `Unknown`.

---

## 7. Kill Detection Rules (Current)

A kill candidate is accepted only when all are true:

1. Kill-feed row template is detected in top-right ROI.
2. OCR username match score passes threshold (fuzzy).
3. Row background is non-red (red ratio below threshold).

Then detections are clustered by:
- temporal gap
- row vertical band consistency

Only cluster starts become final kill timestamps.

---

## 8. Running The Script

## 8.1 Single video mode

```bash
python main.py gameplay.mp4
```

Example with options:

```bash
python main.py gameplay.mp4 --player "Rangnar Lothbrok" --interval 0.5 --output output/result.json --log-level INFO
```

## 8.2 Folder mode (non-recursive by default)

Pass a folder path. Script scans only that folder for `.mp4` by default.

```bash
python main.py "E:\matches_root" --player "Rangnar Lothbrok"
```

Behavior:
- Only videos directly inside the given folder are processed.
- One fresh JSON is written in that same folder.
- Default filename is `<folder_name>.json`.

Override output filename:

```bash
python main.py "E:\matches_root" --output-name folder_result.json
```

Enable recursive scan when needed:

```bash
python main.py "E:\matches_root" --recursive
```

Debug OCR window (shows detected kill frame + OCR details) and pause on every kill:

```bash
python main.py "E:\matches_root" --debug-ocr --pause-on-kill --debug-dir output/debug_kill_events
```

---

## 9. Output Formats

## 9.1 Single video output

```json
{
  "video_name": "match.mp4",
  "agent_name": "Jett",
  "kills": 2,
  "kill_events": [
    {"timestamp": 15.9},
    {"timestamp": 33.5}
  ],
  "kill_event_details": [
    {
      "timestamp": 15.9,
      "site_name": "A",
      "weapon_used": "Phantom"
    }
  ]
}
```

## 9.2 Folder output

```json
{
  "folder": "E:\\matches_root\\BatchA",
  "agent_name": "Jett",
  "videos": [
    {
      "video_name": "A.mp4",
      "kills": 1,
      "kill_events": [{"timestamp": 15.9}],
      "kill_event_details": [
        {
          "timestamp": 15.9,
          "site_name": "A",
          "weapon_used": "Phantom"
        }
      ]
    },
    {
      "video_name": "B.mp4",
      "kills": 2,
      "kill_events": [{"timestamp": 18.4}, {"timestamp": 29.6}],
      "kill_event_details": []
    }
  ]
}
```

---

## 10. Configuration Reference (`config.py`)

Key knobs:
- `SAMPLE_INTERVAL_SECONDS`: base frame sampling interval
- `MAP_OCR_MAX_FRAMES`: max frames used for map OCR voting
- `AGENT_DETECTION_MAX_FRAMES`: max frames for HUD agent detection
- `KILL_UI_TEMPLATE_FILENAME`: kill-feed row template filename
- `KILL_UI_MATCH_THRESHOLD`: template match threshold
- `PLAYER_USERNAME`: default username
- `USERNAME_MATCH_THRESHOLD`: OCR username match threshold
- `KILL_BACKGROUND_MAX_RED_RATIO`: red background rejection threshold
- `KILL_CANDIDATE_CLUSTER_GAP_SECONDS`: candidate clustering gap
- `KILL_CANDIDATE_MIN_HITS`: minimum detections per cluster
- ROI config:
  - `map_name`
  - `kill_feed`
  - `agent_icon`

---

## 11. Troubleshooting

No map detected:
- Verify Tesseract installation.
- Increase map ROI accuracy in `config.py`.

No kills detected:
- Update `agent_name_v2.png` from your current UI style.
- Ensure player name string matches in `--player`.
- Lower `USERNAME_MATCH_THRESHOLD` carefully.

Too many false kills:
- Increase `USERNAME_MATCH_THRESHOLD`.
- Decrease `KILL_BACKGROUND_MAX_RED_RATIO`.
- Increase cluster strictness (`KILL_CANDIDATE_MIN_HITS`).

Agent remains Unknown:
- Improve `assets/agent_icons` with cleaner, closer icon templates.
- Tune `KILLFEED_AGENT_ICON_MATCH_THRESHOLD`.

---

## 12. Performance Notes

- Folder mode can be expensive with many videos.
- For kill detection recall, app may run an internal finer pass at `0.1s`.
- Large/high-FPS videos increase runtime.

---

## 13. Git Hygiene

This repo includes `.gitignore` for:
- generated outputs
- local temp folders
- local video files
- Python caches/venvs

---

## 14. Quick Commands

Single file:

```bash
python main.py VALORANTv2.mp4 --player "Rangnar Lothbrok" --output output/result_v2.json
```

Folder (scan only this folder):

```bash
python main.py "E:\matches_root" --player "Rangnar Lothbrok"
```

Folder with fixed output name:

```bash
python main.py "E:\matches_root" --player "Rangnar Lothbrok" --output-name folder_result.json
```
