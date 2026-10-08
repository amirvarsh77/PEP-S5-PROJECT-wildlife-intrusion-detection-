# 🐾 Wildlife Intrusion Detection & Alert System

AI-based wildlife monitoring prototype: upload a CCTV video, a trained YOLO model detects animals,
and the dashboard shows the annotated video, an intrusion alert and a log of recent intrusions.

## 1. Folder structure

```
wildlife_intrusion/
├── app.py                 <- Streamlit dashboard + video processing
├── dummy_api.py           <- SIMULATED location / geofence data
├── requirements.txt
├── yolo2026s_best.pt      <- YOUR trained model (copy it here)
└── README.md
```

Your uploaded file is named `yolo2026s_best_.pt` (with an extra underscore). Rename it to
`yolo2026s_best.pt`. (The app also accepts similar names as a fallback.)

## 2. Install (once)

Open a terminal **inside the wildlife_intrusion folder**, then:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## 3. Run

```bash
streamlit run app.py
```

The browser opens at http://localhost:8501.

## 4. How it works

1. **Model loading** – `load_model()` loads `yolo2026s_best.pt` automatically (cached, so it loads once).
2. **Video processing** – `process_video()` reads the ORIGINAL video frame by frame with OpenCV,
   calls `model.predict()` on each frame, draws boxes/class/confidence with `result.plot()`
   and writes a new annotated video.
3. **Browser playback** – OpenCV writes `mp4v`, which browsers can't play, so
   `make_browser_compatible()` converts it to H.264 using ffmpeg (bundled via `imageio-ffmpeg`).
4. **Events** – `build_events()` takes the model's detections (animal + highest confidence per animal),
   stamps them with the current system time, and adds location data from `dummy_api.py`.
5. **Dashboard** – results are kept in `st.session_state` and shown as: processed video →
   summary cards → intrusion alert → recent intrusions table.

| Item | Source |
|---|---|
| Animal, confidence, bounding boxes | **Real** – your YOLO model |
| Detection time | **Real** – system clock when the detection happened |
| Location, latitude, longitude, geofence | **Simulated** – `dummy_api.py` (labelled in the UI) |

## 5. Replacing the dummy API later

Edit `get_intrusion_metadata()` in `dummy_api.py` so it calls your real service and returns
`{"location", "latitude", "longitude", "geofence_status", "is_simulated": False}`. `app.py` needs no change.

## 6. Settings

* **Confidence threshold** (sidebar, default 0.50) – ignore weaker detections.
* **Minimum frames to confirm** (default 3) – an animal must appear in this many frames to trigger an alert.

## 7. If your model's output differs

* Class names come from `model.names` automatically – nothing to change.
* If the model is a **segmentation** model, `result.boxes` still works.
* If it is an **OBB** (oriented box) model, use `result.obb` instead of `result.boxes` in `process_video()`.
* If the video is slow on CPU, use a smaller `imgsz` (e.g. `model.predict(..., imgsz=480)`) or a GPU.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| "Model file not found" | Put `yolo2026s_best.pt` next to `app.py` |
| Processed video won't play | `pip install imageio-ffmpeg`, or download it with the button |
| Very slow | Normal on CPU for long/4K videos; try a shorter clip |
| Large uploads rejected | `streamlit run app.py --server.maxUploadSize 1000` |

## 9. Research resources

The project research and related information are part of this repository. The dataset references are:

* [African wildlife dataset](https://docs.ultralytics.com/datasets/detect/african-wildlife)
* [Project repository](https://github.com/amirvarsh77/PEP-S5-PROJECT-wildlife-intrusion-detection-)
