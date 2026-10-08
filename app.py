"""
app.py  --  AI-Based Wildlife Intrusion Detection & Real-Time Alert System
==========================================================================

Run with:   streamlit run app.py

DATA FLOW (read this first)
---------------------------
 Upload original video
        |
        v
 process_video()  -- reads the video frame by frame with OpenCV
        |             runs YOUR trained YOLO model on every frame
        |             draws boxes / class / confidence (result.plot())
        |             writes a NEW annotated video
        |             records every detected animal + its confidence
        v
 build_events()   -- turns the model's detections into "intrusion events"
        |             (animal + confidence = REAL model output)
        |             (time = current system time)
        |             (location / lat / long / geofence = SIMULATED, dummy_api.py)
        v
 st.session_state -- results are stored so they survive Streamlit reruns
        v
 Dashboard        -- processed video, summary cards, alert, recent intrusions
"""

import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from dummy_api import get_intrusion_metadata

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "yolo2026s_best.pt"        # must sit in the same folder as app.py
SUPPORTED_FORMATS = ["mp4", "avi", "mov", "mkv"]

# The 17 classes your model was trained on. Only used to double-check the model;
# the names shown on the dashboard always come from the model itself.
EXPECTED_CLASSES = [
    "AmurTiger", "Badger", "BlackBear", "Cow", "Dog", "Hare", "Leopard",
    "LeopardCat", "MuskDeer", "RaccoonDog", "RedFox", "RoeDeer", "Sable",
    "SikaDeer", "Weasel", "WildBoar", "Y.T.Marten",
]


# ---------------------------------------------------------------------------
# PAGE SETUP + CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Wildlife Intrusion Detection", page_icon="🐾", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1200px;}

    .hero {
        background: linear-gradient(135deg, #064e3b 0%, #0f766e 55%, #0e7490 100%);
        color: #fff; padding: 28px 34px; border-radius: 16px; margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,.18);
    }
    .hero h1 {margin: 0; font-size: 2.1rem; color: #fff;}
    .hero p  {margin: 6px 0 0 0; font-size: 1.05rem; opacity: .9;}

    .section-title {font-size: 1.25rem; font-weight: 700; margin: 26px 0 10px 0;}

    .card {
        border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
        padding: 18px 20px; background: rgba(128,128,128,.07); height: 100%;
    }
    .card .label {font-size: .85rem; opacity: .7; margin-bottom: 4px;}
    .card .value {font-size: 1.7rem; font-weight: 700;}

    .alert-box {
        border-radius: 16px; padding: 22px 28px; margin-top: 8px;
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        color: #fff; box-shadow: 0 6px 22px rgba(185,28,28,.35);
    }
    .alert-box.outside {background: linear-gradient(135deg, #14532d 0%, #15803d 100%);
                        box-shadow: 0 6px 22px rgba(21,128,61,.35);}
    .alert-box h2 {margin: 0 0 14px 0; color: #fff; font-size: 1.6rem;}
    .alert-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px 28px;}
    .alert-grid div span {display:block; font-size:.78rem; opacity:.8; text-transform:uppercase; letter-spacing:.04em;}
    .alert-grid div b {font-size: 1.2rem;}

    .badge {
        display:inline-block; font-size:.68rem; font-weight:700; padding:2px 8px;
        border-radius: 999px; margin-left: 6px; vertical-align: middle; letter-spacing:.04em;
    }
    .badge.real {background:#bbf7d0; color:#14532d;}
    .badge.sim  {background:#fde68a; color:#78350f;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
def find_model_file():
    """
    Look for yolo2026s_best.pt next to app.py.
    If it isn't found, also accept names like 'yolo2026s_best_.pt' or
    'yolo2026s_best (1).pt' (browsers often rename downloads like that).
    """
    exact = APP_DIR / MODEL_FILENAME
    if exact.exists():
        return exact
    candidates = sorted(APP_DIR.glob("yolo2026s_best*.pt"))
    return candidates[0] if candidates else None


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Load YOUR trained weights once and keep them in memory.
    @st.cache_resource means Streamlit will not reload the model on every click.
    """
    from ultralytics import YOLO          # imported here so the page opens quickly
    return YOLO(model_path)


# ---------------------------------------------------------------------------
# VIDEO HELPERS
# ---------------------------------------------------------------------------
def get_ffmpeg_exe():
    """Find an ffmpeg program: first the one bundled with imageio-ffmpeg, then the system one."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def make_browser_compatible(src: str, dst: str) -> bool:
    """
    OpenCV writes 'mp4v' videos, which most browsers CANNOT play.
    This re-encodes the video to H.264 (yuv420p), which every browser plays.
    Returns True on success, False if ffmpeg is unavailable / fails.
    """
    ffmpeg = get_ffmpeg_exe()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg, "-y", "-i", src,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # H.264 needs even width/height
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", "-an", dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return Path(dst).exists() and Path(dst).stat().st_size > 0
    except Exception:
        return False


def process_video(model, input_path, output_path, conf_threshold, progress_callback=None):
    """
    THE CORE FUNCTION: runs YOLO on every frame of the ORIGINAL video.

    Returns (stats, info)
      stats : {animal_name: {...}} built ONLY from the model's predictions
      info  : fps, frame counts, etc.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video. The file may be corrupted or use an unsupported codec.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output video file.")

    stats = {}
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ---- 1. Run YOUR model on this frame -----------------------------
            result = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]

            # ---- 2. Draw boxes + class + confidence, save frame ---------------
            writer.write(result.plot())

            # ---- 3. Record what the model found ------------------------------
            seen_this_frame = set()
            if result.boxes is not None:
                for box in result.boxes:
                    name = model.names[int(box.cls[0].item())]
                    conf = float(box.conf[0].item())

                    s = stats.setdefault(name, {
                        "max_conf": 0.0, "frames": 0,
                        "detected_at": datetime.now(),       # dynamic system time
                        "video_second": frame_idx / fps,
                    })
                    if conf > s["max_conf"]:
                        s["max_conf"] = conf
                    seen_this_frame.add(name)

            for name in seen_this_frame:
                stats[name]["frames"] += 1

            frame_idx += 1
            if progress_callback and (frame_idx % 5 == 0 or frame_idx == total_frames):
                progress_callback(frame_idx, total_frames)
    finally:
        cap.release()
        writer.release()

    if frame_idx == 0:
        raise ValueError("The video contains no readable frames.")

    return stats, {"fps": fps, "frames": frame_idx, "width": width, "height": height}


def build_events(stats, min_frames, video_name):
    """
    Convert model detections into intrusion events.
      REAL  : animal, confidence            (from YOLO)
      REAL  : time                          (current system clock)
      FAKE  : location/lat/long/geofence    (from dummy_api.py)
    Animals seen in fewer than `min_frames` frames are ignored (filters one-frame false alarms).
    """
    events = []
    for animal, s in stats.items():
        if s["frames"] < min_frames:
            continue
        meta = get_intrusion_metadata(animal)
        events.append({
            "animal": animal,
            "confidence": s["max_conf"],
            "frames": s["frames"],
            "video_second": s["video_second"],
            "detected_at": s["detected_at"],
            "location": meta["location"],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "geofence_status": meta["geofence_status"],
            "is_simulated": meta.get("is_simulated", True),
            "video_name": video_name,
        })
    events.sort(key=lambda e: e["confidence"], reverse=True)
    return events


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------
def geofence_label(status: str) -> str:
    return "🔴 INSIDE" if str(status).upper() == "INSIDE" else "🟢 OUTSIDE"


def summary_card(label, value):
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'


def render_summary_cards(event):
    st.markdown('<div class="section-title">📊 Detection Summary '
                '<span class="badge real">FROM YOLO MODEL</span></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(summary_card("🐾 Animal", event["animal"]), unsafe_allow_html=True)
    c2.markdown(summary_card("📊 Confidence", f"{event['confidence'] * 100:.1f}%"), unsafe_allow_html=True)
    c3.markdown(summary_card("🕐 Detection Time", event["detected_at"].strftime("%I:%M %p")),
                unsafe_allow_html=True)


def render_alert(event):
    inside = str(event["geofence_status"]).upper() == "INSIDE"
    st.markdown('<div class="section-title">🚨 Intrusion Alert</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="alert-box {'' if inside else 'outside'}">
          <h2>🚨 WILDLIFE INTRUSION DETECTED</h2>
          <div class="alert-grid">
            <div><span>Animal <i class="badge real">MODEL</i></span><b>{event['animal']}</b></div>
            <div><span>Confidence <i class="badge real">MODEL</i></span><b>{event['confidence'] * 100:.1f}%</b></div>
            <div><span>Detection Time</span><b>{event['detected_at'].strftime('%d-%m-%Y %I:%M %p')}</b></div>
            <div><span>Location <i class="badge sim">SIMULATED</i></span><b>{event['location']}</b></div>
            <div><span>Latitude <i class="badge sim">SIMULATED</i></span><b>{event['latitude']}</b></div>
            <div><span>Longitude <i class="badge sim">SIMULATED</i></span><b>{event['longitude']}</b></div>
            <div><span>Geofence Status <i class="badge sim">SIMULATED</i></span><b>{geofence_label(event['geofence_status'])}</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if event.get("is_simulated", True):
        st.caption("⚠️ Location, latitude, longitude and geofence status are SIMULATED demo data "
                   "(dummy_api.py). The CCTV camera is not providing real GPS coordinates.")


def render_recent_intrusions():
    st.markdown('<div class="section-title">📋 Recent Intrusions</div>', unsafe_allow_html=True)
    history = st.session_state["history"]
    if not history:
        st.info("No intrusions recorded yet.")
        return

    df = pd.DataFrame([{
        "Animal": e["animal"],
        "Confidence": f"{e['confidence'] * 100:.1f}%",
        "Time": e["detected_at"].strftime("%I:%M %p"),
        "Date": e["detected_at"].strftime("%d-%m-%Y"),
        "Location": e["location"],
        "Latitude": e["latitude"],
        "Longitude": e["longitude"],
        "Geofence Status": geofence_label(e["geofence_status"]),
        "Video": e["video_name"],
    } for e in reversed(history)])                      # newest first

    st.dataframe(df, hide_index=True)
    st.caption("Animal / Confidence / Time come from the model run. "
               "Location, Latitude, Longitude and Geofence are simulated demo data.")
    if st.button("🗑️ Clear history"):
        st.session_state["history"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    # Session state = memory that survives Streamlit reruns (every click reruns the script)
    st.session_state.setdefault("history", [])      # all events ever detected this session
    st.session_state.setdefault("result", None)     # result of the latest run

    # ---- Header ------------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
          <h1>🐾 Wildlife Intrusion Detection &amp; Alert System</h1>
          <p>AI-Based Wildlife Monitoring and Real-Time Intrusion Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Sidebar -----------------------------------------------------------
    st.sidebar.header("⚙️ Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05,
                                       help="Detections below this confidence are ignored.")
    min_frames = st.sidebar.slider("Minimum frames to confirm", 1, 30, 3,
                                   help="An animal must appear in at least this many frames to raise an alert. "
                                        "Filters out one-frame false detections.")

    model_file = find_model_file()
    st.sidebar.divider()
    st.sidebar.subheader("🧠 Model")
    if model_file is None:
        st.sidebar.error(f"{MODEL_FILENAME} not found")
    else:
        st.sidebar.success(f"Loaded from: {model_file.name}")
    st.sidebar.info("**Real (from YOLO):** animal, confidence, bounding boxes.\n\n"
                    "**Simulated (dummy_api.py):** location, latitude, longitude, geofence.")

    # ---- Error: model missing ---------------------------------------------
    if model_file is None:
        st.error(f"❌ Model file **{MODEL_FILENAME}** was not found. "
                 f"Place it in the same folder as app.py:\n\n`{APP_DIR}`")
        st.stop()

    # ---- Step 3: upload ------------------------------------------------------
    st.markdown('<div class="section-title">📤 Upload CCTV / Wildlife Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Supported: MP4 | AVI | MOV | MKV", type=SUPPORTED_FORMATS)

    if uploaded is None:
        st.info("👆 Please upload a wildlife / CCTV video to begin.")
        render_recent_intrusions()
        st.stop()

    # ---- Step 4: show the uploaded video ------------------------------------
    ext = Path(uploaded.name).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        st.error(f"❌ Unsupported format “.{ext}”. Please upload: {', '.join(f.upper() for f in SUPPORTED_FORMATS)}.")
        st.stop()

    st.markdown('<div class="section-title">🎞️ Original Video</div>', unsafe_allow_html=True)
    st.caption(f"File: **{uploaded.name}**  ({uploaded.size / 1_048_576:.1f} MB)")
    with st.expander("Preview original video", expanded=False):
        st.video(uploaded)

    # ---- Step 5/6: run detection --------------------------------------------
    st.markdown('<div class="section-title">🔍 Detection</div>', unsafe_allow_html=True)
    if st.button("🚀 Run Wildlife Detection", type="primary"):
        in_path = out_raw = out_final = None
        try:
            with st.spinner("Loading yolo2026s_best.pt ..."):
                model = load_model(str(model_file))

            # Save the upload to disk so OpenCV can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
                f.write(uploaded.getvalue())
                in_path = f.name
            out_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            out_final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            bar = st.progress(0.0, text="Starting detection ...")

            def on_progress(done, total):
                if total:
                    bar.progress(min(done / total, 1.0), text=f"Analysing frame {done} of {total} ...")
                else:
                    bar.progress(0.5, text=f"Analysing frame {done} ...")

            stats, info = process_video(model, in_path, out_raw, conf_threshold, on_progress)

            bar.progress(1.0, text="Preparing video for browser playback ...")
            playable = make_browser_compatible(out_raw, out_final)
            video_path = out_final if playable else out_raw
            bar.empty()

            events = build_events(stats, min_frames, uploaded.name)
            st.session_state["history"].extend(events)
            st.session_state["result"] = {
                "events": events,
                "video_bytes": Path(video_path).read_bytes(),
                "playable": playable,
                "info": info,
                "total_raw": sum(s["frames"] for s in stats.values()),
            }
        except Exception as exc:                       # show a friendly message instead of crashing
            st.session_state["result"] = None
            st.error(f"❌ Video processing failed: {exc}")
        finally:
            for p in (in_path, out_raw, out_final):    # clean up temp files
                if p:
                    Path(p).unlink(missing_ok=True)

    # ---- Step 7+: display results --------------------------------------------
    result = st.session_state["result"]
    if result:
        st.markdown('<div class="section-title">🎥 AI Detection Output</div>', unsafe_allow_html=True)
        st.video(result["video_bytes"])
        st.download_button("⬇️ Download processed video", result["video_bytes"],
                           file_name="wildlife_detection_output.mp4", mime="video/mp4")
        if not result["playable"]:
            st.warning("ffmpeg was not available, so the video could not be converted for browsers. "
                       "If it doesn't play, download it, or run `pip install imageio-ffmpeg`.")

        events = result["events"]
        if not events:
            st.warning("No wildlife detected in the uploaded video.")
        else:
            top = events[0]                            # strongest detection
            render_summary_cards(top)
            render_alert(top)
            if len(events) > 1:
                others = ", ".join(f"{e['animal']} ({e['confidence'] * 100:.1f}%)" for e in events[1:])
                st.info(f"Other animals also detected: {others}")
            i = result["info"]
            st.caption(f"Processed {i['frames']} frames ({i['width']}×{i['height']} @ {i['fps']:.0f} fps) "
                       f"at confidence ≥ {conf_threshold:.2f}.")

    render_recent_intrusions()


if __name__ == "__main__":
    main()
