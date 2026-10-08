"""
dummy_api.py  --  SIMULATED location / geofence data layer
============================================================

IMPORTANT: Nothing in this file is real GPS tracking.
The CCTV video does not contain coordinates, so for this prototype we return
FIXED, made-up values for the camera location and geofence status.

The animal name, confidence and bounding boxes do NOT come from here.
They come from your trained YOLO model (see app.py).

HOW TO REPLACE THIS WITH A REAL API LATER
-----------------------------------------
Keep the function name `get_intrusion_metadata` and keep returning a
dictionary with the same keys. Then app.py needs NO changes.

    {
        "location": "Forest Zone A",
        "latitude": 12.8456,
        "longitude": 80.2267,
        "geofence_status": "INSIDE",      # or "OUTSIDE"
        "is_simulated": False,            # set to False for real data
        "source": "Real camera API",
    }
"""

# One fake camera. Add more entries here if you want to demo several cameras.
DUMMY_CAMERAS = {
    "CAM-01": {
        "location": "Forest Zone A",
        "latitude": 12.8456,
        "longitude": 80.2267,
        "geofence_status": "INSIDE",
    },
}

DEFAULT_CAMERA_ID = "CAM-01"


def get_intrusion_metadata(animal_name: str, camera_id: str = DEFAULT_CAMERA_ID) -> dict:
    """
    Return SIMULATED location + geofence information for a detection.

    animal_name : the class predicted by YOLO. It is accepted so a real API
                  can use it later; the dummy version ignores it.
    camera_id   : which camera produced the video.
    """
    camera = DUMMY_CAMERAS.get(camera_id, DUMMY_CAMERAS[DEFAULT_CAMERA_ID])

    # ---- A real API call would go here, for example: ---------------------
    # import requests
    # r = requests.get("https://your-server/api/camera-location",
    #                  params={"camera_id": camera_id}, timeout=5)
    # return r.json()   # must contain the keys shown above
    # ----------------------------------------------------------------------

    return {
        "location": camera["location"],
        "latitude": camera["latitude"],
        "longitude": camera["longitude"],
        "geofence_status": camera["geofence_status"],   # "INSIDE" or "OUTSIDE"
        "is_simulated": True,                           # the UI uses this flag for the DEMO label
        "source": "Simulated demo data (dummy_api.py)",
    }
