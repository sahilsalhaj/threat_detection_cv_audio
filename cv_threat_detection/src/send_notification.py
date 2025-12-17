# send_notification.py
import requests
import os

URL = "https://2806d5c9416f.ngrok-free.app/api/v1/threats/threat"
DEVICE_KEY = "sucky-sucky-hehe"


def send_threat_notification(
    image_path: str,
    hive_id: int,
    threat_type: str,
    confidence: float,
    timeout: int = 10,
):
    """
    Sends a threat notification to backend.

    Args:
        image_path (str): Path to image file with drawn detections
        hive_id (int): Hive number (1, 2, 3, or 4)
        threat_type (str): e.g. "wasp", "hornet"
        confidence (float): Detection confidence (0-1)
        timeout (int): Request timeout in seconds
    """

    if not os.path.exists(image_path):
        print(f"[NOTIFY] Image not found: {image_path}")
        return False

    headers = {
        "X-Device-Key": DEVICE_KEY
    }

    files = {
        "image": open(image_path, "rb")
    }

    data = {
        "hive_id": hive_id,
        "threat_type": threat_type,
        "confidence": round(float(confidence), 3)
    }

    try:
        response = requests.post(
            URL,
            headers=headers,
            files=files,
            data=data,
            timeout=timeout
        )

        print(f"[NOTIFY] Hive {hive_id} → {response.status_code}")
        return response.ok

    except requests.exceptions.RequestException as e:
        print("[NOTIFY] Request failed:", e)
        return False

    finally:
        files["image"].close()
