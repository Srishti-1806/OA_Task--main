import requests
import os
import json
import time

BASE = "https://storage.googleapis.com/upload_goai"

def build_urls(user_id, rec_id):
    return {
        "json": f"{BASE}/{user_id}/{rec_id}_transcription.json",
        "audio_wav": f"{BASE}/{user_id}/{rec_id}.wav",
        "audio_mp3": f"{BASE}/{user_id}/{rec_id}.mp3"
    }


def safe_get(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r
    except:
        return None
    return None


def download_sample(row):
    os.makedirs("data/audio", exist_ok=True)
    os.makedirs("data/text", exist_ok=True)

    user_id = str(row["user_id"])
    rec_id = str(row["recording_id"])

    urls = build_urls(user_id, rec_id)

    # -------- JSON --------
    r = safe_get(urls["json"])
    if not r:
        print(f"JSON fail: {rec_id}")
        return False

    try:
        data = r.json()
    except:
        print(f"Bad JSON: {rec_id}")
        return False

    with open(f"data/text/{rec_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"JSON OK: {rec_id}")

    # -------- AUDIO (try wav → mp3) --------
    audio_saved = False

    for key in ["audio_wav", "audio_mp3"]:
        r = safe_get(urls[key])
        if r:
            ext = "wav" if "wav" in key else "mp3"
            path = f"data/audio/{rec_id}.{ext}"

            with open(path, "wb") as f:
                f.write(r.content)

            print(f"AUDIO OK ({ext}): {rec_id}")
            audio_saved = True
            break

    if not audio_saved:
        print(f"AUDIO fail: {rec_id}")

    time.sleep(0.5)  # avoid rate limit

    return audio_saved