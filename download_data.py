"""
Download audio and transcription data from GCS.
Transforms original URLs to working upload_goai format.
"""
import csv
import os
import re
import urllib.request
import time
import sys

DATA_CSV = "FT Data - data.csv"
AUDIO_DIR = "data/audio"
TEXT_DIR = "data/text"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


def transform_url(original_url):
    """
    Transform: .../joshtalks-data-collection/hq_data/hi/{folder}/{file}
    To:        .../upload_goai/{folder}/{file}
    """
    m = re.search(r"hq_data/hi/(\d+)/(.+)$", original_url)
    if m:
        folder_id = m.group(1)
        filename = m.group(2)
        return f"https://storage.googleapis.com/upload_goai/{folder_id}/{filename}"
    return original_url


def download_file(url, dest, retries=3):
    """Download a file with retry logic. Skip if already exists."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED after {retries} attempts: {e}")
                return False
    return False


def main():
    with open(DATA_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    success = 0
    failed = []

    print(f"Downloading data for {total} recordings...\n")

    for i, row in enumerate(rows):
        rec_id = row["recording_id"]
        audio_url = transform_url(row["rec_url_gcp"])
        trans_url = transform_url(row["transcription_url_gcp"])

        audio_dest = os.path.join(AUDIO_DIR, f"{rec_id}.wav")
        trans_dest = os.path.join(TEXT_DIR, f"{rec_id}.json")

        print(f"[{i+1}/{total}] Recording {rec_id}...")

        ok_audio = download_file(audio_url, audio_dest)
        ok_trans = download_file(trans_url, trans_dest)

        if ok_audio and ok_trans:
            success += 1
        else:
            failed.append(rec_id)
            if not ok_audio:
                print(f"  Audio failed: {audio_url}")
            if not ok_trans:
                print(f"  Transcription failed: {trans_url}")

        # Progress
        sys.stdout.flush()

    print(f"\n{'='*50}")
    print(f"Download complete: {success}/{total} successful")
    if failed:
        print(f"Failed recordings: {failed}")


if __name__ == "__main__":
    main()
