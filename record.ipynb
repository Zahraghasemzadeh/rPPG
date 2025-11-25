import os
import time
from pathlib import Path
import cv2

# -------------------- Config (env) --------------------
OUT_DIR      = Path(os.getenv("IMAGES_DIR", "images"))
FPS          = float(os.getenv("FPS", "30"))
DURATION_SEC = float(os.getenv("DURATION_SEC", "30"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
RES_WIDTH    = int(os.getenv("RES_WIDTH", "640"))
RES_HEIGHT   = int(os.getenv("RES_HEIGHT", "480"))
SHOW_PREVIEW = os.getenv("SHOW_PREVIEW", "1") not in ("0", "false", "False")


ROI_WIDTH = 200
ROI_HEIGHT = 200

# Global variables to store the clicked point
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked point: {clicked_point}")

def main():
    global clicked_point
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # Try to set resolution & fps (not all cameras honor these)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RES_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    
    # --- Step 1: Show first frame and let user click ---
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame for ROI selection")

    cv2.imshow("Click ROI Center", frame)
    cv2.setMouseCallback("Click ROI Center", mouse_callback)

    print("Click on the image to select the center of your ROI...")
    while clicked_point is None:
        cv2.waitKey(1)
    cv2.destroyWindow("Click ROI Center")

    ROI_CENTER_X, ROI_CENTER_Y = clicked_point

    show_preview = SHOW_PREVIEW
    if show_preview:
        try:
            cv2.namedWindow("Recording (Q/Esc to stop)", cv2.WINDOW_NORMAL)
        except Exception:
            show_preview = False

    total_frames = int(round(FPS * DURATION_SEC))
    print(f"Recording ~{total_frames} frames @ {FPS} FPS for {DURATION_SEC}s → {OUT_DIR}/")

    start = time.perf_counter()
    saved = 0

    for i in range(total_frames):
        next_time = start + (i / FPS)

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.001)
            continue
        
        # Crop ROI around the clicked pixel
        x1 = max(0, ROI_CENTER_X - ROI_WIDTH // 2)
        y1 = max(0, ROI_CENTER_Y - ROI_HEIGHT // 2)
        x2 = min(frame.shape[1], ROI_CENTER_X + ROI_WIDTH // 2)
        y2 = min(frame.shape[0], ROI_CENTER_Y + ROI_HEIGHT // 2)
        roi_frame = frame[y1:y2, x1:x2]

        # Resize ROI before saving
        roi_resized = cv2.resize(roi_frame, (RES_WIDTH, RES_HEIGHT))
        
        ts = time.time()
        cv2.imwrite(str(OUT_DIR / f"{ts}.png"), roi_resized)
        saved += 1

        if show_preview:
            cv2.imshow("Recording (Q/Esc to stop)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                print("Stopped by user.")
                break

        if saved % 60 == 0:
            elapsed = time.perf_counter() - start
            print(f"Saved {saved}/{total_frames} frames (@ {saved/elapsed:.2f} fps)")

        delay = next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    avg_fps = (saved / elapsed) if elapsed > 0 else 0.0
    print(f"Done. Saved {saved} frames in {elapsed:.2f}s (avg {avg_fps:.2f} fps).")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
