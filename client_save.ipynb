#!/usr/bin/env python3
import asyncio
import base64
import os
import re
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Union
from datetime import datetime
import contextlib

import numpy as np
import cv2
import websockets
import orjson

# -------------------- Config (env overridable) --------------------
BACKEND_WS_BASE   = os.getenv("BACKEND_WS_BASE", "ws://3.67.186.245:8003/ws/")
API_KEY           = os.getenv("API_KEY", "gLw7B_60VXDNvZItsNhkwnVXqD6VJ40mpbgszmipy3k")
IMAGES_DIR        = os.getenv("IMAGES_DIR", "images")
OUTPUT_DIR        = Path(os.getenv("OUTPUT_DIR", "outputs"))
FPS               = float(os.getenv("FPS", "30"))
CLIENT            = os.getenv("CLIENT", "pythonClient")
OBJECT_ID         = os.getenv("OBJECT_ID", "")
CALLBACK_URL      = os.getenv("CALLBACK_URL", "")  # leave empty unless required

# Performance/transport knobs
FRAME_FORMAT      = os.getenv("FRAME_FORMAT", "jpeg").lower()  # "raw" | "jpeg"
JPEG_QUALITY      = int(os.getenv("JPEG_QUALITY", "75"))
ENC_WORKERS       = int(os.getenv("ENC_WORKERS", str(os.cpu_count() or 4)))
QUEUE_MAXSIZE     = int(os.getenv("QUEUE_MAXSIZE", "512"))
WS_MAX_SIZE       = 2**22  # 4 MiB
WS_TEXT_FRAMES    = os.getenv("WS_TEXT_FRAMES", "1") not in ("0", "false", "False")

# -------------------- Files & encoding --------------------
_TS_PNG = re.compile(r"^\d+(?:\.\d+)?\.png$")  # matches 1747154380.5511632.png

def list_timestamped_pngs() -> list[Path]:
    paths = [p for p in Path(IMAGES_DIR).glob("*.png") if _TS_PNG.match(p.name)]
    paths.sort(key=lambda p: float(p.stem) if p.stem.replace('.', '', 1).isdigit() else 0.0)
    return paths

def get_size_from_image(path: Path) -> tuple[int, int]:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 640, 480
    h, w = img.shape[:2]
    return (w, h)

def b64_raw(path: Path) -> str:
    data = np.fromfile(str(path), dtype=np.uint8)
    return base64.b64encode(data.tobytes()).decode("ascii")

def b64_jpeg(path: Path, quality: int) -> str:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to encode JPEG: {path}")
    return base64.b64encode(buf.tobytes()).decode("ascii")

async def encoder_producer(paths, q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Offload encoding to threads and feed an asyncio.Queue."""
    with ThreadPoolExecutor(max_workers=ENC_WORKERS) as pool:
        for p in paths:
            if FRAME_FORMAT == "raw":
                b64 = await loop.run_in_executor(pool, b64_raw, p)
            else:
                b64 = await loop.run_in_executor(pool, b64_jpeg, p, JPEG_QUALITY)
            ts_str = p.stem  # filename (without .png) as timestamp
            await q.put((p.name, ts_str, b64))
    await q.put(None)  # sentinel

# -------------------- WebSocket helpers --------------------
def build_ws_url() -> str:
    from urllib.parse import urlencode
    params = {"api_key": API_KEY, "client": CLIENT}
    if OBJECT_ID:
        params["objectId"] = OBJECT_ID
    if CALLBACK_URL:
        params["callback_url"] = CALLBACK_URL
    return f"{BACKEND_WS_BASE.rstrip('/')}/?{urlencode(params)}"

def build_payload(datapt_id: str, state: str, timestamp: str, frame_b64: str, advanced: bool = True) -> dict:
    # Centralized payload builder — mirrors JS client
    return {
        "datapt_id": datapt_id,
        "state": state,             # "stream" | "end"
        "advanced": bool(advanced),
        "timestamp": timestamp,     # from filename stem
        "frame_data": frame_b64,    # base64 image (jpeg or raw png bytes)
    }

def dump_payload(obj: dict) -> Union[bytes, str]:
    if WS_TEXT_FRAMES:
        return orjson.dumps(obj).decode("utf-8")
    return orjson.dumps(obj)

def _pretty_server_log(msg_obj):
    """Print full server message (no filtering)."""
    try:
        if isinstance(msg_obj, (dict, list)):
            return "<< " + orjson.dumps(msg_obj).decode("utf-8", errors="ignore")
        return f"<< {msg_obj}"
    except Exception:
        return f"<< {repr(msg_obj)}"

# -------------------- Collector listener & saver --------------------
async def server_listener(ws, rppg_buf: list, ts_buf: list, hr_buf: list):
    """Listen to server, print messages, and collect rPPG/HR."""
    try:
        async for msg in ws:
            # decode JSON (text or binary)
            if isinstance(msg, (bytes, bytearray)):
                s = msg.decode("utf-8", errors="ignore")
            else:
                s = msg
            try:
                obj = orjson.loads(s)
            except Exception:
                print("<< " + (s if isinstance(s, str) else repr(s)))
                continue

            # always log
            print(_pretty_server_log(obj))

            # collect rPPG + timestamps
            adv = obj.get("advanced") or {}
            rppg_chunk = adv.get("rppg") or []
            ts_chunk   = adv.get("rppg_timestamps") or []

            if isinstance(rppg_chunk, list) and len(rppg_chunk) > 0:
                if isinstance(ts_chunk, list) and len(ts_chunk) > 0:
                    n = min(len(rppg_chunk), len(ts_chunk))
                    rppg_buf.extend(rppg_chunk[:n])
                    ts_buf.extend(ts_chunk[:n])
                else:
                    rppg_buf.extend(rppg_chunk)

            # collect HR if valid (> 0)
            inf = obj.get("inference") or {}
            hr  = inf.get("hr", -1)
            if isinstance(hr, (int, float)) and hr > 0:
                hr_buf.append(float(hr))

    except websockets.exceptions.ConnectionClosed:
        pass

def save_outputs(run_id: str, rppg_buf: list, ts_buf: list, hr_buf: list, also_npz: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # De-duplicate by timestamp if lengths match
    if ts_buf and len(ts_buf) == len(rppg_buf):
        seen = set()
        rppg_out, ts_out = [], []
        for x, t in zip(rppg_buf, ts_buf):
            if t in seen:
                continue
            seen.add(t)
            rppg_out.append(x)
            ts_out.append(t)
    else:
        rppg_out, ts_out = rppg_buf, ts_buf

    rppg_arr = np.asarray(rppg_out, dtype=np.float32)
    ts_arr   = np.asarray(ts_out,  dtype=np.float64) if ts_out else np.array([], dtype=np.float64)
    hr_arr   = np.asarray(hr_buf,  dtype=np.float32) if hr_buf else np.array([], dtype=np.float32)

    np.save(OUTPUT_DIR / f"rppg_{run_id}.npy", rppg_arr)
    np.save(OUTPUT_DIR / f"rppg_timestamps_{run_id}.npy", ts_arr)
    if hr_arr.size > 0:
        np.save(OUTPUT_DIR / f"hr_values_{run_id}.npy", hr_arr)
        np.save(OUTPUT_DIR / f"hr_mean_{run_id}.npy", np.array([hr_arr.mean()], dtype=np.float32))

    if also_npz:
        np.savez(OUTPUT_DIR / f"session_{run_id}.npz",
                 rppg=rppg_arr,
                 rppg_timestamps=ts_arr,
                 hr_values=hr_arr,
                 hr_mean=(np.array([hr_arr.mean()], dtype=np.float32) if hr_arr.size > 0 else np.array([], dtype=np.float32)))

    print(f"[saved] {OUTPUT_DIR}/rppg_{run_id}.npy                  shape={rppg_arr.shape}")
    print(f"[saved] {OUTPUT_DIR}/rppg_timestamps_{run_id}.npy      shape={ts_arr.shape}")
    if hr_arr.size > 0:
        print(f"[saved] {OUTPUT_DIR}/hr_values_{run_id}.npy         shape={hr_arr.shape}, mean={hr_arr.mean():.2f}")
        print(f"[saved] {OUTPUT_DIR}/hr_mean_{run_id}.npy           shape=(1,)")
    if also_npz:
        print(f"[saved] {OUTPUT_DIR}/session_{run_id}.npz           (bundle)")

# -------------------- Main --------------------
async def main():
    if FRAME_FORMAT not in {"raw", "jpeg"}:
        raise ValueError(f"FRAME_FORMAT must be 'raw' or 'jpeg', got: {FRAME_FORMAT}")

    paths = list_timestamped_pngs()
    if not paths:
        print(f"No timestamped PNGs found under {IMAGES_DIR}/ (e.g. 1747154380.5511632.png)")
        return

    # # (Optionally) fetch the first image size (not used but kept for parity)
    # _w, _h = get_size_from_image(paths[0])

    ws_url = build_ws_url()
    print("Connecting to:", ws_url)

    async with websockets.connect(ws_url, max_size=WS_MAX_SIZE, compression=None) as ws:
        # buffers for this run
        rppg_buf, ts_buf, hr_buf = [], [], []
        listener_task = asyncio.create_task(server_listener(ws, rppg_buf, ts_buf, hr_buf))

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        _ = asyncio.create_task(encoder_producer(paths, q, loop))

        # prefill ~0.5s
        prefill_target = max(1, int(FPS * 0.5))
        stash = []
        while len(stash) < prefill_target:
            item = await q.get()
            if item is None:
                break
            stash.append(item)

        start = perf_counter()
        sent = 0

        async def send_item(item, state="stream"):
            _, ts_str, b64 = item
            payload = build_payload(datapt_id=str(uuid.uuid4()), state=state, timestamp=ts_str, frame_b64=b64, advanced=True)
            await ws.send(dump_payload(payload))

        # send prefilled
        for it in stash:
            await send_item(it, "stream")
            sent += 1

            next_time = start + sent / FPS
            delay = next_time - perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if sent % 60 == 0:
                elapsed = perf_counter() - start
                print(f">> sent {sent} frames @ {sent/elapsed:.2f} fps")

        # drain the queue
        last_item = None
        while True:
            it = await q.get()
            if it is None:
                break
            last_item = it
            await send_item(it, "stream")
            sent += 1

            next_time = start + sent / FPS
            delay = next_time - perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

            if sent % 60 == 0:
                elapsed = perf_counter() - start
                print(f">> sent {sent} frames @ {sent/elapsed:.2f} fps")

        # final "end"
        if last_item is not None:
            await send_item(last_item, "end")
        print(">> sent END frame; awaiting server completion...")

        try:
            await asyncio.wait_for(listener_task, timeout=20.0)
        except asyncio.TimeoutError:
            listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listener_task

        # Save outputs for this run
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_outputs(run_id, rppg_buf, ts_buf, hr_buf)

    print("Done.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
