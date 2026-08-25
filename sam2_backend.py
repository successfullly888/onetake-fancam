"""Optional SAM 2.1 video-object tracker used by the cloud/GPU build.

The module is safe to import in the lightweight Windows build: torch and SAM 2
are imported lazily only when the high-precision engine is selected.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path


_PREDICTOR = None
_PREDICTOR_LOCK = threading.Lock()


def status() -> dict:
    torch_ready = importlib.util.find_spec("torch") is not None
    sam2_ready = importlib.util.find_spec("sam2") is not None
    checkpoint = Path(os.environ.get("ONETAKE_SAM2_CHECKPOINT", "/opt/models/sam2.1_hiera_small.pt"))
    return {
        "available": bool(torch_ready and sam2_ready and checkpoint.exists()),
        "engine": "SAM 2.1 Hiera Small",
        "runtime": "gpu-video-object-segmentation",
        "checkpoint": checkpoint.name,
        "error": None if torch_ready and sam2_ready and checkpoint.exists() else "SAM 2.1 runtime is not installed in this build",
    }


def _predictor():
    global _PREDICTOR
    with _PREDICTOR_LOCK:
        if _PREDICTOR is not None:
            return _PREDICTOR
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 2.1 high-precision mode requires a CUDA GPU in the deployed build")
        checkpoint = os.environ.get("ONETAKE_SAM2_CHECKPOINT", "/opt/models/sam2.1_hiera_small.pt")
        config = os.environ.get("ONETAKE_SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
        _PREDICTOR = build_sam2_video_predictor(config, checkpoint, device="cuda")
        return _PREDICTOR


def _ffmpeg() -> str:
    configured = os.environ.get("ONETAKE_FFMPEG")
    candidates = [configured, shutil.which("ffmpeg"), "E:/ffmpeg/bin/ffmpeg.exe"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate).resolve())
    raise RuntimeError("FFmpeg is required to prepare SAM 2.1 video frames")


def _box_from_mask(mask, width: int, height: int):
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) < 16 or len(ys) < 16:
        return None
    x1 = max(0.0, float(xs.min()))
    y1 = max(0.0, float(ys.min()))
    x2 = min(float(width), float(xs.max() + 1))
    y2 = min(float(height), float(ys.max() + 1))
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
    }


def track_video_object(video_path: Path, start: float, end: float, selection_time: float,
                       seed_box: dict, corrections: list | None = None,
                       tracking_fps: float = 12.0) -> dict:
    """Track one user-selected object with SAM 2.1 and return camera-ready boxes."""
    import cv2
    import numpy as np
    import torch

    predictor = _predictor()
    corrections = corrections or []
    tracking_fps = max(6.0, min(float(tracking_fps), 15.0))
    duration = max(0.05, float(end) - float(start))

    with tempfile.TemporaryDirectory(prefix="onetake-sam2-") as temporary:
        frame_dir = Path(temporary) / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        command = [
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{float(start):.4f}", "-t", f"{duration:.4f}",
            "-i", str(video_path), "-vf", f"fps={tracking_fps:.4f}",
            "-q:v", "2", str(frame_dir / "%06d.jpg"),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
        frames = sorted(frame_dir.glob("*.jpg"))
        if completed.returncode != 0 or not frames:
            raise RuntimeError((completed.stderr or "SAM 2.1 frame extraction failed")[-1200:])

        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            raise RuntimeError("SAM 2.1 could not read the extracted video frames")
        height, width = first_frame.shape[:2]
        inference_state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=True,
        )
        prompts = [{"time": float(selection_time), "box": seed_box, "is_correction": False}]
        prompts.extend(
            {
                "time": float(item.get("timestamp", selection_time)),
                "box": item["box"],
                "is_correction": True,
            }
            for item in corrections
            if isinstance(item, dict) and isinstance(item.get("box"), dict)
        )
        prompt_by_frame = {}
        for prompt in prompts:
            frame_index = int(round((prompt["time"] - float(start)) * tracking_fps))
            frame_index = max(0, min(len(frames) - 1, frame_index))
            prompt_by_frame[frame_index] = {**prompt, "frame_index": frame_index}

        object_id = 1
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        with torch.inference_mode(), autocast:
            for frame_index in sorted(prompt_by_frame):
                prompt = prompt_by_frame[frame_index]
                box = prompt["box"]
                box_prompt = np.asarray(
                    [[float(box["x1"]), float(box["y1"])], [float(box["x2"]), float(box["y2"])]],
                    dtype=np.float32,
                )
                predictor.add_new_points_or_box(
                    inference_state,
                    frame_idx=frame_index,
                    obj_id=object_id,
                    box=box_prompt,
                    clear_old_points=True,
                )

            masks_by_frame = {}
            first_prompt = min(prompt_by_frame)
            for reverse in (False, True):
                for frame_index, object_ids, mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=first_prompt,
                    reverse=reverse,
                ):
                    object_ids = list(object_ids)
                    if object_id not in object_ids:
                        continue
                    object_index = object_ids.index(object_id)
                    masks_by_frame[int(frame_index)] = (
                        mask_logits[object_index] > 0.0
                    ).detach().cpu().numpy().squeeze()

        keyframes = []
        previous_box = None
        for frame_index in range(len(frames)):
            timestamp = float(start) + frame_index / tracking_fps
            mask = masks_by_frame.get(frame_index)
            box = _box_from_mask(mask, width, height) if mask is not None else None
            is_anchor = frame_index in prompt_by_frame
            if is_anchor:
                prompt_box = prompt_by_frame[frame_index]["box"]
                box = {
                    "x1": float(prompt_box["x1"]), "y1": float(prompt_box["y1"]),
                    "x2": float(prompt_box["x2"]), "y2": float(prompt_box["y2"]),
                }
                box["cx"] = (box["x1"] + box["x2"]) / 2.0
                box["cy"] = (box["y1"] + box["y2"]) / 2.0
            if box is None:
                box = previous_box or {
                    "x1": 0.0, "y1": 0.0, "x2": float(width), "y2": float(height),
                    "cx": width / 2.0, "cy": height / 2.0,
                }
                status_name, confidence = "missing", 0.0
            else:
                status_name, confidence = ("anchor", 1.0) if is_anchor else ("tracked", 0.90)
                previous_box = box
            keyframes.append({
                "time": round(min(float(end), timestamp), 3),
                "relative_time": round(timestamp - float(start), 3),
                "box": box,
                "confidence": confidence,
                "detector_confidence": confidence,
                "status": status_name,
                "ambiguous": False,
                "overlap": False,
                "appearance_distance": 0.0,
                "identity_distance": 0.0,
                "motion_distance": 0.0,
                "performer_score": 1.0,
            })

        return {
            "engine": "SAM 2.1 Small · single object mask memory · multi-prompt propagation",
            "tracker_mode": "sam2",
            "source": {
                "width": width, "height": height, "start": round(float(start), 3),
                "end": round(float(end), 3), "duration": round(duration, 3),
            },
            "sample_fps": tracking_fps,
            "keyframes": keyframes,
            "anchors": [
                {
                    "time": round(float(start) + frame_index / tracking_fps, 3),
                    "box": prompt["box"],
                    "is_correction": bool(prompt["is_correction"]),
                }
                for frame_index, prompt in sorted(prompt_by_frame.items())
            ],
            "correction_summary": {
                "applied_count": sum(bool(item["is_correction"]) for item in prompt_by_frame.values()),
                "all_hard_anchors_applied": True,
                "recomputed_full_track": bool(corrections),
                "changed_samples": None,
                "changed_ratio": None,
                "anchors": [
                    {"time": round(float(start) + index / tracking_fps, 3), "status": "anchor", "hard_anchor_applied": True}
                    for index, prompt in sorted(prompt_by_frame.items()) if prompt["is_correction"]
                ],
            },
        }
