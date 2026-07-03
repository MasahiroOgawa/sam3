#!/usr/bin/env python
"""Run SAM 3 text-prompted segmentation on an arbitrary video and write an
overlay result video.

Example:
    scripts/run_sam3 --prompt "powerline wire" \
        --input-video /home/mas/data/sensyn/wire_sag/202606/powerline_sag.MOV

Frames are sampled from the input video (``--sample-fps``, capped by
``--max-frames`` so a full-length clip does not exhaust GPU memory), the text
prompt is applied on the first sampled frame and propagated through the rest,
and a mask overlay video is saved under ``--result-dir`` (default ``result/``).
"""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

# Keep torch.compile artifacts off shared /tmp roots and out of the repo.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), f"torchinductor_{os.getlogin()}"),
)
# Reduce CUDA fragmentation OOMs on smaller (e.g. 12 GB laptop) GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch

# BGR colors (OpenCV order), one per tracked object id.
MASK_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (255, 0, 128),
    (255, 128, 0),
    (128, 0, 255),
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def extract_frames(
    video_path: str, frame_dir: str, sample_fps: float, max_frames: int, start: float
) -> list[str]:
    """Sample frames from the video into frame_dir as zero-padded JPEGs."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video_path, "-vf", f"fps={sample_fps}"]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-qscale:v", "2", os.path.join(frame_dir, "%06d.jpg")]
    subprocess.run(cmd, check=True)
    frames = sorted(Path(frame_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg extracted no frames from {video_path}")
    return [str(p) for p in frames]


def masks_from_response(response: dict) -> dict[int, np.ndarray]:
    """Extract {obj_id: HxW bool mask} from a propagate_in_video response."""
    outputs = response.get("outputs", {})
    obj_ids = outputs.get("out_obj_ids", [])
    binary_masks = outputs.get("out_binary_masks")
    if binary_masks is None:
        return {}
    if isinstance(obj_ids, torch.Tensor):
        obj_ids = obj_ids.cpu().numpy()
    if isinstance(binary_masks, torch.Tensor):
        binary_masks = binary_masks.cpu().numpy()
    masks = {}
    for i, oid in enumerate(obj_ids):
        m = binary_masks[i]
        if m.ndim == 3:
            m = m[0]
        masks[int(oid)] = m.astype(bool)
    return masks


def draw_overlay(
    frame_bgr: np.ndarray, masks: dict[int, np.ndarray], title: str, alpha: float = 0.5
) -> np.ndarray:
    out = frame_bgr.copy()
    for obj_id, mask in sorted(masks.items()):
        color = MASK_COLORS[obj_id % len(MASK_COLORS)]
        out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        # Outline improves visibility of thin structures like wires.
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)
    cv2.putText(
        out,
        title,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM 3 text-prompted segmentation on a video."
    )
    parser.add_argument(
        "--prompt", required=True, help="Text prompt, e.g. 'powerline wire'"
    )
    parser.add_argument(
        "--input-video", required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--result-dir", default="result", help="Output directory (default: result/)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output video path (overrides --result-dir)",
    )
    # Default "sam3": its init_state is compatible with the start_session API.
    # "sam3.1" (multiplex) currently rejects start_session's offload_state_to_cpu.
    parser.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    parser.add_argument(
        "--sample-fps", type=float, default=5.0, help="Frames sampled per second"
    )
    parser.add_argument(
        "--max-frames", type=int, default=60, help="Max frames to process (0 = all)"
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start offset in seconds"
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Mask overlay opacity")
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Use Flash Attention 3 (must be installed)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_video):
        raise SystemExit(f"Input video not found: {args.input_video}")

    if args.output:
        out_path = args.output
    else:
        stem = Path(args.input_video).stem
        out_path = os.path.join(args.result_dir, f"{stem}_{slugify(args.prompt)}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    frame_dir = tempfile.mkdtemp(prefix="sam3_frames_")
    try:
        print(
            f"[1/4] Sampling frames from {args.input_video} "
            f"(fps={args.sample_fps}, max={args.max_frames}, start={args.start}s)"
        )
        frame_paths = extract_frames(
            args.input_video, frame_dir, args.sample_fps, args.max_frames, args.start
        )
        h, w = cv2.imread(frame_paths[0]).shape[:2]
        print(f"      {len(frame_paths)} frames at {w}x{h}")

        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        from sam3 import build_sam3_predictor

        print(f"[2/4] Building {args.version} predictor (auto-downloads checkpoint)")
        model = build_sam3_predictor(
            version=args.version,
            compile=False,
            async_loading_frames=False,
            use_fa3=args.use_fa3,
        )

        session_id = model.handle_request(
            {
                "type": "start_session",
                "resource_path": frame_dir,
                # Offload frames and tracking state to CPU to bound GPU memory.
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            }
        )["session_id"]
        model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": args.prompt,
            }
        )

        print(f"[3/4] Propagating prompt '{args.prompt}' through the video")
        masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
        for response in model.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            frame_idx = response.get("frame_index")
            if frame_idx is not None:
                masks_by_frame[frame_idx] = masks_from_response(response)
        torch.cuda.synchronize()

        frames_with_obj = sum(1 for m in masks_by_frame.values() if m)
        max_objs = max((len(m) for m in masks_by_frame.values()), default=0)
        print(
            f"      objects on >=1 frame in {frames_with_obj}/{len(frame_paths)} frames "
            f"(max {max_objs} simultaneous)"
        )

        print(f"[4/4] Writing overlay video to {out_path}")
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.sample_fps, (w, h)
        )
        title = f"SAM3 [{args.version}] | {args.prompt}"
        for idx, fp in enumerate(frame_paths):
            frame = cv2.imread(fp)
            writer.write(
                draw_overlay(frame, masks_by_frame.get(idx, {}), title, args.alpha)
            )
        writer.release()
    finally:
        cv2.destroyAllWindows()
        __import__("shutil").rmtree(frame_dir, ignore_errors=True)

    print(f"Done. Result video: {out_path}")


if __name__ == "__main__":
    main()
