import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect ORB keypoints/descriptors on satellite images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Directory with raw satellite images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/orb_features",
        help="Directory to save ORB visualizations and descriptors.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=1000,
        help="Maximum number of ORB keypoints per image.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.2,
        help="Pyramid decimation ratio (ORB scaleFactor).",
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        default=8,
        help="Number of pyramid levels (ORB nLevels).",
    )
    return parser.parse_args()


def process_image(
    img_path: Path,
    out_dir: Path,
    detector: cv2.ORB,
) -> None:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    # Draw keypoints on a copy of the original image
    vis = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    stem = img_path.stem
    vis_path = out_dir / f"{stem}_orb.png"
    npz_path = out_dir / f"{stem}_orb.npz"

    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(vis_path), vis)

    # Save keypoints and descriptors
    if descriptors is None:
        # No keypoints found; still save empty placeholders
        np.savez_compressed(
            npz_path,
            keypoints=np.empty((0, 2), dtype=np.float32),
            sizes=np.empty((0,), dtype=np.float32),
            angles=np.empty((0,), dtype=np.float32),
            responses=np.empty((0,), dtype=np.float32),
            octaves=np.empty((0,), dtype=np.int32),
            descriptors=np.empty((0, 32), dtype=np.uint8),
        )
    else:
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)
        angles = np.array([kp.angle for kp in keypoints], dtype=np.float32)
        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
        octaves = np.array([kp.octave for kp in keypoints], dtype=np.int32)

        np.savez_compressed(
            npz_path,
            keypoints=pts,
            sizes=sizes,
            angles=angles,
            responses=responses,
            octaves=octaves,
            descriptors=descriptors,
        )

    print(
        f"{img_path.name}: {len(keypoints)} keypoints "
        f"-> {vis_path.name}, {npz_path.name}"
    )


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    img_paths = sorted(
        [p for p in input_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    )

    if not img_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    orb = cv2.ORB_create(
        nfeatures=args.max_features,
        scaleFactor=args.scale_factor,
        nlevels=args.n_levels,
    )

    print(
        f"Processing {len(img_paths)} images from {input_dir} "
        f"-> saving to {output_dir}"
    )

    for img_path in img_paths:
        process_image(img_path, output_dir, orb)


if __name__ == "__main__":
    main()

