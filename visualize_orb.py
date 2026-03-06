import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize ORB keypoints on satellite images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Directory with satellite images.",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="Specific image to visualize (e.g., '1_square.jpg'). If not provided, processes all images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="orb_visualizations",
        help="Directory to save ORB visualizations.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=1000,
        help="Maximum number of ORB keypoints to detect.",
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
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the visualization window (requires GUI).",
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["rich", "simple", "circles"],
        default="rich",
        help="Visualization style: 'rich' (with orientation/size), 'simple' (just points), 'circles' (with circles).",
    )
    return parser.parse_args()


def visualize_orb_features(
    img_path: Path,
    output_dir: Path,
    detector: cv2.ORB,
    style: str = "rich",
    show: bool = False,
) -> None:
    """Detect and visualize ORB features on a satellite image."""
    # Read image
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    # Convert to grayscale for ORB detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    print(f"{img_path.name}: Detected {len(keypoints)} ORB keypoints")

    # Create visualization based on style
    if style == "rich":
        # Draw keypoints with orientation and size information
        vis = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            color=(0, 255, 0),  # Green
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
    elif style == "simple":
        # Draw simple keypoints (just points)
        vis = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            color=(0, 255, 0),  # Green
            flags=0,
        )
    elif style == "circles":
        # Draw keypoints as circles with size information
        vis = img.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2) if kp.size > 0 else 5
            cv2.circle(vis, (x, y), radius, (0, 255, 0), 2)
            # Draw orientation line if available
            if kp.angle >= 0:
                angle_rad = np.radians(kp.angle)
                end_x = int(x + radius * np.cos(angle_rad))
                end_y = int(y + radius * np.sin(angle_rad))
                cv2.line(vis, (x, y), (end_x, end_y), (0, 255, 0), 1)

    # Save visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    output_path = output_dir / f"{stem}_orb_{style}.png"
    cv2.imwrite(str(output_path), vis)
    print(f"  Saved visualization to: {output_path}")

    # Display if requested
    if show:
        # Resize if image is too large for display
        h, w = vis.shape[:2]
        max_display_size = 1200
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            vis_display = cv2.resize(vis, (new_w, new_h))
        else:
            vis_display = vis

        cv2.imshow(f"ORB Features - {img_path.name}", vis_display)
        print("  Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=args.max_features,
        scaleFactor=args.scale_factor,
        nlevels=args.n_levels,
    )

    # Find images to process
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    
    if args.image_name:
        # Process specific image
        img_path = input_dir / args.image_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if img_path.suffix.lower() not in exts:
            raise ValueError(f"Unsupported image format: {img_path.suffix}")
        img_paths = [img_path]
    else:
        # Process all images
        img_paths = sorted(
            [p for p in input_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
        )

    if not img_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    print(
        f"Processing {len(img_paths)} image(s) from {input_dir}\n"
        f"Style: {args.style}\n"
        f"Max features: {args.max_features}\n"
        f"Output directory: {output_dir}\n"
    )

    for img_path in img_paths:
        visualize_orb_features(
            img_path,
            output_dir,
            orb,
            style=args.style,
            show=args.show,
        )

    print(f"\nDone! Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

