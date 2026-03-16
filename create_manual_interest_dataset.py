import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively create a small dataset from big_sample5_17.tif.\n"
            "1) You manually select a 200x200 TARGET square on the big image.\n"
            "2) Around this square the script samples random 2000x2000 crops that CONTAIN this square.\n"
            "3) For each crop, it creates a row in annotations CSV compatible with NavigationDataset.\n"
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the large source image.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations_manual_interest.csv",
        help="Path to CSV where annotations will be saved/appended.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of 2000x2000 crops to generate around the selected 200x200 target square.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=2000,
        help="Map crop size in pixels (e.g., 2000 for larger context).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size in pixels (NavigationDataset assumes 200).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1200,
        help="Maximum display window size (longest side in pixels).",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def select_interest_square(
    big_img_bgr: np.ndarray,
    img_w: int,
    img_h: int,
    tile_size: int,
    window_size: int,
) -> tuple[float, float] | tuple[None, None]:
    """
    Interactive selection of a 200x200 (tile_size x tile_size) TARGET square.
    Returns center coordinates (cx, cy) in original image space.
    """
    half_tile = tile_size // 2

    # Resize for display if needed
    h, w = big_img_bgr.shape[:2]
    if max(h, w) > window_size:
        scale = window_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        disp = cv2.resize(big_img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        display_scale = scale
    else:
        disp = big_img_bgr.copy()
        display_scale = 1.0

    state = {"cx": None, "cy": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert from display coords to original image coords
            cx = x / display_scale
            cy = y / display_scale
            # Clamp so the full 200x200 tile stays inside image
            cx = clamp(cx, half_tile, img_w - half_tile)
            cy = clamp(cy, half_tile, img_h - half_tile)
            state["cx"], state["cy"] = cx, cy
            print(f"Selected interest center at ({int(cx)}, {int(cy)})")

    cv2.namedWindow("Select 200x200 Interest Square", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select 200x200 Interest Square", on_mouse)

    print("\n=== Manual Selection of 200x200 TARGET Square ===")
    print("LEFT CLICK  - set center of 200x200 target square")
    print("U           - undo last selection (clear)")
    print("ENTER       - confirm selection")
    print("ESC         - cancel\n")

    while True:
        vis = disp.copy()

        # Draw selected square if any
        if state["cx"] is not None and state["cy"] is not None:
            cx_disp = int(state["cx"] * display_scale)
            cy_disp = int(state["cy"] * display_scale)
            tl = (cx_disp - int(half_tile * display_scale), cy_disp - int(half_tile * display_scale))
            br = (cx_disp + int(half_tile * display_scale), cy_disp + int(half_tile * display_scale))
            cv2.rectangle(vis, tl, br, (0, 255, 0), 2)
            cv2.putText(
                vis,
                "INTEREST",
                (tl[0] + 5, tl[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Select 200x200 Interest Square", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow("Select 200x200 Interest Square")
            print("Selection cancelled.")
            return None, None
        if key in (ord("u"), ord("U")):
            # Undo last selection (clear)
            if state["cx"] is not None or state["cy"] is not None:
                print("Last selection cleared.")
            state["cx"], state["cy"] = None, None
        if key == 13:  # ENTER
            if state["cx"] is not None and state["cy"] is not None:
                cv2.destroyWindow("Select 200x200 Interest Square")
                return state["cx"], state["cy"]
            else:
                print("Please select a square (left click) before pressing ENTER.")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Image is huge; disable decompression bomb limit for this tool
    Image.MAX_IMAGE_PIXELS = None

    big_img = Image.open(input_path).convert("RGB")
    img_w, img_h = big_img.size

    map_size = args.map_size
    tile_size = args.tile_size
    half_tile = tile_size // 2

    if img_w < map_size or img_h < map_size:
        raise ValueError(
            f"Image {input_path} is too small for map_size={map_size}: size={img_w}x{img_h}"
        )

    # Prepare BGR image for OpenCV display (used across multiple selections)
    big_np = np.array(big_img)
    big_bgr = cv2.cvtColor(big_np, cv2.COLOR_RGB2BGR)

    # Prepare annotations CSV (append if exists, else create) ONCE,
    # and allow multiple manual selections in a single persistent window.
    annotations_path = Path(args.annotations)
    if annotations_path.exists():
        df = pd.read_csv(annotations_path)
        if "sample_id" in df.columns:
            next_sample_id = int(df["sample_id"].max()) + 1
        else:
            next_sample_id = 0
        records = df.to_dict("records")
    else:
        records = []
        next_sample_id = 0

    # map_id for this big image (without extension)
    map_id = input_path.stem

    print(
        "\nYou can make SEVERAL selections in one session.\n"
        "Each ENTER confirms the current 200x200 TARGET square and generates a batch of crops.\n"
        "Use U to clear the current square, ESC to finish and save annotations.\n"
    )

    # One RNG reused for all selections (for reproducibility)
    rng = np.random.default_rng(42)

    # Setup a single persistent window for all selections
    # Resize for display if needed
    h_disp, w_disp = big_bgr.shape[:2]
    if max(h_disp, w_disp) > args.window_size:
        scale = args.window_size / max(h_disp, w_disp)
        new_w, new_h = int(w_disp * scale), int(h_disp * scale)
        disp_base = cv2.resize(big_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        display_scale = scale
    else:
        disp_base = big_bgr.copy()
        display_scale = 1.0

    selection_state = {"cx": None, "cy": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert from display coords to original image coords
            cx = x / display_scale
            cy = y / display_scale
            # Clamp so the full 200x200 tile stays inside image
            cx = clamp(cx, half_tile, img_w - half_tile)
            cy = clamp(cy, half_tile, img_h - half_tile)
            selection_state["cx"], selection_state["cy"] = cx, cy
            print(f"Selected interest center at ({int(cx)}, {int(cy)})")

    cv2.namedWindow("Manual Interest Selection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Interest Selection", on_mouse)

    print("Controls in the window:")
    print("  LEFT CLICK  - set center of 200x200 TARGET square")
    print("  U           - undo last selection (clear)")
    print("  ENTER       - confirm selection and generate crops")
    print("  ESC         - finish session and save annotations\n")

    while True:
        vis = disp_base.copy()

        # Draw current selection if any
        if selection_state["cx"] is not None and selection_state["cy"] is not None:
            cx_disp = int(selection_state["cx"] * display_scale)
            cy_disp = int(selection_state["cy"] * display_scale)
            tl = (
                cx_disp - int(half_tile * display_scale),
                cy_disp - int(half_tile * display_scale),
            )
            br = (
                cx_disp + int(half_tile * display_scale),
                cy_disp + int(half_tile * display_scale),
            )
            cv2.rectangle(vis, tl, br, (0, 255, 0), 2)
            cv2.putText(
                vis,
                "INTEREST",
                (tl[0] + 5, tl[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Manual Interest Selection", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            print("Finishing session.")
            break
        if key in (ord("u"), ord("U")):
            # Clear current selection
            if selection_state["cx"] is not None or selection_state["cy"] is not None:
                print("Last selection cleared.")
            selection_state["cx"], selection_state["cy"] = None, None
            continue
        if key == 13:  # ENTER
            cx = selection_state["cx"]
            cy = selection_state["cy"]
            if cx is None or cy is None:
                print("Please select a square (left click) before pressing ENTER.")
                continue

            # 2) Compute valid range of map crops that contain this 200x200 target square
            min_x0 = int(cx - (map_size - half_tile))
            max_x0 = int(cx - half_tile)
            min_y0 = int(cy - (map_size - half_tile))
            max_y0 = int(cy - half_tile)

            # Clamp so crop stays inside image
            min_x0 = clamp(min_x0, 0, img_w - map_size)
            max_x0 = clamp(max_x0, 0, img_w - map_size)
            min_y0 = clamp(min_y0, 0, img_h - map_size)
            max_y0 = clamp(max_y0, 0, img_h - map_size)

            if min_x0 > max_x0 or min_y0 > max_y0:
                print("No valid crops can contain this 200x200 target square. Try another place.")
                continue

            print(
                f"\nGenerating {args.samples} crops of size {map_size}x{map_size} "
                f"that contain the 200x200 TARGET square at ({int(cx)}, {int(cy)})...\n"
            )

            for _ in range(args.samples):
                x0 = int(rng.integers(min_x0, int(max_x0) + 1))
                y0 = int(rng.integers(min_y0, int(max_y0) + 1))

                # Map crop is [x0, x0+map_size) x [y0, y0+map_size)
                # Target tile center is the selected interest center relative to this crop
                tar_cx = float(cx - x0)
                tar_cy = float(cy - y0)

                # Current tile center: random inside crop (full tile fits)
                cur_cx = float(
                    rng.integers(half_tile, map_size - half_tile + 1)
                )
                cur_cy = float(
                    rng.integers(half_tile, map_size - half_tile + 1)
                )

                dx = tar_cx - cur_cx
                dy = tar_cy - cur_cy

                records.append(
                    {
                        "sample_id": next_sample_id,
                        "map_id": map_id,
                        "map_crop_x": int(x0),
                        "map_crop_y": int(y0),
                        "cur_tile_x": cur_cx,
                        "cur_tile_y": cur_cy,
                        "tar_tile_x": tar_cx,
                        "tar_tile_y": tar_cy,
                        "vector_dx": float(dx),
                        "vector_dy": float(dy),
                        "split": "manual_interest",
                    }
                )
                next_sample_id += 1

            print(f"  Added {args.samples} samples for TARGET square at ({int(cx)}, {int(cy)}).")
            # Optionally clear selection after generation
            selection_state["cx"], selection_state["cy"] = None, None

    # 3) Save updated CSV after all selections
    if records:
        out_df = pd.DataFrame.from_records(records)
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(annotations_path, index=False)

        print(
            f"\nSaved annotations to {annotations_path}\n"
            f"Total rows now: {len(out_df)}\n"
            "Use with NavigationDataset / network_training*.py by pointing --annotations "
            f"to this file and --satellite-dir/--map-dir to the directory that contains {input_path.name}."
        )
    else:
        print("\nNo samples were generated (no confirmed selections). Nothing was written.")


if __name__ == "__main__":
    main()


