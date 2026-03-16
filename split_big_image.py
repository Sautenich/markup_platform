import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a large image into overlapping 1000x1000 tiles.\n"
            "Tiles are generated from the bottom‑left corner to the top‑right corner,\n"
            "with 100‑pixel overlap between neighboring tiles."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="big_sample5_17.tif",
        help="Path to the big source image (default: big_sample5_17.tif).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="big_sample_tiles",
        help="Directory to save 1000x1000 tiles.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1000,
        help="Tile size in pixels (square, default: 1000).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap size in pixels between neighboring tiles (default: 100).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="Output image format/extension (e.g. png, jpg, tif).",
    )
    return parser.parse_args()


def compute_tile_origins(width: int, height: int, tile_size: int, overlap: int):
    """
    Compute (x, y) origins for tiles that:
      - are exactly tile_size x tile_size
      - overlap neighbors by 'overlap' pixels
      - cover the full image, aligning the last tiles with the right/top borders if needed
      - are ordered from bottom‑left to top‑right (scan bottom row left->right, then up)
    """
    stride = tile_size - overlap
    assert stride > 0, "stride must be positive (tile_size must be > overlap)."

    # Compute candidate starts in x direction
    x_starts = []
    x = 0
    while x + tile_size <= width:
        x_starts.append(x)
        x += stride
    # Ensure last tile touches the right edge
    if x_starts:
        last_x = x_starts[-1]
        if last_x + tile_size < width:
            x_starts.append(width - tile_size)
    else:
        x_starts.append(0)

    # Compute candidate starts in y direction (top‑down first)
    y_starts = []
    y = 0
    while y + tile_size <= height:
        y_starts.append(y)
        y += stride
    # Ensure last tile touches the bottom edge
    if y_starts:
        last_y = y_starts[-1]
        if last_y + tile_size < height:
            y_starts.append(height - tile_size)
    else:
        y_starts.append(0)

    # We need order from bottom‑left to top‑right.
    # PIL coordinates: (0,0) is top‑left, y increases downward.
    # "Bottom" means larger y values. So iterate y from high to low.
    origins = []
    for row_idx, y0 in enumerate(sorted(y_starts, reverse=True)):
        for col_idx, x0 in enumerate(sorted(x_starts)):
            origins.append((x0, y0, col_idx, row_idx))

    return origins


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    big_img = Image.open(input_path).convert("RGB")
    width, height = big_img.size
    tile_size = args.tile_size
    overlap = args.overlap

    print(
        f"Splitting {input_path} (size: {width}x{height}) into {tile_size}x{tile_size} tiles\n"
        f"with {overlap}px overlap (stride = {tile_size - overlap})."
    )

    origins = compute_tile_origins(width, height, tile_size, overlap)
    print(f"Total tiles: {len(origins)}")

    for x0, y0, col_idx, row_idx in origins:
        x1 = x0 + tile_size
        y1 = y0 + tile_size

        tile = big_img.crop((x0, y0, x1, y1))

        # Naming: tile_row{row}_col{col}_x{x0}_y{y0}.ext
        out_name = f"tile_r{row_idx:03d}_c{col_idx:03d}_x{x0}_y{y0}.{args.format}"
        out_path = output_dir / out_name
        tile.save(out_path)

    print(f"Done. Tiles saved to: {output_dir}")


if __name__ == "__main__":
    main()


