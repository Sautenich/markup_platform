import numpy as np
from PIL import Image

def create_rgb_gradient(width=1000, height=1000, out_path="rgb_gradient.png"):
    # Create coordinate grid
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)

    # R: horizontal gradient, G: vertical gradient, B: constant or some function
    r = xv
    g = yv
    b = np.full_like(r, 128, dtype=np.uint8)  # mid-level blue

    # Stack into (H, W, 3)
    rgb = np.stack([r, g, b], axis=2)

    img = Image.fromarray(rgb, mode="RGB")
    img.save(out_path)
    print(f"Saved gradient image to {out_path}")

if __name__ == "__main__":
    create_rgb_gradient()