import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from create_dataset import NavigationDataset
from network_training import AzimuthNet
from network_training_attention import AzimuthNetAttention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive navigation simulator.\n"
            "Target tile is fixed, you move the current tile with arrow keys and "
            "the NN predicts the direction."
        )
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="dataset/annotations.csv",
        help="Path to annotations CSV (same format as for training).",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="dataset/satellite_maps",
        help="Directory with original satellite images.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="azimuth_net.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Initial sample index in test split. If -1, choose random.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size in pixels for automatic movement (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=1.0,
        help="Proportional gain for PID controller.",
    )
    parser.add_argument(
        "--ki",
        type=float,
        default=0.0,
        help="Integral gain for PID controller.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=0.1,
        help="Derivative gain for PID controller.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between updates in seconds (default: 0.05 = 20 FPS).",
    )
    parser.add_argument(
        "--use-pid",
        action="store_true",
        help="Use PID controller to smooth movement based on target position.",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.01,
        help="Speed scaling factor: movement speed = base_step * (1 + speed_scale * distance_to_target).",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=None,
        help="Maximum movement speed in pixels per step (None = no limit).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=800,
        help="Display window size (width/height in pixels, default: 800).",
    )
    parser.add_argument(
        "--manual-placement",
        action="store_true",
        help="Enable manual placement mode: click to place current and target tiles before simulation starts.",
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help=(
            "Use AzimuthNetAttention (map encoder with attention pooling) and load its weights "
            "instead of the base AzimuthNet."
        ),
    )
    return parser.parse_args()


def build_normalizer():
    # Same normalization as in training
    return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def tensor_to_bgr(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor image in [0,1], shape (C, H, W), RGB, to OpenCV BGR uint8.
    """
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()  # H x W x C, RGB, [0,1]
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Global variables for manual placement
placement_state = {
    "cur_x": None,
    "cur_y": None,
    "tar_x": None,
    "tar_y": None,
    "clicked": False,
    "button": None,
}


def manual_placement_callback(event, x, y, flags, param):
    """Mouse callback for manual tile placement."""
    global placement_state
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click: place current tile
        placement_state["cur_x"] = x
        placement_state["cur_y"] = y
        placement_state["clicked"] = True
        placement_state["button"] = "current"
        print(f"  Current tile placed at: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click: place target tile
        placement_state["tar_x"] = x
        placement_state["tar_y"] = y
        placement_state["clicked"] = True
        placement_state["button"] = "target"
        print(f"  Target tile placed at: ({x}, {y})")


def manual_placement_mode(
    img_bgr: np.ndarray,
    map_size: int,
    tile_size: int,
    window_size: int,
) -> tuple[float, float, float, float]:
    """
    Interactive manual placement mode.
    Returns: (cur_x, cur_y, tar_x, tar_y)
    """
    global placement_state
    
    # Reset placement state
    placement_state = {
        "cur_x": None,
        "cur_y": None,
        "tar_x": None,
        "tar_y": None,
        "clicked": False,
        "button": None,
    }
    
    half_tile = tile_size // 2
    
    # Resize image for display
    h, w = img_bgr.shape[:2]
    if max(h, w) > window_size:
        scale = window_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_display = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        display_scale = scale
    else:
        img_display = img_bgr.copy()
        display_scale = 1.0
    
    # Scale coordinates back to original image size
    def scale_coords(x, y):
        return x / display_scale, y / display_scale
    
    cv2.namedWindow("Manual Placement Mode", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Placement Mode", manual_placement_callback)
    
    print("\n" + "="*60)
    print("MANUAL PLACEMENT MODE")
    print("="*60)
    print("Instructions:")
    print("  LEFT CLICK  - Place CURRENT tile (green)")
    print("  RIGHT CLICK - Place TARGET tile (yellow)")
    print("  ENTER       - Confirm and start simulation")
    print("  ESC         - Cancel and use default positions")
    print("="*60 + "\n")
    
    cur_x, cur_y = None, None
    tar_x, tar_y = None, None
    
    while True:
        img_vis = img_display.copy()
        
        # Draw current tile if placed
        if cur_x is not None and cur_y is not None:
            cx_disp = int(cur_x * display_scale)
            cy_disp = int(cur_y * display_scale)
            cur_tl = (cx_disp - half_tile, cy_disp - half_tile)
            cur_br = (cx_disp + half_tile, cy_disp + half_tile)
            cv2.rectangle(img_vis, cur_tl, cur_br, color=(0, 255, 0), thickness=2)
            cv2.putText(
                img_vis,
                "CUR",
                (cur_tl[0] + 5, cur_tl[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        
        # Draw target tile if placed
        if tar_x is not None and tar_y is not None:
            tx_disp = int(tar_x * display_scale)
            ty_disp = int(tar_y * display_scale)
            tar_tl = (tx_disp - half_tile, ty_disp - half_tile)
            tar_br = (tx_disp + half_tile, ty_disp + half_tile)
            cv2.rectangle(img_vis, tar_tl, tar_br, color=(0, 255, 255), thickness=2)
            cv2.putText(
                img_vis,
                "TAR",
                (tar_tl[0] + 5, tar_tl[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        
        # Status text
        status = []
        if cur_x is not None and cur_y is not None:
            status.append(f"Current: ({int(cur_x)}, {int(cur_y)})")
        else:
            status.append("Current: Not placed")
        if tar_x is not None and tar_y is not None:
            status.append(f"Target: ({int(tar_x)}, {int(tar_y)})")
        else:
            status.append("Target: Not placed")
        
        cv2.putText(
            img_vis,
            " | ".join(status),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_vis,
            "Left click: Current | Right click: Target | ENTER: Start | ESC: Cancel",
            (10, img_vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        
        cv2.imshow("Manual Placement Mode", img_vis)
        
        # Check for mouse clicks
        if placement_state["clicked"]:
            if placement_state["button"] == "current":
                # Scale back to original coordinates
                x, y = placement_state["cur_x"], placement_state["cur_y"]
                cur_x, cur_y = scale_coords(x, y)
                cur_x = clamp(cur_x, half_tile, map_size - half_tile)
                cur_y = clamp(cur_y, half_tile, map_size - half_tile)
            elif placement_state["button"] == "target":
                # Scale back to original coordinates
                x, y = placement_state["tar_x"], placement_state["tar_y"]
                tar_x, tar_y = scale_coords(x, y)
                tar_x = clamp(tar_x, half_tile, map_size - half_tile)
                tar_y = clamp(tar_y, half_tile, map_size - half_tile)
            placement_state["clicked"] = False
        
        key = cv2.waitKey(1) & 0xFF
        
        # ENTER: confirm and return
        if key == 13:  # Enter key
            if cur_x is not None and cur_y is not None and tar_x is not None and tar_y is not None:
                cv2.destroyWindow("Manual Placement Mode")
                print(f"\nStarting simulation with:")
                print(f"  Current tile: ({int(cur_x)}, {int(cur_y)})")
                print(f"  Target tile: ({int(tar_x)}, {int(tar_y)})\n")
                return cur_x, cur_y, tar_x, tar_y
            else:
                print("  Please place both current and target tiles before starting!")
        
        # ESC: cancel and return None
        if key == 27:  # ESC key
            cv2.destroyWindow("Manual Placement Mode")
            print("\nManual placement cancelled. Using default positions.\n")
            return None, None, None, None


class PIDController:
    """PID controller for smooth navigation movement."""

    def __init__(self, kp: float = 10.0, ki: float = 0.0, kd: float = 0.1):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

    def update(self, target_x: float, target_y: float, current_x: float, current_y: float):
        """
        Compute PID-controlled movement direction.
        Returns (dx, dy) movement vector.
        """
        # Error: direction to target
        error_x = target_x - current_x
        error_y = target_y - current_y
        
        # Normalize error to get direction
        error_norm = np.hypot(error_x, error_y)
        if error_norm < 1e-6:
            return 0.0, 0.0
        
        error_x_norm = error_x / error_norm
        error_y_norm = error_y / error_norm
        
        # Proportional term
        p_x = self.kp * error_x_norm
        p_y = self.kp * error_y_norm
        
        # Integral term
        self.integral_x += error_x_norm
        self.integral_y += error_y_norm
        i_x = self.ki * self.integral_x
        i_y = self.ki * self.integral_y
        
        # Derivative term
        d_x = self.kd * (error_x_norm - self.prev_error_x)
        d_y = self.kd * (error_y_norm - self.prev_error_y)
        self.prev_error_x = error_x_norm
        self.prev_error_y = error_y_norm
        
        # Total PID output
        output_x = p_x + i_x + d_x
        output_y = p_y + i_y + d_y
        
        # Normalize to unit vector
        output_norm = np.hypot(output_x, output_y)
        if output_norm < 1e-6:
            return 0.0, 0.0
        
        return output_x / output_norm, output_y / output_norm

    def reset(self):
        """Reset integral and previous error terms."""
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0


def main() -> None:
    args = parse_args()

    annotations = Path(args.annotations)
    map_dir = Path(args.map_dir)
    device = torch.device(args.device)

    # Dataset WITHOUT augmentation/normalization so that we can visualize easily.
    dataset = NavigationDataset(
        csv_file=str(annotations),
        map_dir=str(map_dir),
        split="test",
        transform=None,
    )

    if len(dataset) == 0:
        raise RuntimeError("Test split is empty. Check annotations CSV and 'split' column.")

    # Choose initial sample
    if args.index < 0 or args.index >= len(dataset):
        idx = torch.randint(0, len(dataset), (1,)).item()
    else:
        idx = args.index

    sat_img_t, _cur_tile_t, _tar_tile_t, labels = dataset[idx]
    # sat_img_t is a tensor in [0,1] of shape (3, map_size, map_size)

    map_size = dataset.map_size
    tile_size = dataset.tile_size
    half_tile = tile_size // 2

    # Initial current and target coordinates in pixels
    cur_xy = labels["cur_xy"] * map_size
    tar_xy = labels["tar_xy"] * map_size

    cur_x, cur_y = cur_xy[0].item(), cur_xy[1].item()
    tar_x, tar_y = tar_xy[0].item(), tar_xy[1].item()

    # Clamp just in case
    cur_x = clamp(cur_x, half_tile, map_size - half_tile)
    cur_y = clamp(cur_y, half_tile, map_size - half_tile)
    tar_x = clamp(tar_x, half_tile, map_size - half_tile)
    tar_y = clamp(tar_y, half_tile, map_size - half_tile)

    # Manual placement mode
    if args.manual_placement:
        base_bgr_temp = tensor_to_bgr(sat_img_t)
        manual_cur_x, manual_cur_y, manual_tar_x, manual_tar_y = manual_placement_mode(
            base_bgr_temp, map_size, tile_size, args.window_size
        )
        if manual_cur_x is not None:
            cur_x, cur_y = manual_cur_x, manual_cur_y
            tar_x, tar_y = manual_tar_x, manual_tar_y

    # Precompute constant tensors
    normalizer = build_normalizer()
    sat_in = normalizer(sat_img_t.clone()).unsqueeze(0).to(device)

    # Tar tile is always centered on fixed target
    ty0 = int(clamp(tar_y - half_tile, 0, map_size - tile_size))
    tx0 = int(clamp(tar_x - half_tile, 0, map_size - tile_size))
    ty1 = ty0 + tile_size
    tx1 = tx0 + tile_size
    tar_tile_t = sat_img_t[:, ty0:ty1, tx0:tx1]
    tar_in = normalizer(tar_tile_t.clone()).unsqueeze(0).to(device)

    # Prepare visualization base image
    base_bgr = tensor_to_bgr(sat_img_t)

    # Load model
    if args.use_attention:
        print("Using attention-based model: AzimuthNetAttention")
        model = AzimuthNetAttention().to(device)
    else:
        print("Using base model: AzimuthNet")
        model = AzimuthNet().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Initialize PID controller
    pid = PIDController(kp=args.kp, ki=args.ki, kd=args.kd)

    print(
        f"Navigation simulator started on sample idx={idx} "
        f"(map_size={map_size}, tile_size={tile_size}).\n"
        f"Mode: Automatic movement (1 pixel per step in NN-predicted direction)\n"
        f"PID gains: Kp={args.kp}, Ki={args.ki}, Kd={args.kd}\n"
        f"Controls:\n"
        f"  SPACE - pause/resume\n"
        f"  R - random new sample\n"
        f"  Q or ESC - quit\n"
    )

    paused = False

    while True:
        # Clamp current center so that tile stays inside the map
        cur_x = clamp(cur_x, half_tile, map_size - half_tile)
        cur_y = clamp(cur_y, half_tile, map_size - half_tile)

        cy0 = int(cur_y - half_tile)
        cx0 = int(cur_x - half_tile)
        cy1 = cy0 + tile_size
        cx1 = cx0 + tile_size

        cur_tile_t = sat_img_t[:, cy0:cy1, cx0:cx1]
        cur_in = normalizer(cur_tile_t.clone()).unsqueeze(0).to(device)

        # NN prediction
        with torch.inference_mode():
            pred = model(sat_in, cur_in, tar_in)[0]  # (2,)

        cos_t, sin_t = pred[0].item(), pred[1].item()

        # Build visualization
        img_bgr = base_bgr.copy()

        # Draw tiles
        cur_top_left = (int(cur_x - half_tile), int(cur_y - half_tile))
        cur_bottom_right = (int(cur_x + half_tile), int(cur_y + half_tile))
        tar_top_left = (int(tar_x - half_tile), int(tar_y - half_tile))
        tar_bottom_right = (int(tar_x + half_tile), int(tar_y + half_tile))

        cv2.rectangle(img_bgr, cur_top_left, cur_bottom_right, color=(0, 255, 0), thickness=2)
        cv2.rectangle(img_bgr, tar_top_left, tar_bottom_right, color=(0, 255, 255), thickness=2)

        # Predicted direction arrow from current position
        arrow_len = map_size * 0.4
        pred_dx = cos_t * arrow_len
        pred_dy = sin_t * arrow_len
        end_point_pred = (int(cur_x + pred_dx), int(cur_y + pred_dy))

        cv2.arrowedLine(
            img_bgr,
            (int(cur_x), int(cur_y)),
            end_point_pred,
            color=(0, 0, 255),
            thickness=3,
            tipLength=0.05,
        )

        # Ground truth direction (for reference, in blue)
        # gt_dx = tar_x - cur_x
        # gt_dy = tar_y - cur_y
        # gt_norm = np.hypot(gt_dx, gt_dy)
        # if gt_norm > 1e-6:
        #     gt_cos = gt_dx / gt_norm
        #     gt_sin = gt_dy / gt_norm
        #     gt_dx_vis = gt_cos * arrow_len
        #     gt_dy_vis = gt_sin * arrow_len
        #     end_point_gt = (int(cur_x + gt_dx_vis), int(cur_y + gt_dy_vis))
        #     cv2.arrowedLine(
        #         img_bgr,
        #         (int(cur_x), int(cur_y)),
        #         end_point_gt,
        #         color=(255, 0, 0),
        #         thickness=2,
        #         tipLength=0.05,
        #     )

        # Text overlay
        cv2.putText(
            img_bgr,
            "CUR",
            (cur_top_left[0] + 5, cur_top_left[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_bgr,
            "TAR",
            (tar_top_left[0] + 5, tar_top_left[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Display status text
        status_text = "PAUSED" if paused else "RUNNING"
        distance_to_target = np.hypot(tar_x - cur_x, tar_y - cur_y)
        dynamic_step = args.step * (1.0 + args.speed_scale * distance_to_target)
        if args.max_speed is not None:
            dynamic_step = min(dynamic_step, args.max_speed)
        cv2.putText(
            img_bgr,
            f"Status: {status_text} | Distance: {distance_to_target:.1f} px | Speed: {dynamic_step:.2f} px/step",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Resize image for display if needed
        h, w = img_bgr.shape[:2]
        if max(h, w) > args.window_size:
            scale = args.window_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_display = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_display = img_bgr

        cv2.imshow("Navigation Simulator", img_display)

        # Non-blocking key check
        key = cv2.waitKey(1) & 0xFF

        # Quit: q or ESC
        if key in (ord("q"), 27):
            break

        # Pause/resume: SPACE
        if key == ord(" "):
            paused = not paused
            if not paused:
                pid.reset()  # Reset PID when resuming

        # Random new sample: r
        if key == ord("r"):
            if len(dataset) > 0:
                idx = torch.randint(0, len(dataset), (1,)).item()
                sat_img_t, _cur_tile_t, _tar_tile_t, labels = dataset[idx]
                cur_xy = labels["cur_xy"] * map_size
                tar_xy = labels["tar_xy"] * map_size
                cur_x, cur_y = cur_xy[0].item(), cur_xy[1].item()
                tar_x, tar_y = tar_xy[0].item(), tar_xy[1].item()
                cur_x = clamp(cur_x, half_tile, map_size - half_tile)
                cur_y = clamp(cur_y, half_tile, map_size - half_tile)
                # Recompute static tensors
                sat_in = normalizer(sat_img_t.clone()).unsqueeze(0).to(device)
                base_bgr = tensor_to_bgr(sat_img_t)
                ty0 = int(clamp(tar_y - half_tile, 0, map_size - tile_size))
                tx0 = int(clamp(tar_x - half_tile, 0, map_size - tile_size))
                ty1 = ty0 + tile_size
                tx1 = tx0 + tile_size
                tar_tile_t = sat_img_t[:, ty0:ty1, tx0:tx1]
                tar_in = normalizer(tar_tile_t.clone()).unsqueeze(0).to(device)
                pid.reset()  # Reset PID for new sample

        # Automatic movement: move in NN-predicted direction
        # Speed scales with distance to target (bigger distance = faster movement)
        if not paused:
            # Calculate distance to target
            distance_to_target = np.hypot(tar_x - cur_x, tar_y - cur_y)
            
            # Calculate dynamic step size based on distance
            # Speed = base_step * (1 + speed_scale * distance)
            dynamic_step = args.step * (1.0 + args.speed_scale * distance_to_target)
            
            # Apply maximum speed limit if specified
            if args.max_speed is not None:
                dynamic_step = min(dynamic_step, args.max_speed)
            
            # Get NN predicted direction (cos_t, sin_t)
            # Move in that direction with dynamic speed
            move_dx = cos_t * dynamic_step
            move_dy = sin_t * dynamic_step

            # If PID is enabled, blend NN direction with PID correction toward target
            if args.use_pid:
                pid_dx, pid_dy = pid.update(tar_x, tar_y, cur_x, cur_y)
                # Blend: 70% NN direction, 30% PID correction
                move_dx = 0.7 * move_dx + 0.3 * pid_dx * dynamic_step
                move_dy = 0.7 * move_dy + 0.3 * pid_dy * dynamic_step
                # Normalize to maintain step size
                move_norm = np.hypot(move_dx, move_dy)
                if move_norm > 1e-6:
                    move_dx = move_dx / move_norm * dynamic_step
                    move_dy = move_dy / move_norm * dynamic_step

            cur_x += move_dx
            cur_y += move_dy

        # Add delay to control update rate
        time.sleep(args.delay)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


