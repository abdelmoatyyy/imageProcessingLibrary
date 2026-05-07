import cv2
import numpy as np
from PIL import Image
import os


def extract_frames_from_gif(gif_path):
    gif = Image.open(gif_path)
    frames = []
    for frame_idx in range(gif.n_frames):
        gif.seek(frame_idx)
        frame = np.array(gif.convert('RGB'))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
    return frames


def rgb_to_grayscale(img):
    if img is None:
        raise ValueError("Input image to rgb_to_grayscale cannot be None.")
    return (0.2989 * img[:, :, 2] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 0]).astype(np.uint8)


def compute_sad(block1, block2):
    return np.sum(np.abs(block1.astype(np.int16) - block2.astype(np.int16)))


def fast_block_matching(left, right, block_size=9, max_disparity=64):
    h, w = left.shape
    half_block = block_size // 2
    disparity_map = np.zeros((h, w), dtype=np.float32)

    for y in range(half_block, h - half_block):
        y_start = y - half_block
        y_end = y + half_block + 1
        for x in range(half_block, w - half_block):
            x_start = x - half_block
            x_end = x + half_block + 1
            left_block = left[y_start:y_end, x_start:x_end]
            best_disparity = 0
            best_cost = float('inf')
            min_x_right = max(half_block, x - max_disparity)
            for x_right in range(min_x_right, x - half_block + 1):
                right_block = right[y_start:y_end, x_right - half_block:x_right + half_block + 1]
                if right_block.shape == left_block.shape:
                    diff = left_block.astype(np.int16) - right_block.astype(np.int16)
                    cost = np.sum(np.abs(diff))
                    if cost < best_cost:
                        best_cost = cost
                        best_disparity = x - x_right
            disparity_map[y, x] = best_disparity

    return disparity_map


def normalize_disparity(disparity):
    valid = disparity[disparity > 0]
    if len(valid) > 0:
        min_val = np.percentile(valid, 2)
        max_val = np.percentile(valid, 98)
        disparity_clipped = np.clip(disparity, min_val, max_val)
        normalized = ((disparity_clipped - min_val) / (max_val - min_val + 1e-6)) * 255
    else:
        normalized = np.zeros_like(disparity)
    return normalized.astype(np.uint8)


def apply_jet_colormap(depth_map):
    h, w = depth_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = depth_map[y, x] / 255.0
            if v <= 0.125:
                r, g, b = 0, 0, 0.5 + v * 4
            elif v <= 0.375:
                r, g, b = 0, v * 4 - 0.5, 1
            elif v <= 0.625:
                r, g, b = v * 4 - 1.5, 1, 1 - (v * 4 - 1.5)
            elif v <= 0.875:
                r, g, b = 1, 1 - (v * 4 - 2.5), 0
            else:
                r, g, b = 1 - (v * 4 - 3.5), 0, 0
            colored[y, x] = [int(b * 255), int(g * 255), int(r * 255)]
    return colored


def save_image(image, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"[Saved] {filepath}")
    return filepath


def compute_depth_from_scratch(left, right, output_dir="output"):
    print("Converting to grayscale...")
    left_gray = rgb_to_grayscale(left)
    right_gray = rgb_to_grayscale(right)

    save_image(left, output_dir, "original_left.png")

    print("Computing disparity map...")
    disparity = fast_block_matching(left_gray, right_gray)

    print("Normalizing...")
    depth_map = normalize_disparity(disparity)

    print("Applying colormap...")
    colored_depth = apply_jet_colormap(depth_map)

    save_image(depth_map, output_dir, "depth_map_normalized.png")
    save_image(colored_depth, output_dir, "depth_map_colored.png")

    return depth_map, colored_depth


def depth_from_gif(gif_path, frame_left=0, frame_right=1, output_dir="output"):
    print("Extracting frames from GIF...")
    frames = extract_frames_from_gif(gif_path)
    if len(frames) < 2:
        raise ValueError("GIF needs at least 2 frames")
    left_img = frames[frame_left]
    right_img = frames[frame_right]
    depth_map, colored_depth = compute_depth_from_scratch(left_img, right_img, output_dir)
    return depth_map, colored_depth


def depth_from_two_images(left_path, right_path, output_dir="output"):
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None:
        raise FileNotFoundError(f"Left image not found at {left_path}")
    if right is None:
        raise FileNotFoundError(f"Right image not found at {right_path}")
    depth_map, colored_depth = compute_depth_from_scratch(left, right, output_dir)
    return depth_map, colored_depth
