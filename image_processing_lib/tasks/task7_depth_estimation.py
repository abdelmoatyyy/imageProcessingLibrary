from .task7_utils import depth_from_two_images


def run(left_path: str, right_path: str, output_dir: str):
    """Same call as original task7.py line 22: depth_from_two_images(left_path, right_path, output_dir)."""
    depth_map, colored_depth = depth_from_two_images(left_path, right_path, output_dir)
    return depth_map, colored_depth
