import cv2
import os
from utils import depth_from_two_images

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

print("\n STEREO DEPTH ESTIMATION - FROM SCRATCH IMPLEMENTATION\n")

left_path = "images/left.ppm"
right_path = "images/right.ppm"

if not os.path.exists(left_path):
    print(f"Error: {left_path} not found")
    exit(1)
if not os.path.exists(right_path):
    print(f"Error: {right_path} not found")
    exit(1)

print(f"Loading images from: {left_path}, {right_path}\n")

depth_map, colored_depth = depth_from_two_images(left_path, right_path, output_dir)

print(f"\n COMPLETE - All outputs saved to '{output_dir}/' folder\n")

left = cv2.imread(left_path)

cv2.imshow("Original Image", left)
cv2.imshow("Depth Map", depth_map)
cv2.imshow("Depth Map Colored", colored_depth)

cv2.waitKey(0)
cv2.destroyAllWindows()