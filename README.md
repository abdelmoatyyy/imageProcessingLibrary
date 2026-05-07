# Image Processing Library + GUI

A unified Python library and Streamlit GUI that combines all tasks into one workflow while preserving each task's original processing logic.

## Included Tasks

1. Selective Object Enhancement (Color-Based Editing Tool)
2. Comparative Study of Sharpening Pipelines for X-Ray Images
3. Intelligent Auto Image Enhancement System
4. Document Cleaning System
5. Panorama Image Stitching
6. Transformed Object Recognition
7. Depth Approximation from Two Images
8. High Dynamic Range (HDR) Imaging

## Project Structure

- `app.py` - Streamlit GUI entry point
- `image_processing_lib/` - Library package
  - `tasks/` - Task modules (`task1` ... `task8`)
  - `io_utils.py` - Shared output saving helper
- `test-images/taskN/` - Sample inputs per task
- `output-images/taskN/` - Saved outputs per task

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the GUI

```bash
streamlit run app.py
```

Then in the app:
- Select a task from the dropdown.
- Upload image(s) from `test-images/taskN/`.
- Click **Run**.
- View result(s) in the GUI.
- Outputs are saved automatically to `output-images/taskN/`.

## Input Count Per Task

- Task 1: 1 image
- Task 2: 1 image
- Task 3: 1 image
- Task 4: 1 image
- Task 5: 2 images
- Task 6: 2 images
- Task 7: 2 images (left + right)
- Task 8: 3 images (dark + normal + bright)

## Notes

- Task 3 displays detected image problem and enhancement strategy in the GUI.
- Task 7 writes depth outputs (`original_left`, normalized depth, colored depth) to `output-images/task7/`.
- Task 8 expects three exposures in order: dark, normal, bright.

## Git Ignore

`myenv/` is ignored via `.gitignore`.
