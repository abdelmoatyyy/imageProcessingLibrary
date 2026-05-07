"""
Streamlit GUI for image processing tasks (1, 2, 3, 4, 5, 6, 7, 8).
Run: streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from image_processing_lib.io_utils import save_output
from image_processing_lib.tasks import task1_selective_enhancement as t1
from image_processing_lib.tasks import task2_xray_sharpen as t2
from image_processing_lib.tasks import task3_intelligent_enhance as t3
from image_processing_lib.tasks import task4_document_clean as t4
from image_processing_lib.tasks import task5_panorama_stitching as t5
from image_processing_lib.tasks import task6_object_recognition as t6
from image_processing_lib.tasks import task7_depth_estimation as t7
from image_processing_lib.tasks import task8_hdr as t8

PROJECT_ROOT = Path(__file__).resolve().parent

TASK_CHOICES = [
    ("1. Selective Object Enhancement (Color-Based Editing Tool)", "task1"),
    ("2. Comparative Study of Sharpening Pipelines for X-Ray Images", "task2"),
    ("3. Intelligent Auto Image Enhancement System", "task3"),
    ("4. Document Cleaning System", "task4"),
    ("5. Panorama Image Stitching", "task5"),
    ("6. Transformed Object Recognition", "task6"),
    ("7. Depth Approximation from Two Images", "task7"),
    ("8. High Dynamic Range (HDR) Imaging", "task8"),
]


def _decode_upload_bgr(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _decode_upload_gray(file_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def _rgb_for_display(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(page_title="Image Processing Library", layout="wide")
    st.title("Image Processing Library")

    labels = [x[0] for x in TASK_CHOICES]
    keys = [x[1] for x in TASK_CHOICES]
    idx = st.selectbox("Choose a task", range(len(labels)), format_func=lambda i: labels[i])
    task_id = keys[idx]

    st.sidebar.markdown(
        "**Tip:** In the file picker, browse to "
        f"`{PROJECT_ROOT / 'test-images' / task_id}` "
        "to use the provided sample images."
    )

    # --- Uploaders per task ---
    f1 = f2 = f3 = f4 = None
    f5a = f5b = None
    f6a = f6b = None
    f7l = f7r = None
    f8d = f8n = f8b = None

    if task_id == "task1":
        f1 = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    elif task_id == "task2":
        f2 = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
    elif task_id == "task3":
        f3 = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    elif task_id == "task4":
        f4 = st.file_uploader("Upload document image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    elif task_id == "task5":
        c1, c2 = st.columns(2)
        with c1:
            f5a = st.file_uploader("Panorama image 1", type=["jpg", "jpeg", "png", "bmp", "webp"], key="t5a")
        with c2:
            f5b = st.file_uploader("Panorama image 2", type=["jpg", "jpeg", "png", "bmp", "webp"], key="t5b")
    elif task_id == "task6":
        c1, c2 = st.columns(2)
        with c1:
            f6a = st.file_uploader("Image A", type=["jpg", "jpeg", "png", "bmp", "webp"], key="t6a")
        with c2:
            f6b = st.file_uploader("Image B", type=["jpg", "jpeg", "png", "bmp", "webp"], key="t6b")
    elif task_id == "task7":
        c1, c2 = st.columns(2)
        with c1:
            f7l = st.file_uploader("Left image", type=["ppm", "jpg", "jpeg", "png", "bmp", "webp"], key="t7l")
        with c2:
            f7r = st.file_uploader("Right image", type=["ppm", "jpg", "jpeg", "png", "bmp", "webp"], key="t7r")
    elif task_id == "task8":
        st.caption("Upload in order: dark (under-exposed), normal, bright (over-exposed).")
        f8d = st.file_uploader("Dark exposure", type=["jpg", "jpeg", "png", "bmp", "webp"], key="t8d")
        f8n = st.file_uploader("Normal exposure", type=["jpg", "jpeg", "png", "bmp", "webp"], key="t8n")
        f8b = st.file_uploader("Bright exposure", type=["jpg", "jpeg", "png", "bmp", "webp"], key="t8b")

    run = st.button("Run", type="primary")
    if not run:
        return

    out_paths: list[str] = []

    try:
        if task_id == "task1":
            if f1 is None:
                st.warning("Please upload an image.")
                return
            bgr = _decode_upload_bgr(f1.getvalue())
            if bgr is None:
                st.error("Could not decode image.")
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result, red_m, yel_m, grn_m = t1.selective_enhancement(rgb)
            st.subheader("Output")
            c0, c1 = st.columns(2)
            with c0:
                st.image(rgb, caption="Original", use_container_width=True)
            with c1:
                st.image(result, caption="Enhanced result", use_container_width=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.image(red_m, caption="Red mask", use_container_width=True)
            with m2:
                st.image(yel_m, caption="Yellow mask", use_container_width=True)
            with m3:
                st.image(grn_m, caption="Green mask", use_container_width=True)
            out_paths.append(save_output(cv2.cvtColor(result, cv2.COLOR_RGB2BGR), "task1", "result.png"))
            out_paths.append(save_output(red_m, "task1", "red_mask.png"))
            out_paths.append(save_output(yel_m, "task1", "yellow_mask.png"))
            out_paths.append(save_output(grn_m, "task1", "green_mask.png"))

        elif task_id == "task2":
            if f2 is None:
                st.warning("Please upload an image.")
                return
            gray = _decode_upload_gray(f2.getvalue())
            if gray is None:
                st.error("Could not decode image as grayscale.")
                return
            p1 = t2.pipline_global(gray)
            p2 = t2.pipline_adaptive(gray)
            p3 = t2.pipeline_frequency_natural(gray)
            p4 = t2.pipeline_enhance_xray(gray)

            def _to_u8(img: np.ndarray) -> np.ndarray:
                return np.clip(img, 0, 255).astype(np.uint8)

            st.subheader("Outputs")
            titles = ["Original", "HE+Gamma", "CLAHE+UM", "Frequency domain", "Frequency domain + CLAHE"]
            imgs = [gray, p1, p2, p3, p4]
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    disp = _to_u8(imgs[i])
                    st.image(disp, caption=titles[i], use_container_width=True)
            out_paths.append(save_output(gray, "task2", "00_original.png"))
            out_paths.append(save_output(p1, "task2", "01_he_gamma.png"))
            out_paths.append(save_output(p2, "task2", "02_clahe_um.png"))
            out_paths.append(save_output(_to_u8(p3), "task2", "03_frequency_natural.png"))
            out_paths.append(save_output(p4, "task2", "04_frequency_clahe.png"))

        elif task_id == "task3":
            if f3 is None:
                st.warning("Please upload an image.")
                return
            bgr = _decode_upload_bgr(f3.getvalue())
            if bgr is None:
                st.error("Could not decode image.")
                return
            enhancer = t3.IntelligentEnhancer()
            enhanced, category, stats, strategy = enhancer.enhance(bgr)
            annotated = t3.draw_info(enhanced, category, stats, strategy)

            st.subheader("Image analysis")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Detected problem (type)", category)
            with m2:
                st.metric("Brightness", f"{stats['brightness']:.1f}")
            with m3:
                st.metric("Contrast", f"{stats['contrast']:.1f}")
            st.write("**Strategy applied:**")
            for s in strategy:
                st.markdown(f"- {s}")

            st.subheader("Output")
            c0, c1 = st.columns(2)
            with c0:
                st.image(_rgb_for_display(bgr), caption="Original", use_container_width=True)
            with c1:
                st.image(_rgb_for_display(annotated), caption="Enhanced (with info overlay)", use_container_width=True)
            out_paths.append(save_output(annotated, "task3", "enhanced_annotated.png"))

        elif task_id == "task4":
            if f4 is None:
                st.warning("Please upload an image.")
                return
            image = Image.open(f4).convert("RGB")
            image_array = np.array(image)
            result = t4.process_document(image_array)
            st.subheader("Output")
            c0, c1 = st.columns(2)
            with c0:
                st.image(image_array, caption="Original", use_container_width=True)
            with c1:
                st.image(result, caption="Cleaned (binary)", use_container_width=True)
            out_paths.append(save_output(result, "task4", "cleaned.png"))

        elif task_id == "task5":
            if f5a is None or f5b is None:
                st.warning("Please upload both images.")
                return
            image1 = _decode_upload_bgr(f5a.getvalue())
            image2 = _decode_upload_bgr(f5b.getvalue())
            if image1 is None or image2 is None:
                st.error("Could not decode one or both images.")
                return

            matched_image, panorama, final_panorama, matches_count = t5.run_panorama_pipeline(image1, image2)

            st.subheader("Output")
            st.caption(f"Good Matches: {matches_count}")
            c0, c1, c2 = st.columns(3)
            with c0:
                st.image(_rgb_for_display(matched_image), caption="Feature Matching", use_container_width=True)
            with c1:
                st.image(_rgb_for_display(panorama), caption="Panorama Before Cropping", use_container_width=True)
            with c2:
                st.image(_rgb_for_display(final_panorama), caption="Final Panorama", use_container_width=True)

            out_paths.append(save_output(matched_image, "task5", "feature_matching.png"))
            out_paths.append(save_output(panorama, "task5", "panorama_before_cropping.png"))
            out_paths.append(save_output(final_panorama, "task5", "panorama_result.png"))

        elif task_id == "task6":
            if f6a is None or f6b is None:
                st.warning("Please upload both images.")
                return
            a = _decode_upload_bgr(f6a.getvalue())
            b = _decode_upload_bgr(f6b.getvalue())
            if a is None or b is None:
                st.error("Could not decode one or both images.")
                return
            flag, x, y = t6.detect_and_match_features(a, b)
            st.subheader("Output")
            if flag == 1 and x is not None and y is not None:
                st.success("there is an object")
                c0, c1 = st.columns(2)
                with c0:
                    st.image(_rgb_for_display(x), caption="Image A (matches)", use_container_width=True)
                with c1:
                    st.image(_rgb_for_display(y), caption="Image B (matches)", use_container_width=True)
                out_paths.append(save_output(x, "task6", "image_a_matches.png"))
                out_paths.append(save_output(y, "task6", "image_b_matches.png"))
            else:
                st.info("there is no common object")
                c0, c1 = st.columns(2)
                with c0:
                    st.image(_rgb_for_display(a), caption="Image A", use_container_width=True)
                with c1:
                    st.image(_rgb_for_display(b), caption="Image B", use_container_width=True)
                out_paths.append(save_output(a, "task6", "image_a.png"))
                out_paths.append(save_output(b, "task6", "image_b.png"))

        elif task_id == "task7":
            if f7l is None or f7r is None:
                st.warning("Please upload left and right images.")
                return
            out_dir = PROJECT_ROOT / "output-images" / "task7"
            out_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=Path(f7l.name).suffix, delete=False) as tl:
                tl.write(f7l.getvalue())
                left_path = tl.name
            with tempfile.NamedTemporaryFile(suffix=Path(f7r.name).suffix, delete=False) as tr:
                tr.write(f7r.getvalue())
                right_path = tr.name
            try:
                depth_map, colored_depth = t7.run(left_path, right_path, str(out_dir))
            finally:
                Path(left_path).unlink(missing_ok=True)
                Path(right_path).unlink(missing_ok=True)
            st.subheader("Output")
            c0, c1 = st.columns(2)
            with c0:
                st.image(depth_map, caption="Depth map (normalized)", use_container_width=True)
            with c1:
                st.image(_rgb_for_display(colored_depth), caption="Depth map (colored)", use_container_width=True)
            out_paths.append(str(out_dir / "depth_map_normalized.png"))
            out_paths.append(str(out_dir / "depth_map_colored.png"))
            out_paths.append(str(out_dir / "original_left.png"))

        elif task_id == "task8":
            if f8d is None or f8n is None or f8b is None:
                st.warning("Please upload all three exposure images.")
                return
            imgs = []
            for uf in (f8d, f8n, f8b):
                bgr = _decode_upload_bgr(uf.getvalue())
                if bgr is None:
                    st.error("Could not decode one of the images.")
                    return
                imgs.append(bgr)
            exposure_times = np.array([0.8, 2.5, 6.0], dtype=np.float32)
            hdr_tool = t8.VibrantHDR()
            final_image = hdr_tool.process(imgs, exposure_times)
            st.subheader("Output")
            st.image(_rgb_for_display(final_image), caption="HDR result", use_container_width=True)
            out_paths.append(save_output(final_image, "task8", "vibrant_hdr_result.png"))

    except Exception as e:
        st.exception(e)
        return

    for p in out_paths:
        st.caption(p)
    st.success("Done. Outputs saved under `output-images/` for this task.")


main()
