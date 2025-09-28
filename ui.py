import streamlit as st
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from grader import grader, get_classes
from avneh import load_model, get_image_files, preprocess_image
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

# -----------------------------------------------------------------
# Config
DEFAULT_OUTPUT_DIR = "inference_results"

st.set_page_config(
    page_title="Model Inference Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# -----------------------------------------------------------------
# Dark-mode styling with Plus Jakarta Sans
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>

body { 
    background-color: #111827; 
    color: #f9fafb; 
}

.metric-card { 
    background: #1f2937; 
    padding: 1.5rem; 
    border-radius: 12px; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.5); 
    border-left: 4px solid #3b82f6; 
    margin-bottom: 1rem; 
}

.metric-value { 
    font-size: 2rem; 
    font-weight: 700; 
    color: #f9fafb; 
    margin: 0; 
}

.metric-label { 
    font-size: 0.9rem; 
    color: #9ca3af; 
    margin: 0; 
    text-transform: uppercase; 
    letter-spacing: 0.5px; 
}

.positive-metric { border-left-color: #10b981; }
.negative-metric { border-left-color: #ef4444; }

.stButton button { 
    width: 100%; 
    background: #3b82f6; 
    color: white; 
    border: none; 
    padding: 0.6rem 1rem; 
    border-radius: 8px; 
    font-weight: 600; 
    transition: background 0.2s ease; 
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stButton button:hover { 
    background: #2563eb; 
    color: white; 
}

.tab-container { 
    background: #1f2937; 
    border-radius: 12px; 
    padding: 1.5rem; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.5); 
}

.error-label { 
    font-weight: 600; 
    color: #f9fafb; 
    margin-bottom: 0.5rem; 
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# Header
st.markdown("## Model Inference Dashboard")
st.markdown("### Performance & Analytics\n---")

# -----------------------------------------------------------------
# Function to open folder selector
def select_folder():
    """Open a folder selection dialog and return the selected path"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    folder_path = filedialog.askdirectory(
        title="Select Test Images Folder",
        initialdir=os.getcwd()  # Start from current working directory
    )
    
    root.destroy()
    return folder_path

# -----------------------------------------------------------------
# Sidebar configuration
with st.sidebar:
    st.markdown("## Configuration")

    # Mode selection
    mode = st.radio(
        "Mode",
        ["Single Model", "Comparison Mode"],
        index=0,
        help="Choose to run inference on a single model or compare multiple models"
    )

    # Model upload
    if mode == "Single Model":
        model_file = st.file_uploader(
            "Upload YOLO Model",
            type=["pt"],
            help="Select a trained YOLOv8 model (.pt)"
        )
    else:
        model_files = st.file_uploader(
            "Upload YOLO Models",
            type=["pt"],
            accept_multiple_files=True,
            help="Select multiple trained YOLOv8 models for side-by-side comparison"
        )

    st.markdown("---")
    
    # Folder selection section
    st.markdown("### Images Folder")

    test_dir = st.text_input(
        "Type Folder Path",
        value="",
    )
    
    st.markdown("---")

    # Parameters
    st.markdown("### Parameters")
    conf_threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.25, 0.01,
        help="Filter out detections below this confidence level"
    )

    st.markdown("---")

    run_button = st.button(
        "Run Inference",
        type="primary",
        use_container_width=True
    )
# -----------------------------------------------------------------
# Inference generator
def run_with_progress(model, test_dir, output_dir, conf_threshold):
    import time
    image_files = get_image_files(test_dir)
    total = len(image_files)

    results_summary = {
        'total_images': total,
        'processed_images': 0,
        'total_detections': 0,
        'detections_per_image': [],
        'processing_time': 0,
        'average_confidence': 0,
        'class_counts': {}
    }

    start_time = time.time()
    results_dir = Path(output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_files, 1):
        try:
            preprocessed_img = preprocess_image(image_path)
            results = model(preprocessed_img, conf=conf_threshold, verbose=False)
            result = results[0]

            detections = len(result.boxes) if result.boxes else 0
            results_summary['total_detections'] += detections
            results_summary['detections_per_image'].append({
                'image': image_path.name,
                'detections': detections,
                'confidences': [],
                'classes': []
            })

            if result.boxes:
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for conf, cls in zip(confidences, classes):
                    results_summary['detections_per_image'][-1]['confidences'].append(float(conf))
                    results_summary['detections_per_image'][-1]['classes'].append(int(cls))
                    cls_name = f"class_{int(cls)}"
                    results_summary['class_counts'][cls_name] = results_summary['class_counts'].get(cls_name, 0) + 1

            results_summary['processed_images'] += 1

            if detections > 0:
                annotated_img = result.plot()
                cv2.imwrite(str(results_dir / f"result_{image_path.name}"), annotated_img)

        except Exception as e:
            st.error(f"Error on {image_path.name}: {e}")

        yield i, total, results_summary

    end_time = time.time()
    results_summary['processing_time'] = end_time - start_time
    if results_summary['total_detections'] > 0:
        all_conf = [conf for det in results_summary['detections_per_image'] for conf in det['confidences']]
        results_summary['average_confidence'] = float(np.mean(all_conf))

    yield total, total, results_summary

# -----------------------------------------------------------------
# Display functions
def display_results(results_summary, test_dir):
    if results_summary:
        st.success("Inference completed")
        st.markdown("---")

        # KPI metrics
        col1, col2, col3, col4 = st.columns(4, gap="large")
        col1.metric("Images Processed", results_summary['processed_images'])
        col2.metric("Total Detections", results_summary['total_detections'])
        col3.metric("Avg Confidence", f"{results_summary['average_confidence']:.3f}")
        img_per_sec = results_summary['processed_images'] / max(results_summary['processing_time'],0.001)
        col4.metric("Images/Sec", f"{img_per_sec:.1f}")

        # TP/FP/FN
        try:
            tp, fp, fn = grader.main(f"{DEFAULT_OUTPUT_DIR}\\results", test_dir)
            col_tp, col_fp, col_fn, col_precision = st.columns(4, gap="large")
            col_tp.metric("True Positives", tp)
            col_fp.metric("False Positives", fp)
            col_fn.metric("False Negatives", fn)
            precision = tp / max((tp+fp),1)
            recall = tp / max((tp+fn),1)
            f1 = 2*precision*recall / max((precision+recall),0.001)
            col_precision.metric("Precision", f"{precision:.1%}")
        except:
            st.warning("Detailed metrics unavailable")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Data Table","Analytics","Results Gallery","Error Analysis"])

        with tab1:
            df = pd.DataFrame(results_summary['detections_per_image'])
            if not df.empty:
                df['avg_confidence'] = df['confidences'].apply(lambda x: sum(x)/len(x) if x else 0)
                st.dataframe(df[['image','detections','avg_confidence']], width='stretch', height=400)
            else:
                st.info("No detection data")

        with tab2:
            col_chart1, col_chart2 = st.columns(2, gap="large")
            with col_chart1:
                if results_summary['class_counts']:
                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.bar(results_summary['class_counts'].keys(),
                           results_summary['class_counts'].values(), color='#3b82f6')
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
                else:
                    st.info("No class detections")
            with col_chart2:
                try:
                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.bar(['Precision','Recall','F1'], [precision, recall, f1], color=['#10b981','#3b82f6','#8b5cf6'])
                    ax.set_ylim(0,1)
                    st.pyplot(fig)
                except:
                    st.info("Performance metrics unavailable")

        with tab3:
            results_dir = Path(DEFAULT_OUTPUT_DIR)/"results"
            if results_dir.exists():
                imgs = list(results_dir.glob("*"))
                if imgs:
                    cols = st.columns(3)
                    for i, img in enumerate(imgs):
                        with cols[i%3]:
                            st.image(str(img), width='stretch')
                else:
                    st.info("No annotated images")
            else:
                st.warning("Results directory not found")

        with tab4:
            get_classes.main(test_dir)
            fp_dir = Path(DEFAULT_OUTPUT_DIR)/"fp"
            fn_dir = Path(DEFAULT_OUTPUT_DIR)/"fn"
            col_fp, col_fn = st.columns(2, gap="large")

            with col_fp:
                st.markdown('<div class="error-label">False Positives</div>', unsafe_allow_html=True)
                if fp_dir.exists():
                    for img in fp_dir.glob("*"):
                        st.image(str(img), width='stretch')
                else:
                    st.info("No false positives found")

            with col_fn:
                st.markdown('<div class="error-label">False Negatives</div>', unsafe_allow_html=True)
                if fn_dir.exists():
                    for img in fn_dir.glob("*"):
                        st.image(str(img), width='stretch')
                else:
                    st.info("No false negatives found")

# -----------------------------------------------------------------
# Comparison Mode Display
def display_comparison_results(comparison_results, test_dir):
    if not comparison_results:
        st.warning("No valid results to compare")
        return

    st.markdown("## Comparison Summary")
    summary_table = []
    for idx, res in enumerate(comparison_results):
        summary = res['summary']
        tp, fp, fn = grader.main(f"{DEFAULT_OUTPUT_DIR}\\model_{1+idx}\\results", test_dir)

        precision = tp / max((tp+fp),1)
        recall = tp / max((tp+fn),1)
        f1 = 2*precision*recall / max((precision+recall),0.001)
        avg_conf = summary['average_confidence'] if summary else 0
        summary_table.append({
            "Model": res['name'],
            "Images Processed": summary['processed_images'] if summary else 0,
            "Total Detections": summary['total_detections'] if summary else 0,
            "Avg Confidence": f"{avg_conf:.3f}",
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": f"{precision:.2%}",
            "Recall": f"{recall:.2%}",
            "F1": f"{f1:.2%}"
        })
    st.dataframe(pd.DataFrame(summary_table))

    # Annotated Image Galleries
    st.markdown("### Annotated Images per Model")
    for res_idx, res in enumerate(comparison_results, 1):
        st.markdown(f"**{res['name']}**")
        results_dir = Path(DEFAULT_OUTPUT_DIR)/f"model_{res_idx}"/"results"
        if results_dir.exists():
            imgs = list(results_dir.glob("*"))
            if imgs:
                cols = st.columns(3)
                for i, img in enumerate(imgs):
                    with cols[i%3]:
                        st.image(str(img), width='stretch')
            else:
                st.info("No annotated images")
        else:
            st.warning("Results directory not found")

# -----------------------------------------------------------------
# Main flow
if run_button:
    if os.path.exists(Path("temp")):
        shutil.rmtree(Path("temp"))
        
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
                
    if not test_dir or not Path(test_dir).exists():
        st.error("Enter a valid test images folder path")
    else:
        test_dir_path = Path(test_dir)

        if mode == "Single Model":
            if model_file is None:
                st.error("Select a YOLO model file")
            else:
                if os.path.exists(DEFAULT_OUTPUT_DIR):
                    shutil.rmtree(DEFAULT_OUTPUT_DIR)

                # Load model
                with st.spinner("Loading model..."):
                    temp_dir = Path("temp")
                    model_path = temp_dir / f"temp_model.pt"
                    with open(model_path, "wb") as f:
                        f.write(model_file.getbuffer())
                    model = load_model(model_path)

                if model is None:
                    st.error("Failed to load model")
                else:
                    results_summary = None
                    progress_col, status_col = st.columns([4,1])
                    progress_bar = progress_col.progress(0.0)
                    status_display = status_col.empty()
                    for i, total, results_summary in run_with_progress(model, test_dir_path, DEFAULT_OUTPUT_DIR, conf_threshold):
                        progress_bar.progress(i / total)
                        status_display.markdown(f"{i}/{total}")
                    display_results(results_summary, test_dir_path)

        else:  # Comparison Mode
            if not model_files:
                st.error("Upload at least one YOLO model for comparison")
            else:
                if os.path.exists(DEFAULT_OUTPUT_DIR):
                    shutil.rmtree(DEFAULT_OUTPUT_DIR)

                comparison_results = []
                for idx, model_file in enumerate(model_files, 1):
                    st.markdown(f"### Running Model {idx}/{len(model_files)}: {model_file.name}")
                    temp_dir = Path("temp")
                    model_path = temp_dir / f"temp_model_{idx}.pt"
                    with open(model_path, "wb") as f:
                        f.write(model_file.getbuffer())

                    model = load_model(model_path)
                    if model is None:
                        st.error(f"Failed to load {model_file.name}")
                        continue

                    model_output_dir = Path(DEFAULT_OUTPUT_DIR) / f"model_{idx}"
                    model_output_dir.mkdir(parents=True, exist_ok=True)

                    results_summary = None
                    progress_col, status_col = st.columns([4,1])
                    progress_bar = progress_col.progress(0.0)
                    status_display = status_col.empty()
                    for i, total, results_summary in run_with_progress(model, test_dir_path, model_output_dir, conf_threshold):
                        progress_bar.progress(i / total)
                        status_display.markdown(f"{i}/{total}")

                    comparison_results.append({
                        "name": model_file.name,
                        "summary": results_summary
                    })
                display_comparison_results(comparison_results, test_dir_path)