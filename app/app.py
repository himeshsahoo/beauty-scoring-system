from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from analysis import (
    load_model_bundle,
    occlusion_sensitivity,
    predict_ablation_table,
    predict_background_table,
    predict_face,
    predict_quality_table,
)
from common import FaceDetector, device_info, load_config


def score_agreement_label(std: float | None) -> str:
    if std is None:
        return 'Unavailable'
    if std < 0.35:
        return 'High'
    if std < 0.75:
        return 'Moderate'
    return 'Low'


st.set_page_config(page_title='Human-rated facial aesthetics', page_icon='◌', layout='wide')
cfg = load_config(ROOT / 'config.yaml')
st.title('AI-Based Facial Aesthetic Scoring')
st.warning(
    'This predicts patterns in historical human aesthetic ratings, not anyone’s objective beauty or worth. '
    'Results may reflect demographic, cultural, and image-quality bias; do not use it for hiring, admissions, '
    'dating, medical, or other high-impact decisions.'
)

if 'analysis_bundle' not in st.session_state:
    st.session_state.analysis_bundle = None

upload = st.file_uploader('Upload a face image', type=['jpg', 'jpeg', 'png'])

if upload:
    data = upload.getvalue()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        st.error('The uploaded file is not a valid image.')
    else:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption='Original image', use_container_width=True)
        detector = FaceDetector(cfg)
        boxes = detector.detect(rgb)
        if not boxes:
            st.error('No face detected. Try a clear, frontal image.')
        else:
            st.info(f'{len(boxes)} face(s) detected. Select the face to score; no face is silently chosen.')
            face_index = st.selectbox(
                'Face',
                range(len(boxes)),
                format_func=lambda index: f'Face {index + 1} - box {boxes[index]}',
            )
            crop = detector.crop(rgb, boxes[face_index])
            st.image(crop, caption='Selected face crop', width=300)

            checkpoint_paths = sorted((ROOT / 'models').glob('*_best.pth'))
            if not checkpoint_paths:
                st.error('No trained model checkpoint found. Train a model first.')
            elif st.button('Predict human-rated aesthetic score', type='primary'):
                device = device_info()
                models, ensemble_enabled, loaded_paths = load_model_bundle(checkpoint_paths, device, minimum_ensemble=5)
                summary, gradcam_overlay, gradcam_heatmap = predict_face(crop, models, cfg, device)
                ablation_table = predict_ablation_table(models, detector, cfg, device, rgb, boxes[face_index])
                background_table = predict_background_table(models, detector, cfg, device, rgb, boxes[face_index])
                quality_table = predict_quality_table(models, cfg, device, crop)
                occlusion_grid, occlusion_heatmap, base_score = occlusion_sensitivity(models, cfg, device, crop)
                occlusion_overlay = cv2.applyColorMap(np.uint8(occlusion_heatmap * 255), cv2.COLORMAP_TURBO)
                occlusion_overlay = cv2.cvtColor(occlusion_overlay, cv2.COLOR_BGR2RGB)
                gradcam_resized = cv2.resize(gradcam_heatmap, (occlusion_heatmap.shape[1], occlusion_heatmap.shape[0]), interpolation=cv2.INTER_CUBIC)
                gradcam_flat = gradcam_resized.reshape(-1).astype(np.float32)
                occlusion_flat = occlusion_heatmap.reshape(-1).astype(np.float32)
                corr = float(np.corrcoef(gradcam_flat, occlusion_flat)[0, 1]) if gradcam_flat.size > 1 else float('nan')
                if np.isnan(corr):
                    agreement = 'Unavailable'
                elif corr > 0.6:
                    agreement = 'High'
                elif corr > 0.3:
                    agreement = 'Moderate'
                else:
                    agreement = 'Low'
                st.session_state.analysis_bundle = {
                    'ensemble_enabled': ensemble_enabled,
                    'checkpoint_count': len(loaded_paths),
                    'summary': summary,
                    'gradcam_overlay': gradcam_overlay,
                    'gradcam_heatmap': gradcam_heatmap,
                    'ablation_table': ablation_table,
                    'background_table': background_table,
                    'quality_table': quality_table,
                    'occlusion_overlay': occlusion_overlay,
                    'occlusion_grid': occlusion_grid,
                    'occlusion_base': base_score,
                    'gradcam_occlusion_corr': corr,
                    'gradcam_occlusion_agreement': agreement,
                }

analysis = st.session_state.analysis_bundle
if analysis:
    summary = analysis['summary']
    st.subheader('Prediction')
    left, right = st.columns(2)
    with left:
        metric_label = 'Ensemble prediction' if analysis['ensemble_enabled'] else 'Predicted aesthetic score'
        st.metric(metric_label, f'{summary.score:.2f} / 10')
        if analysis['ensemble_enabled']:
            st.write(f'Ensemble uncertainty: ±{summary.uncertainty:.2f}')
            st.write(f'Model agreement: {summary.agreement}')
            st.caption(f'Ensemble loaded from {analysis["checkpoint_count"]} checkpoint(s).')
        else:
            st.write('Ensemble not trained — uncertainty unavailable')
            st.caption('Train at least 5 independently trained checkpoints to enable ensemble uncertainty.')
        st.write('Face detected: ✓')
    with right:
        st.image(
            analysis['gradcam_overlay'],
            caption='Grad-CAM saliency for the selected face crop. Bright regions show where the model was most sensitive; this is not an objective beauty map.',
            use_container_width=True,
        )

    st.markdown(f'Grad-CAM and occlusion agreement: **{analysis["gradcam_occlusion_agreement"]}**  ')
    st.caption(f'Heatmap correlation: {analysis["gradcam_occlusion_corr"]:.2f}')

    tab_ablation, tab_background, tab_quality, tab_occlusion, tab_details = st.tabs(
        ['Ablation', 'Background', 'Image quality', 'Occlusion sensitivity', 'Details']
    )

    with tab_ablation:
        st.write('Face-part ablation and masking tests on the selected face crop.')
        st.dataframe(analysis['ablation_table'], use_container_width=True, hide_index=True)

    with tab_background:
        st.write('Background dependence test with the same face held constant.')
        st.dataframe(analysis['background_table'], use_container_width=True, hide_index=True)

    with tab_quality:
        st.write('Image-quality dependence test on the selected face crop.')
        st.dataframe(analysis['quality_table'], use_container_width=True, hide_index=True)

    with tab_occlusion:
        st.write('Occlusion sensitivity heatmap: absolute score change after masking each grid region.')
        st.image(analysis['occlusion_overlay'], caption='Occlusion sensitivity heatmap', use_container_width=True)
        st.write(f'Base ensemble score used for occlusion test: {analysis["occlusion_base"]:.2f} / 10')

    with tab_details:
        st.write('Interpretability details and model status.')
        st.write(f'Checkpoint count: {analysis["checkpoint_count"]}')
        st.write(f'Ensemble enabled: {analysis["ensemble_enabled"]}')
        st.write(f'Predictions were computed from the cropped face, not the full image.')
