import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import joblib
from skimage.feature import graycomatrix, graycoprops


# ==============================================================================
# 1. FUNZIONI DI ESTRAZIONE FEATURE (IDENTICHE AL TRAINING)
# ==============================================================================

def load_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.uint16)
    img = (img / 16).astype(np.uint8)
    return img


def get_object_mask(image):
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    circularity = 0.0
    aspect_ratio = 1.0

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(largest)
        if h > 0:
            aspect_ratio = float(w) / h

    return mask, circularity, aspect_ratio


def get_border_mask(mask, kernel_size=7, iterations=9):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    return cv2.subtract(dilated, eroded)


def extract_glcm_features(img, mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi_crop = img[y:y + h, x:x + w].astype(np.uint8)
    else:
        roi_crop = img.astype(np.uint8)

    distances = [1, 2, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(roi_crop, distances=distances, angles=angles, levels=256, symmetric=True, normed=False)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0

    glcm_sum = np.sum(glcm, axis=(0, 1), keepdims=True)
    glcm_norm = np.where(glcm_sum > 0, glcm / glcm_sum, 0.0)

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = {}
    for prop in properties:
        values = graycoprops(glcm_norm, prop)
        features[prop + "_mean"] = values.mean()
        features[prop + "_std"] = values.std()
    return features


def extract_advanced_artifact_features(roi, is_border=False):
    roi = roi.astype(np.float32)
    features = {}
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    gx = np.abs(sobel_x)
    gy = np.abs(sobel_y)

    prefix = "border_" if is_border else "global_"
    features[prefix + 'sobel_x_mean'] = np.mean(gx)
    features[prefix + 'sobel_x_std'] = np.std(gx)
    features[prefix + 'sobel_y_mean'] = np.mean(gy)
    features[prefix + 'sobel_y_std'] = np.std(gy)
    features[prefix + 'sobel_yx_ratio'] = np.mean(gy) / (np.mean(gx) + 1e-6)

    row_profile = np.mean(roi, axis=1)
    features[prefix + 'row_profile_std'] = np.std(row_profile)
    features[prefix + 'row_profile_range'] = np.max(row_profile) - np.min(row_profile)
    features[prefix + 'row_profile_cv'] = np.std(row_profile) / (np.mean(row_profile) + 1e-6)

    grad_row = np.diff(row_profile)
    features[prefix + 'row_gradient_max'] = np.max(np.abs(grad_row)) if len(grad_row) > 0 else 0
    features[prefix + 'row_gradient_std'] = np.std(grad_row) if len(grad_row) > 0 else 0

    col_profile = np.mean(roi, axis=0)
    features[prefix + 'col_profile_std'] = np.std(col_profile)
    features[prefix + 'col_profile_range'] = np.max(col_profile) - np.min(col_profile)
    features[prefix + 'col_profile_cv'] = np.std(col_profile) / (np.mean(col_profile) + 1e-6)

    grad_col = np.diff(col_profile)
    features[prefix + 'col_gradient_max'] = np.max(np.abs(grad_col)) if len(grad_col) > 0 else 0
    features[prefix + 'col_gradient_std'] = np.std(grad_col) if len(grad_col) > 0 else 0

    if is_border:
        roi_pixels = roi[roi > 0]
        features['border_intensity_variance'] = np.std(roi_pixels) if len(roi_pixels) > 0 else 0
        features['border_max_to_mean'] = np.max(roi) / (np.mean(roi_pixels) + 1e-6) if len(roi_pixels) > 0 else 0

    return features


def extract_fft_features(img, mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi = img[y:y + h, x:x + w].astype(np.float32)
    else:
        roi = img.astype(np.float32)

    f_transform = np.fft.fft2(roi)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-6)

    features = {}
    features['fft_mean'] = np.mean(magnitude_spectrum)
    features['fft_std'] = np.std(magnitude_spectrum)
    features['fft_max'] = np.max(magnitude_spectrum)

    row_profile = np.mean(magnitude_spectrum, axis=1)
    col_profile = np.mean(magnitude_spectrum, axis=0)

    features['fft_row_std'] = np.std(row_profile)
    features['fft_col_std'] = np.std(col_profile)
    features['fft_row_max_to_mean'] = np.max(row_profile) / (np.mean(row_profile) + 1e-6)
    features['fft_col_max_to_mean'] = np.max(col_profile) / (np.mean(col_profile) + 1e-6)

    return features


def extract_single_image_features(dicom_path):
    img = load_dicom(dicom_path)
    mask, circularity, aspect_ratio = get_object_mask(img)
    border_mask = get_border_mask(mask)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    border_img = cv2.bitwise_and(img, img, mask=border_mask)

    glcm_feats = extract_glcm_features(border_img, border_mask)
    fft_feats = extract_fft_features(masked_img, mask)
    sobel_global = extract_advanced_artifact_features(masked_img, is_border=False)
    sobel_border = extract_advanced_artifact_features(border_img, is_border=True)

    combined = {
        **glcm_feats,
        **fft_feats,
        **sobel_global,
        **sobel_border,
        'geom_circularity': circularity,
        'geom_aspect_ratio': aspect_ratio
    }
    return combined


# ==============================================================================
# 2. INFERENCE / VALIDAZIONE SULLE NUOVE IMMAGINI
# ==============================================================================

if __name__ == "__main__":
    # PERCORSI (modifica solo se sposti le cartelle)
    NEW_IMAGES_FOLDER = "C:/Users/gibog/Desktop/tesi/dataset/DicomForValidation"
    MODEL_PATH = "C:/Users/gibog/Desktop/tesi/output_codici/Results_ML_Firpo_+Val/random_forest_artifact_model.pkl"
    FEATURE_COLS_PATH = "C:/Users/gibog/Desktop/tesi/output_codici/Results_ML_Firpo_+Val/feature_columns.pkl"
    OUTPUT_EXCEL = "C:/Users/gibog/Desktop/tesi/output_codici/Results_ML_Firpo_+Val/risultati_validation.xlsx"

    # Carica modello e lista feature salvate
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)

    # Elenco di TUTTI i file presenti nella cartella (esclude solo eventuali cartelle interne)
    dicom_files = [
        f for f in os.listdir(NEW_IMAGES_FOLDER)
        if os.path.isfile(os.path.join(NEW_IMAGES_FOLDER, f))
    ]

    print(f"Trovate {len(dicom_files)} nuove immagini da classificare.\n")

    results = []

    for file_name in dicom_files:
        full_path = os.path.join(NEW_IMAGES_FOLDER, file_name)
        try:
            feats = extract_single_image_features(full_path)
            feats_df = pd.DataFrame([feats])[feature_columns]

            predicted_class = model.predict(feats_df)[0]
            probs = model.predict_proba(feats_df)[0]

            results.append({
                'filename': file_name,
                'predicted_class': predicted_class,
                'prob_class_0': probs[0],
                'prob_class_1': probs[1],
                'prob_class_2': probs[2],
                'prob_class_3': probs[3],
                'confidence': np.max(probs)
            })

            print(f"File: {file_name:<15} -> Classe: {predicted_class} (Confidenza: {np.max(probs) * 100:.1f}%)")

        except Exception as e:
            print(f"Errore con il file {file_name}: {e}")

    # Salvataggio su Excel
    res_df = pd.DataFrame(results)
    res_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nRisultati salvati con successo in: {OUTPUT_EXCEL}")