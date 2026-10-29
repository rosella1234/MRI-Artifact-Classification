#
import os
import cv2
import joblib
import numpy as np
import pandas as pd
import pydicom
from skimage.feature import graycomatrix, graycoprops

# ==============================================================================
# CONFIGURAZIONE DEI PERCORSI
# ==============================================================================
# Cartella contenente le ~300 sottocartelle di sequenze FSE
#BASE_DIR = r"C:/Users/gibog/Desktop/tesi/PostTaratura/PT_20260911_0567_TEST FSE^P/19_"
BASE_DIR = r"C:/Users/gibog/Desktop/tesi/PreTaratura/FT_20260908_0551_test^fse/2_"

# Modello Random Forest addestrato e lista ordinata delle feature
MODEL_PKL_PATH = r"C:/Users/gibog/Desktop/random_forest_artifact_model.pkl"
FEATURE_COLS_PATH = r"C:/Users/gibog/Desktop/feature_columns.pkl"


# File Excel di output
#OUTPUT_EXCEL = r"C:/Users/gibog/Desktop/tesi/output_codici/Results_Post_Taratura/Sequences_Validation_ML_Results_Post.xlsx"
OUTPUT_EXCEL = r"C:/Users/gibog/Desktop/tesi/output_codici/Results_Pre_Taratura/Sequences_Validation_ML_Results_Pre.xlsx"

# ==============================================================================
# FUNZIONI DI PREPROCESSING ED ESTRAZIONE FEATURE (IDENTICHE AL TRAINING)
# ==============================================================================
def load_dicom(path):
    dicom = pydicom.dcmread(path, force=True)
    if not hasattr(dicom, "pixel_array"):
        raise ValueError("File privo di pixel_array")
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

    glcm = graycomatrix(
        roi_crop,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=False
    )

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

    features[prefix + 'sobel_x_mean'] = float(np.mean(gx))
    features[prefix + 'sobel_x_std'] = float(np.std(gx))
    features[prefix + 'sobel_y_mean'] = float(np.mean(gy))
    features[prefix + 'sobel_y_std'] = float(np.std(gy))
    features[prefix + 'sobel_yx_ratio'] = float(np.mean(gy) / (np.mean(gx) + 1e-6))

    # Profili Orizzontali
    row_profile = np.mean(roi, axis=1)
    features[prefix + 'row_profile_std'] = float(np.std(row_profile))
    features[prefix + 'row_profile_range'] = float(np.ptp(row_profile))  # ptp calcola max - min senza conflitti di tipo
    features[prefix + 'row_profile_cv'] = float(np.std(row_profile) / (np.mean(row_profile) + 1e-6))

    grad_row = np.diff(row_profile)
    features[prefix + 'row_gradient_max'] = float(np.max(np.abs(grad_row))) if len(grad_row) > 0 else 0.0
    features[prefix + 'row_gradient_std'] = float(np.std(grad_row)) if len(grad_row) > 0 else 0.0

    # Profili Verticali
    col_profile = np.mean(roi, axis=0)
    features[prefix + 'col_profile_std'] = float(np.std(col_profile))
    features[prefix + 'col_profile_range'] = float(np.ptp(col_profile))  # ptp calcola max - min
    features[prefix + 'col_profile_cv'] = float(np.std(col_profile) / (np.mean(col_profile) + 1e-6))

    grad_col = np.diff(col_profile)
    features[prefix + 'col_gradient_max'] = float(np.max(np.abs(grad_col))) if len(grad_col) > 0 else 0.0
    features[prefix + 'col_gradient_std'] = float(np.std(grad_col)) if len(grad_col) > 0 else 0.0

    if is_border:
        roi_pixels = roi[roi > 0]
        features['border_intensity_variance'] = float(np.std(roi_pixels)) if len(roi_pixels) > 0 else 0.0
        features['border_max_to_mean'] = float(np.max(roi) / (np.mean(roi_pixels) + 1e-6)) if len(roi_pixels) > 0 else 0.0

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

    features = {
        'fft_mean': np.mean(magnitude_spectrum),
        'fft_std': np.std(magnitude_spectrum),
        'fft_max': np.max(magnitude_spectrum)
    }

    row_profile = np.mean(magnitude_spectrum, axis=1)
    col_profile = np.mean(magnitude_spectrum, axis=0)

    features['fft_row_std'] = np.std(row_profile)
    features['fft_col_std'] = np.std(col_profile)
    features['fft_row_max_to_mean'] = np.max(row_profile) / (np.mean(row_profile) + 1e-6)
    features['fft_col_max_to_mean'] = np.max(col_profile) / (np.mean(col_profile) + 1e-6)

    return features


def extract_all_dicom_features(dicom_path):
    """Estrae l'intero dizionario di feature per la singola immagine."""
    img = load_dicom(dicom_path)
    mask, circularity, aspect_ratio = get_object_mask(img)
    border_mask = get_border_mask(mask)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    border_img = cv2.bitwise_and(img, img, mask=border_mask)

    glcm_features = extract_glcm_features(border_img, border_mask)
    fft_features = extract_fft_features(masked_img, mask)
    sobel_global = extract_advanced_artifact_features(masked_img, is_border=False)
    sobel_border = extract_advanced_artifact_features(border_img, is_border=True)

    combined = {
        **glcm_features,
        **fft_features,
        **sobel_global,
        **sobel_border,
        'geom_circularity': circularity,
        'geom_aspect_ratio': aspect_ratio
    }
    return combined


# ==============================================================================
# PIPELINE DI INFERENZA SULLE SEQUENZE
# ==============================================================================
def run_ml_sequence_validation():
    print(f"Caricamento modello Random Forest da: {MODEL_PKL_PATH}")
    model = joblib.load(MODEL_PKL_PATH)

    # Carica la sequenza esatta delle colonne attesa dal modello (52 feature)
    feature_columns = joblib.load(FEATURE_COLS_PATH)
    print(f"Caricate {len(feature_columns)} colonne di feature di riferimento.")

    sequence_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    print(f"Trovate {len(sequence_dirs)} sequenze in: {BASE_DIR}\n")

    summary_records = []

    for seq_idx, seq_name in enumerate(sequence_dirs, 1):
        seq_path = os.path.join(BASE_DIR, seq_name)
        raw_files = sorted([
            f for f in os.listdir(seq_path)
            if os.path.isfile(os.path.join(seq_path, f))
        ])

        scores = []
        file_evaluations = {}

        for file_name in raw_files:
            file_path = os.path.join(seq_path, file_name)
            try:
                # Estrazione dizionario feature
                feat_dict = extract_all_dicom_features(file_path)

                # Allineamento rigoroso delle colonne con l'ordine di training
                x_input = pd.DataFrame([feat_dict])[feature_columns]

                # Predizione della classe (0, 1, 2, 3)
                pred_class = int(model.predict(x_input)[0])

                scores.append(pred_class)
                file_evaluations[file_name] = pred_class
            except (pydicom.errors.InvalidDicomError, ValueError, KeyError, cv2.error):
                continue

        if len(scores) == 0:
            print(f"[{seq_idx:>3}/{len(sequence_dirs)}] {seq_name}: Nessun file valido.")
            continue

        row = {
            "sequence_name": seq_name,
            "num_images": len(scores),
            "mean_score": round(float(np.mean(scores)), 3),
            "max_score": int(np.max(scores)),
            "min_score": int(np.min(scores))
        }
        row.update(file_evaluations)
        summary_records.append(row)

        print(
            f"[{seq_idx:>3}/{len(sequence_dirs)}] {seq_name:<25} | "
            f"Media: {row['mean_score']:.2f} | Max: {row['max_score']} | Min: {row['min_score']} | Immagini: {len(scores)}"
        )

    # Esportazione Excel
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    df_final = pd.DataFrame(summary_records)

    fixed_cols = ["sequence_name", "num_images", "mean_score", "max_score", "min_score"]
    other_cols = [c for c in df_final.columns if c not in fixed_cols]
    df_final = df_final[fixed_cols + other_cols]

    df_final.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nElaborazione completata con successo. Risultati esportati in: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    run_ml_sequence_validation()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
