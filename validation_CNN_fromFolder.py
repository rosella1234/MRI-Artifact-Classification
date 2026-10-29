# This is a sample Python script.
import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==============================================================================
# CONFIGURAZIONE DEI PERCORSI E PARAMETRI
# ==============================================================================
# Percorso esatto alla cartella "19_" mostrato nello screenshot
#BASE_DIR = r"C:/Users/gibog/Desktop/tesi/PostTaratura/PT_20260911_0567_TEST FSE^P/19_"
BASE_DIR = r"C:/Users/gibog/Desktop/tesi/PreTaratura/FT_20260908_0551_test^fse/2_"

# Pesi del modello CNN salvati
WEIGHTS_PATH = r"C:/Users/gibog/Desktop/tesi/output_codici/results_forced_test_60epochs_cnn_cop1_gemOptim_docker/resnet_mri_artifacts.pth"

# File Excel di output per le sequenze aggregate
#OUTPUT_EXCEL = r"C:/Users/gibog/Desktop/tesi/output_codici/Results_Post_Taratura/Sequences_Validation_Results_Folder.xlsx"
OUTPUT_EXCEL = r"C:/Users/gibog/Desktop/tesi/output_codici/Results_Pre_Taratura/Sequences_Validation_Results_Folder_pre.xlsx"

IMG_SIZE = 256
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# ARCHITETTURA DELLA RETE
# ==============================================================================
def resnet18_mri():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


# ==============================================================================
# PREPROCESSING DICOM
# ==============================================================================
def preprocess_dicom(dicom_path):
    # stop_before_pixels=False assicura il caricamento completo
    ds = pydicom.dcmread(dicom_path, force=True)

    if not hasattr(ds, "pixel_array"):
        raise ValueError("File privo di pixel_array")

    img = ds.pixel_array.astype(np.float32)

    p_low, p_high = np.percentile(img, (1, 99))
    img = np.clip(img, p_low, p_high)

    if p_high > p_low:
        img = (img - p_low) / (p_high - p_low)
    else:
        img = np.zeros_like(img)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


# ==============================================================================
# INFERENCE A LIVELLO DI SEQUENZA
# ==============================================================================
def run_sequence_validation():
    print(f"Dispositivo in uso: {DEVICE}")
    print(f"Caricamento modello da: {WEIGHTS_PATH}")

    model = resnet18_mri()
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Elenco di tutte le sottocartelle (sequenze) all'interno di 19_
    sequence_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    print(f"Trovate {len(sequence_dirs)} sequenze in: {BASE_DIR}\n")

    summary_records = []

    with torch.no_grad():
        for seq_idx, seq_name in enumerate(sequence_dirs, 1):
            seq_path = os.path.join(BASE_DIR, seq_name)

            # Recupera tutti i file all'interno della singola sequenza
            raw_files = sorted([
                f for f in os.listdir(seq_path)
                if os.path.isfile(os.path.join(seq_path, f))
            ])

            scores = []
            file_evaluations = {}

            for file_name in raw_files:
                file_path = os.path.join(seq_path, file_name)
                try:
                    input_tensor = preprocess_dicom(file_path).to(DEVICE)
                    outputs = model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
                    predicted_class = int(np.argmax(probabilities))

                    scores.append(predicted_class)
                    file_evaluations[file_name] = predicted_class
                except Exception:
                    # Salta eventuali file non-DICOM presenti nella cartella
                    continue

            if len(scores) == 0:
                print(f"[{seq_idx}/{len(sequence_dirs)}] {seq_name}: Nessun file DICOM valido.")
                continue

            # Creazione del record riga per la sequenza
            row = {
                "sequence_name": seq_name,
                "num_images": len(scores),
                "mean_score": round(float(np.mean(scores)), 3),
                "max_score": int(np.max(scores)),
                "min_score": int(np.min(scores))
            }

            # Aggiunge una colonna per ciascun file della cartella col suo voto
            row.update(file_evaluations)
            summary_records.append(row)

            print(
                f"[{seq_idx:>3}/{len(sequence_dirs)}] {seq_name:<25} | "
                f"Media: {row['mean_score']:.2f} | Max: {row['max_score']} | Min: {row['min_score']} | Immagini: {len(scores)}"
            )

    # Salvataggio su Excel
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    df_final = pd.DataFrame(summary_records)

    # Ordina le prime 5 colonne metriche prima dei singoli file
    fixed_cols = ["sequence_name", "num_images", "mean_score", "max_score", "min_score"]
    other_cols = [c for c in df_final.columns if c not in fixed_cols]
    df_final = df_final[fixed_cols + other_cols]

    df_final.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nElaborazione terminata con successo. Risultati esportati in: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    run_sequence_validation()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
