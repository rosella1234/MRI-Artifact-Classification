
# This is a sample Python script.
import os
import cv2
import numpy as np  # calcoli numerici
import pandas as pd  # gestione excel/dati
import pydicom  # leggere immagini MRI in formato DICOM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import shap

# CARICAMENTO DATASET
# carico il file excel
df = pd.read_excel("C:/Users/gibog/Desktop/tesi/dataset/Classification.xlsx")

# se voglio solo la 92 o la 98 o altro questo comando lo devo togliere perchè non le include
# se invece volessi tenerlo devo arrivare almeno a 98 e non  a 23

df = df.head(2306)


# CASO 2 :QUANTIZZAZIONE

# LOAD DICOM
# quantizzazione
# glcm lavora mglio con 256 livelli oppure 64
# 4096/16 =256

# leggo file dicom
def load_dicom(path): # in ingresso percorso file DICOM
    dicom = pydicom.dcmread(path)# leggo  formto .dcm
    img = dicom.pixel_array.astype(
        np.uint16)  # pixel_array estrae i pixel , float32 conversione per problemi calcoli e normalizzazione

    # uint16
    # quantizzazione
    img = (img / 16).astype(np.uint8)
    return img


# Le immagini DICOM sono state direttamente quantizzate in 256 livelli di grigio per garantire
# la stabilità della matrice di co-occorrenza (GLCM) e ridurre la complessità computazionale.


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# SEGMENTAZIONE
# consideriamo solo il fantoccio e trascuro lo sfondo


# non utilizzo piu Otsu perchè:
# Otsu interpreta parte del bordo come sfondo soprattutto dove c'è sfocatura (immagine 99-98)
# l'intensità non era uniforme e quindi alcuni pixel dell'immagine venivano confusi

# NON POSSIAMO SEGMENTARE PER INTENSITA (DI PIXEL)-->SEGMENTO PER STRUTTURA(FORMA)
# EDGE-BASED-SEGMENTATION
def get_object_mask(image):
    # Normalizzazione per stabilità di Canny
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Filtro Gaussiano per eliminare rumore ad alta frequenza
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blur, 30, 100)

    # Chiusura morfologica per unire i contorni interrotti
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Trova i contorni geometrici
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    circularity = 0.0
    aspect_ratio = 1.0  # <-- AGGIUNTO di default

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

        # Calcolo della circolarità geometrica
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)

        # <-- AGGIUNTO: Calcolo dell'Aspect Ratio (Larghezza / Altezza)
        x, y, w, h = cv2.boundingRect(largest)
        if h > 0:
            aspect_ratio = float(w) / h

    return mask, circularity, aspect_ratio  # <-- Ora ritorna 3 valori!
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# EDGE WORK: I EXTRACT THE EDGE

def get_border_mask(mask, kernel_size=7, iterations=9):
    # geometric kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # erode the mask towards the inside
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    # expand the mask outwards by the same thickness
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    # create the new crown/border (outer border + inner border)
    border = cv2.subtract(dilated, eroded)

    return border


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FEATURE GLCM
def extract_glcm_features(img, mask):
    # Trova i contorni nella maschera per fare il ritaglio (crop) ed eliminare il nero superfluo
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        # Ritagliamo l'immagine sulla zona stretta del fantoccio
        roi_crop = img[y:y + h, x:x + w].astype(np.uint8)
    else:
        roi_crop = img.astype(np.uint8)

    # Parametri GLCM
    distances = [1, 2, 4,]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Calcoliamo la GLCM standard a 256 livelli
    glcm = graycomatrix(
        roi_crop,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=False  # Calcoliamo i conteggi assoluti prima di pulire
    )

    # TRUCCO FONDAMENTALE: Azzeriamo l'indice 0 (tutti i pixel di sfondo neri)
    # in modo che non influenzino il calcolo delle proprietà delle texture
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0

    # Rinormalizziamo la matrice in modo che la somma di tutte le probabilità sia 1
    glcm_sum = np.sum(glcm, axis=(0, 1), keepdims=True)
    glcm_norm = np.where(glcm_sum > 0, glcm / glcm_sum, 0.0)

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = {}

    # Usiamo le funzioni native di scikit-image passandogli la matrice pulita e normalizzata
    for prop in properties:
        values = graycoprops(glcm_norm, prop)
        features[prop + "_mean"] = values.mean()
        features[prop + "_std"] = values.std()

    return features

# servono altre features?

def extract_advanced_artifact_features(roi, is_border=False):
    """
    Estrae i profili direzionali completi (Orizzontali + Verticali) per catturare
    bande, doppi bordi e alterazioni strutturali sia globali che sul bordo.
    """
    roi = roi.astype(np.float32)
    features = {}

    # 1. Gradienti di Sobel (Direzionalità dei bordi artefatti)
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

    # 2. ANALISI DEI PROFILI DIREZIONALI (Bande Orizzontali)
    row_profile = np.mean(roi, axis=1)
    features[prefix + 'row_profile_std'] = np.std(row_profile)
    features[prefix + 'row_profile_range'] = np.max(row_profile) - np.min(row_profile)
    features[prefix + 'row_profile_cv'] = np.std(row_profile) / (np.mean(row_profile) + 1e-6)

    grad_row = np.diff(row_profile)
    features[prefix + 'row_gradient_max'] = np.max(np.abs(grad_row)) if len(grad_row) > 0 else 0
    features[prefix + 'row_gradient_std'] = np.std(grad_row) if len(grad_row) > 0 else 0

    # 3. ANALISI DEI PROFILI VERTICALI (Bande Verticali - AGGIUNTO!)
    col_profile = np.mean(roi, axis=0)
    features[prefix + 'col_profile_std'] = np.std(col_profile)
    features[prefix + 'col_profile_range'] = np.max(col_profile) - np.min(col_profile)
    features[prefix + 'col_profile_cv'] = np.std(col_profile) / (np.mean(col_profile) + 1e-6)

    grad_col = np.diff(col_profile)
    features[prefix + 'col_gradient_max'] = np.max(np.abs(grad_col)) if len(grad_col) > 0 else 0
    features[prefix + 'col_gradient_std'] = np.std(grad_col) if len(grad_col) > 0 else 0

    # 4. ALONI E DOPPI BORDI (Specifico se applicato sulla maschera del bordo)
    if is_border:
        # Se c'è un doppio bordo o alone, la variazione di intensità nella sola corona circolare è massima
        roi_pixels = roi[roi > 0]
        features['border_intensity_variance'] = np.std(roi_pixels) if len(roi_pixels) > 0 else 0
        features['border_max_to_mean'] = np.max(roi) / (np.mean(roi_pixels) + 1e-6) if len(roi_pixels) > 0 else 0

    return features

#feature FFT
def extract_fft_features(img, mask):
    # Ritagliamo anche qui sul fantoccio per evitare le frequenze spurie dei bordi esterni neri
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi = img[y:y + h, x:x + w].astype(np.float32)
    else:
        roi = img.astype(np.float32)

    # Applicazione della FFT 2D
    f_transform = np.fft.fft2(roi)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-6)

    features = {}
    # Feature globali dello spettro
    features['fft_mean'] = np.mean(magnitude_spectrum)
    features['fft_std'] = np.std(magnitude_spectrum)
    features['fft_max'] = np.max(magnitude_spectrum)

    # Profili monodimensionali per intercettare le righe (banding)
    row_profile = np.mean(magnitude_spectrum, axis=1)
    col_profile = np.mean(magnitude_spectrum, axis=0)

    # Se c'è banding, la deviazione standard del profilo dello spettro schizza verso l'alto
    features['fft_row_std'] = np.std(row_profile)
    features['fft_col_std'] = np.std(col_profile)
    features['fft_row_max_to_mean'] = np.max(row_profile) / (np.mean(row_profile) + 1e-6)
    features['fft_col_max_to_mean'] = np.max(col_profile) / (np.mean(col_profile) + 1e-6)  # <-- AGGIUNTO!

    return features





# ==========================================================
# VISUALIZZAZIONE DELLE FEATURE DI UNA SINGOLA IMMAGINE
# ==========================================================

def show_features(image_number):

    filename = f"{image_number}.dcm"

    dicom_folder = "C:/Users/gibog/Desktop/tesi/dataset/images"

    path = os.path.join(dicom_folder, filename)


    img = load_dicom(path)

    # miglioramento del contrasto SOLO per la visualizzazione
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_vis = clahe.apply(img)

    # Segmentazione
    mask, circularity, aspect_ratio = get_object_mask(img)

    border_mask = get_border_mask(mask)

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    border_img = cv2.bitwise_and(img, img, mask=border_mask)


    # SOBEL

    sobel_x = cv2.Sobel(img_vis.astype(np.float32),cv2.CV_64F,1,0,ksize=3)

    sobel_y = cv2.Sobel(img_vis.astype(np.float32),cv2.CV_64F,0,1,ksize=3)

    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    sobel_x_vis = cv2.normalize(np.abs(sobel_x),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    sobel_y_vis = cv2.normalize(np.abs(sobel_y),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    sobel_mag_vis = cv2.normalize(sobel_magnitude,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)


    # FFT 2D

    f = np.fft.fft2(masked_img)

    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)


    # FIGURA
    # Dizionario con tutte le immagini da plottare singolarmente
    images_dict = {
        f"Immagine {image_number} - Originale (Circ: {circularity:.3f}, AR: {aspect_ratio:.3f})": img_vis,
        f"Immagine {image_number} - Maschera": mask,
        f"Immagine {image_number} - Border Mask": border_mask,
        f"Immagine {image_number} - ROI Bordo": border_img,
        f"Immagine {image_number} - Sobel X": sobel_x_vis,
        f"Immagine {image_number} - Sobel Y": sobel_y_vis,
        f"Immagine {image_number} - Modulo Sobel": sobel_mag_vis,
        f"Immagine {image_number} - FFT": magnitude_spectrum  # da mostrare in bianco e nero
    }

    # Creazione di una finestra separata per ciascuna immagine
    for title, current_img in images_dict.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(current_img, cmap='gray')  # Forza la scala di grigi per tutte (inclusa la FFT)
        plt.title(title)
        plt.axis("off")

    plt.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# MAIN
# ESTRAZIONE FEATURE DATASET COMPLETO

all_features = []
all_labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    try:
        # nome file
        filename = str(int(row['filename'])) + ".dcm"
        dicom_folder = "C:/Users/gibog/Desktop/tesi/dataset/images"
        path = os.path.join(dicom_folder, filename)


        label = row['Label']

        img = load_dicom(path)

        # Segmentazione (Ora estrae anche l'aspect ratio)
        mask, circularity, aspect_ratio = get_object_mask(img)  # <-- MODIFICATO

        # Bordo
        border_mask = get_border_mask(mask)

        # Applico maschera all'immagine originale
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        border_img = cv2.bitwise_and(img, img, mask=border_mask)

        # 1. Feature GLCM
        glcm_features = extract_glcm_features(border_img, border_mask)

        # 2. Feature FFT (aggiornate con colonne)
        fft_features = extract_fft_features(masked_img, mask)

        # 3. Estrazione nuove feature avanzate per gli artefatti (Sobel + Profili H/V + Varianza Bordo)
        sobel_global = extract_advanced_artifact_features(masked_img, is_border=False)  # <-- MODIFICATO
        sobel_border = extract_advanced_artifact_features(border_img, is_border=True)  # <-- MODIFICATO

        # UNIONE DI TUTTE LE FEATURE
        combined_features = {
            **glcm_features,
            **fft_features,
            **sobel_global,
            **sobel_border
        }

        # Inseriamo la geometria completa come feature condizionante per il Random Forest
        combined_features['geom_circularity'] = circularity
        combined_features['geom_aspect_ratio'] = aspect_ratio  # <-- AGGIUNTO! (Rettangolare vs Circolare)


        # Aggiungiamo metadati e label
        combined_features['filename'] = filename
        combined_features['Label'] = label

        # Salvo nel dataset complessivo
        all_features.append(combined_features)
        all_labels.append(label)

    except Exception as e:

        print(f"Errore con file {filename}")

        print(e)



# DATAFRAME FEATURE

features_df = pd.DataFrame(all_features)

print("\nFEATURE ESTRATTE:")
print(features_df.head())


# SALVATAGGIO CSV DELLE FEATURE ESTRATTE DA GLCM

features_df.to_csv(
    "glcm_features.csv",
    index=False
)

print("\nCSV salvato correttamente")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PREPARAZIONE MACHINE LEARNING

# X = feature : input dell'algoritmo
X = features_df.drop(
    columns=['filename', 'Label']
)

# y = le etichette ,i label , gli scores
y = features_df['Label']

# TRAIN/TEST SPLIT
# divido ill test in due parti: addestramento e test
# 20% dei dati totali verrà destinato al test set, mentre il restante 80% andrà al training set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# RANDOM FOREST + CROSS VALIDATION


param_grid = {  # griglia iperparamtri del random forest da testare

    'n_estimators': [50, 100, 150, 200, 300, 400],  # numero alberi : più alberi --> maggior robustezza

    'max_depth': [None, 5, 10, 15, 20, 25],  # profondita degli alberi : quanto cresce
    # CONTROLLA OVERFITTING E PROFONDITA DI TEXTURE

    'min_samples_split': [2, 3, 4, 5],  # numero minimo di campioni necessari per dividere nodo
    # evita alberi troppo complessi (10)

    'min_samples_leaf': [1, 2, 3, 4]  # serve ad evitare foglie con pochi esempi (campioni)
}

# cross validatin da libreria stratifiedkfold...
cv = StratifiedKFold(  # utilizziamo StratifiedKFold perchè si mantiene la stessa proporzione delle classi (sbilanciate)
    # SBILANCIAMENTO: si hanno piu immagini appartenenti a 0 e 1
    n_splits=5,  # CV a 5 fold
    # il training set è diviso in 5 : CV lavora solo su TRAIN e 1 volta sola applica il TEST
    # dataset 2306 immagini , TRAIN (80%)= 1844 immagini , TEST (20%)=462 immagini
    # quindi 1844/5 = 369
    shuffle=True,
    random_state=42  # riproducibilità
)

grid = GridSearchCV(  # classe di python che esegue tutte le combinazioni e tutta CV senza cicli manuali

    estimator=RandomForestClassifier(  # algoritmo utilizzato RF
        class_weight='balanced',  # dato che abbiamo classi sbilanciate
        # Scikit-learn assegna automaticamente più peso alle classi meno rappresentate.
        # penalizza maggiormente gli errori di classificazione commessi sulle classi minoritarie
        random_state=42,
        n_jobs=-1  # usa tutti i core della CPU
    ),

    param_grid=param_grid,  # provo tutte le combinazioni di sopra (griglia)
    # 3 valori n_estimators X 5 valori max dpth X 3 split X 3 leaf
    # 135 combinazioni
    cv=cv,  # per ciascuna delle 81 combinazioni si esegue 5 fold e si calcola ACCURACY MEDIA
    # nel primo fold della prima comb si hanno 4 TRAIN e ultimo fold usato come VALID
    # nel secondo fold della prima comb si hanno 4 TRAIN e il penultimo fold usato come VALID

    scoring='accuracy',
    n_jobs=-1,

    verbose=2  # Scikit-Learn stampa soltanto messaggi tipo:
    # [CV] END max_depth=20, min_samples_leaf=1, ...
)
# in totale ho 135 combinazioni X 5 = 675 modelli allenati da RANDOM FOREST
# la Grid Search non cerca il modello che massimizza l'accuracy sul tuo test set
# , ma il modello che massimizza l'accuracy media in cross validation sul training set.

grid.fit(X_train, y_train)  # dove applico tutto il procedimento

# stampo i risultati completi (tutte le combianzioni) della cross validation su file di testo
results = pd.DataFrame(grid.cv_results_)  # grid.cv_results contiene tutte le informazioni della CV

with open(
        "C:/Users/gibog/Desktop/tesi/output_codici/results_personale1.4/risultati_personale1.4.txt",
        "w",
        encoding="utf-8"
) as f:
    for i in range(len(results)):
        f.write("=" * 80 + "\n")

        f.write(f"COMBINAZIONE {i + 1}\n")

        f.write(
            f"n_estimators={results.loc[i, 'param_n_estimators']} "
            f"max_depth={results.loc[i, 'param_max_depth']} "
            f"min_samples_split={results.loc[i, 'param_min_samples_split']} "
            f"min_samples_leaf={results.loc[i, 'param_min_samples_leaf']}\n\n"
        )

        f.write(
            f"Fold1 = {results.loc[i, 'split0_test_score']:.4f}\n"
        )

        f.write(
            f"Fold2 = {results.loc[i, 'split1_test_score']:.4f}\n"
        )

        f.write(
            f"Fold3 = {results.loc[i, 'split2_test_score']:.4f}\n"
        )

        f.write(
            f"Fold4 = {results.loc[i, 'split3_test_score']:.4f}\n"
        )

        f.write(
            f"Fold5 = {results.loc[i, 'split4_test_score']:.4f}\n"
        )

        f.write(
            f"\nMEDIA = {results.loc[i, 'mean_test_score']:.4f}\n"
        )

        f.write(
            f"DEVIAZIONE STANDARD = {results.loc[i, 'std_test_score']:.4f}\n\n"
        )

best_model = grid.best_estimator_  # restituisco il modello che ha la miglior accuracy delle combinazioni
# in best_model ho i miglior valori dei parametri che forniscono il modello con accuracy migliore


#SALVATAGGIO MODELLO E NOMI DELLE FEATURE
# 1. Salviamo il modello addestrato (.pkl)
model_save_path = "C:/Users/gibog/Desktop/tesi/output_codici/results_personale1.4/random_forest_artifact_model.pkl"
joblib.dump(best_model, model_save_path)

# 2. Salviamo l'elenco esatto dei nomi delle feature per essere certi dell'ordine
feature_columns_path = "C:/Users/gibog/Desktop/tesi/output_codici/results_personale1.4/feature_columns.pkl"
joblib.dump(list(X.columns), feature_columns_path)

print("\n===================================")
print(f"MODELLO SALVATO CON SUCCESSO IN: {model_save_path}")
print(f"COLONNE SALVATE IN: {feature_columns_path}")
print("===================================\n")







print("\n===================================")
print("CROSS VALIDATION COMPLETATA")
print("===================================")

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)  # quale è l'accuretezza media tra i fold di validazione

# TEST FINALE utilizzando le 462 immagini non considerate prima
# #usate per la prima volta

# uso il modello migliore sul test set per la prima volta dati in pasto al modello (evito OVERIFITTING)
y_pred = best_model.predict(X_test)

probs = best_model.predict_proba(X_test)  # con quale prob , immagine appartien a classe 0,1,2,3

acc = accuracy_score(y_test, y_pred)  # accuratezza del modello in base agli esiti del TEST
# ovvero: QUANTE NE INDOVINA GIUSTE!!!!
print("Test accuracy:", acc)

#Calcolo del Quadratic Weighted Kappa
qwk= cohen_kappa_score(y_test, y_pred, weights='quadratic')
print("Test QWK:",qwk)


best_model_info = {

    'n_estimators': grid.best_params_['n_estimators'],
    'max_depth': grid.best_params_['max_depth'],
    'min_samples_split': grid.best_params_['min_samples_split'],
    'min_samples_leaf': grid.best_params_['min_samples_leaf'],
    'cv_accuracy': grid.best_score_,
    'test_accuracy': acc,
    'test_qwk': qwk
}

# ORA MI OCCUPO DI STAMPRE I RISULTATI SU FILE EXCEL
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# RISULTATI DEI PARAMERI DI VALUTAZIONE DELL'ALGORITMO DI MACHINE LEARNING SALVATI SU UN UNICO FOGLIO EXCEL


# il problema è che il nostro dataset è sbilanciato : alla classe 0 appartengono piu immagini rispetto a classe 2 o 3
# per questo non ci basiamo solo su questa percentuale per accertare la validità dell'algortimo

# -->tabella CLASSIFICATION REPORT:

# Precision (Precisione): Di tutti i casi che il modello ha classificato come "Classe X"
# , quanti lo erano veramente? (Evita i falsi positivi).

# Recall (Richiamo/Sensibilità): Di tutti i casi reali di "Classe X" presenti nel dataset
# , quanti ne ha scovati il modello? (Evita i falsi negativi).

# F1-Score: Una media armonica tra Precision e Recall. È l'indicatore perfetto per capire se
# il modello è davvero bilanciato ed efficace.

# 2. CLASSIFICATION REPORT (convertito in DataFrame)
report_dict = classification_report(y_test, y_pred, output_dict=True)
# con output_dict=True in uscita non abbiamo un testo semplice ma messo in una tabella : ci pensa classification_report a farla
# estrae lei recall, f1-score
df_report = pd.DataFrame(report_dict).transpose().reset_index()
df_report.rename(columns={'index': 'Classe/Media'}, inplace=True)

# 3. CONFUSION MATRIX : È una tabella quadrata che mostra visivamente dove il modello si è confuso
cm = confusion_matrix(y_test, y_pred)  # creazione tabella
# confusion matrix visualizzata su excel su excel
class_labels = [f"Reale_{c}" for c in best_model.classes_]
col_labels = [f"Predetto_{c}" for c in best_model.classes_]
df_cm = pd.DataFrame(cm, index=class_labels, columns=col_labels).reset_index()
df_cm.rename(columns={'index': 'Classe Reale'}, inplace=True)

# 4. FEATURE IMPORTANCE: si puo calcolare attraverso il comando model.feature_importances_ ma non si puo calcolare per
# tutti gli algoritmi . solo Random Forest (algoritmi come alberi decisionali) ce l'hanno nativa/ integrata questa funzionalità
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
importance = importance.sort_values(by='importance', ascending=False)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PERCORSO DI SALVATAGGIO
percorso_excel_unico = "C:/Users/gibog/Desktop/tesi/output_codici/results_personale1.4/metriche_modello_personale1.4.xlsx"

df_best_model = pd.DataFrame([{

    'n_estimators':
        grid.best_params_['n_estimators'],

    'max_depth':
        grid.best_params_['max_depth'],

    'min_samples_split':
        grid.best_params_['min_samples_split'],

    'min_samples_leaf':
        grid.best_params_['min_samples_leaf'],

    'cv_accuracy':
        grid.best_score_,

    'test_accuracy':
        acc,

    'test_qwk':
        qwk
}])
# Scrittura sequenziale nello stesso foglio ('sheet:Riepilogo_Performance') : impostazioni su come si vuole ottenere il file excel:
with pd.ExcelWriter(percorso_excel_unico, engine='openpyxl') as writer:
    df_best_model.to_excel(
        writer,
        sheet_name='Riepilogo_Performance',
        index=False,
        startrow=0
    )

    # Tabella 2: Classification Report (Inizia alla riga 4, lasciando spazio)
    # Creiamo una riga di testo come titolo del blocco nel file Excel
    pd.DataFrame(["CLASSIFICATION REPORT:"]).to_excel(writer, sheet_name='Riepilogo_Performance', index=False,
                                                      header=False, startrow=3)
    df_report.to_excel(writer, sheet_name='Riepilogo_Performance', index=False, startrow=4)

    # Tabella 3: Confusion Matrix (Inizia alla riga 13, calcolando l'ingombro del report)
    pd.DataFrame(["MATRICE DI CONFUSIONE:"]).to_excel(writer, sheet_name='Riepilogo_Performance', index=False,
                                                      header=False, startrow=13)
    df_cm.to_excel(writer, sheet_name='Riepilogo_Performance', index=False, startrow=14)

    # Tabella 4: Feature Importance (Inizia alla riga 22)
    pd.DataFrame(["FEATURE IMPORTANCE:"]).to_excel(writer, sheet_name='Riepilogo_Performance', index=False,
                                                   header=False, startrow=21)
    importance.to_excel(writer, sheet_name='Riepilogo_Performance', index=False, startrow=22)

# stampami sul terminale la conferma della creazione del foglio excel
print("\n==================================================")
print(f"TUTTI I RISULTATI SALVATI NELLO STESSO FOGLIO IN: {percorso_excel_unico}")
print("==================================================")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# VALIDATION / TEST RESULTS IN EXCEL
# prediction sul test set
y_test_pred = y_pred

# probabilità predette
probs = best_model.predict_proba(
    X_test)  # percentuale di sicurezza (probabilità) che il modello ha per ciascuna classe.
# Ad esempio, per una riga potrebbe essere sicuro al 70% che sia Classe 1, al 20% Classe 0, ecc.


# TABELLA SU EXCEL
# dataframe risultati
test_results = pd.DataFrame({

    'filename': features_df.loc[X_test.index, 'filename'].values,

    'true_label': y_test.values,  # valori di classificazione di riferimento reali

    'predicted_label': y_test_pred,

    'correct': (
            y_test.values == y_test_pred
    ),
    'absolute_error':
        abs(
            y_test.values - y_pred
        ),
    # probabilità classi
    'prob_0': probs[:, 0],
    'prob_1': probs[:, 1],
    'prob_2': probs[:, 2],
    'prob_3': probs[:, 3]

})

# creazione e salvataggio excel nella cartella
output_test = "C:/Users/gibog/Desktop/tesi/output_codici/results_personale1.4/test_predictions_personale1.4.xlsx"

with pd.ExcelWriter(output_test, engine='openpyxl') as writer:
    # tabella immagini
    test_results.to_excel(
        writer,
        sheet_name='Test_Predictions',
        index=False,
        startrow=0
    )

    # riga vuota
    start_param = len(test_results) + 3

    # intestazione
    pd.DataFrame([[
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'cv_accuracy',
        'test_accuracy'
        'test_qwk'
    ]]).to_excel(
        writer,
        sheet_name='Test_Predictions',
        index=False,
        header=False,
        startrow=start_param
    )

    # valori
    pd.DataFrame([[
        grid.best_params_['n_estimators'],
        grid.best_params_['max_depth'],
        grid.best_params_['min_samples_split'],
        grid.best_params_['min_samples_leaf'],
        grid.best_score_,
        acc,
        qwk
    ]]).to_excel(
        writer,
        sheet_name='Test_Predictions',
        index=False,
        header=False,
        startrow=start_param + 1
    )

# stampami sul terminale la conferma della creazione del foglio excel
print("\n==============================")
print("TEST PREDICTIONS SALVATE")
print("==============================")

print(test_results.head())



print("\n=========================================")
print("VISUALIZZAZIONE FEATURE")
print("=========================================")

numero = int(input("Inserisci il numero dell'immagine (es. 92): "))

show_features(numero)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
