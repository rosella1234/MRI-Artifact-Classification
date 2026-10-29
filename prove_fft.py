# This is a sample Python script.


import os
import cv2
import numpy as np  # calcoli numerici
import pandas as pd  # gestione excel/dati
import pydicom  # leggere immagini MRI in formato DICOM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
def load_dicom(path):  # in ingresso percorso file DICOM
    dicom = pydicom.dcmread(path)  # leggo  formto .dcm
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
    # normalizzazione: perche quantizzazione aggressiva puo distruggere il bordo e collassare nello sfondo
    # nomralizzo per segmentazione (canny è stabile)
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # applico un filtro gaussiano blur ->LPF :elimino rumore ad alta frequenza che produce bordi falsi:
    # ogni pixel viene sostituito con una media pesata dei pixel vicini
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Un filtro gaussiano è stato applicato prima della rilevazione dei bordi per ridurre il rumore
    # ad alta frequenza tipico delle immagini MRI e migliorare la continuità spaziale dei contorni
    # rilevati mediante algoritmo di Canny

    # canny: non guarda i pixel , ma cerca discontinuità (gradienti forti)
    # o meglio, vuole trovare bordi continui.
    # CANNY è UN ALGORITMO DI EDGE DETECTION
    # MATMATICAMENTE UN BORDO è UN GRADIENTE ELEVATO (VARIAZIONE IMPROVVISA INTENSITA)
    # se una zona cambia lentamente → non è un bordo
    # se una zona cambia rapidamente → probabilmente
    # è il contorno dell’oggetto

    # in questo modo il bordo del fantoccio diventa evidente , lo sfondo sparisce
    edges = cv2.Canny(blur, 30, 100)  # doppia soglia : grad>100 sicuro bordo
    # grad<30 rumore

    # l'output di canny è una mappa di bordi ma non sono continui , ma interrotti o con gap
    # il bordo esiste ma non è una figura geometrica continua

    # trasformiamo in una figura geometrica chiusa: bordi continui
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # abbiamo bordo continuo, forma chiusa ,struuta geometrica coerente
    # ma immagine ha solo pixel o bianchi o neri (da norm)
    # con findcontours() trovo regioni geometriche nell'immagine binaria
    # trasformo la mappa di bordi in una rappresentazione delle forme presenti
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # abbiamo una lista delle forme presenti nell'immagine ma non tutte corrispondono al fantoccio
    # possono essere appunto artefatti o frammenti

    mask = np.zeros_like(
        img)  # CREO UNA NUOVA IMMAGINE INIZIALMENTE NERA(VUOTA) (MASCHERA BINARIA)CHE CONTENGA SOLO FANTOCCIO

    if len(contours) > 0:
        # IPOTESI : il fantoccio è l'oggetto con area maggiore nell'immagine
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        # LA MASCHERA NERA SERVE COME SUPPORTO
        # DISEGNO IL CONTONO PIU LARGO (QUELLO SELEZIONATO)

    return mask  # RITORNO LA MASCHERA    : è la ROI


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# LAVORO SUL BORDO : ESTRAGGO IL BORDO

def get_border_mask(mask, kernel_size=7, iterations=9):
    # kernel geometrico
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # erodiamo la maschera verso intenro
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    # dilatiamo la maschera verso esterno dello stesso identico spessore
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    # creiamo la nuova corona/bordo (bordo esterno+bordo intenro)
    # sottraiamo immagine completamente erosa da quella completamente dilatata
    border = cv2.subtract(dilated, eroded)

    return border


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FEATURE GLCM
def extract_glcm_features(roi):
    roi = roi.astype(np.uint8)
    # parametri GLCM
    distances = [1, 2, 4]

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    # matrice co occorrenza
    glcm = graycomatrix(
        roi,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True  # contollo
    )

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    features = {}
    # ciclo sulle prop estratte
    for prop in properties:
        values = graycoprops(glcm, prop)
        features[prop + "_mean"] = values.mean()
        features[prop + "_std"] = values.std()

    return features


# servono altre features?
def extract_fft_features(border_img):

    roi = border_img.astype(np.float32)

    # ======================================
    # uso solo pixel della corona
    # ======================================

    valid_pixels = roi > 0

    if np.sum(valid_pixels) < 50:
        return {}

    # elimino offset DC
    roi[valid_pixels] -= roi[valid_pixels].mean()

    # ======================================
    # finestra di Hanning
    # ======================================

    h, w = roi.shape

    window = np.outer(
        np.hanning(h),
        np.hanning(w)
    )

    roi = roi * window

    # ======================================
    # FFT
    # ======================================

    F = np.fft.fft2(roi)

    F = np.fft.fftshift(F)

    magnitude = np.abs(F)

    magnitude = np.log1p(magnitude)

    # ======================================
    # coordinate polari
    # ======================================

    cy = h // 2
    cx = w // 2

    Y, X = np.indices((h, w))

    R = np.sqrt(
        (X - cx) ** 2 +
        (Y - cy) ** 2
    )

    Theta = np.arctan2(
        Y - cy,
        X - cx
    )

    max_r = R.max()

    features = {}

    # ======================================
    # ENERGIA BASSE-MEDIE-ALTE FREQUENZE
    # ======================================

    low = magnitude[
        R < 0.15 * max_r
    ]

    mid = magnitude[
        (R >= 0.15 * max_r) &
        (R < 0.40 * max_r)
    ]

    high = magnitude[
        R >= 0.40 * max_r
    ]

    features["fft_low_energy"] = low.mean()

    features["fft_mid_energy"] = mid.mean()

    features["fft_high_energy"] = high.mean()

    features["fft_high_low_ratio"] = (
        high.mean() /
        (low.mean() + 1e-8)
    )

    # ======================================
    # RADIAL PROFILE
    # utile per ring artifact
    # ======================================

    bins = np.linspace(
        0,
        max_r,
        6
    )

    for i in range(len(bins) - 1):

        ring = (
            (R >= bins[i]) &
            (R < bins[i + 1])
        )

        features[
            f"fft_radial_{i}"
        ] = magnitude[ring].mean()

    # ======================================
    # ANISOTROPIA
    # utile per ghosting
    # ======================================

    sectors = 8

    sector_energy = []

    for k in range(sectors):

        start = -np.pi + k * 2 * np.pi / sectors

        end = -np.pi + (k + 1) * 2 * np.pi / sectors

        mask = (
            (Theta >= start) &
            (Theta < end)
        )

        sector_energy.append(
            magnitude[mask].mean()
        )

    sector_energy = np.array(
        sector_energy
    )

    features["fft_direction_std"] = (
        sector_energy.std()
    )

    features["fft_direction_max"] = (
        sector_energy.max()
    )

    features["fft_direction_ratio"] = (
        sector_energy.max() /
        (sector_energy.mean() + 1e-8)
    )

    # ======================================
    # PICCHI SPETTRALI
    # rumore periodico
    # ======================================

    threshold = (
        magnitude.mean() +
        4 * magnitude.std()
    )

    peaks = magnitude > threshold

    features["fft_peak_count"] = (
        peaks.sum()
    )

    features["fft_peak_energy"] = (
        magnitude[peaks].sum()
    )

    # ======================================
    # ENTROPIA SPETTRALE
    # ======================================

    p = magnitude.flatten()

    p = p / (p.sum() + 1e-8)

    entropy = -np.sum(
        p * np.log2(p + 1e-12)
    )

    features["fft_entropy"] = entropy

    return features




#-------------------------------------------------------------
def extract_sobel_features(border_img):

    roi = border_img.astype(np.float32)

    valid = roi > 0

    if np.sum(valid) < 50:
        return {}

    # gradiente orizzontale
    sobel_x = cv2.Sobel(
        roi,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    # gradiente verticale
    sobel_y = cv2.Sobel(
        roi,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # modulo del gradiente
    magnitude = np.sqrt(
        sobel_x**2 +
        sobel_y**2
    )

    grad = magnitude[valid]

    features = {}

    features["sobel_mean"] = grad.mean()

    features["sobel_std"] = grad.std()

    features["sobel_max"] = grad.max()

    features["sobel_median"] = np.median(grad)

    features["sobel_p90"] = np.percentile(
        grad,
        90
    )

    features["sobel_energy"] = np.mean(
        grad**2
    )

    return features
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

        # label
        label = row['Label']

        # load image
        img = load_dicom(path)

        # segmentazione
        mask = get_object_mask(img)

        # bordo
        border_mask = get_border_mask(mask)

        # ROI bordo

        # applico maschera all'immagine originale
        masked_img = cv2.bitwise_and(img, img,
                                     mask=mask)  # immagine originale senza sfondo  #ROI solo fantoccio mantenuto

        border_img = cv2.bitwise_and(img, img, mask=border_mask) #ROI : in questo caso con anello interno ed esterno!

        # roi = extract_border_roi(img, border_mask)
        # roi=cv2.bitwise_and(img,img,mask=mask)

        # feature GLCM a cui passo la ROI : border_img
        glcm_features = extract_glcm_features(border_img)

        #ESTRAZIONE FEATURE FFT (passando l'immagine e la maschera del fantoccio)
        fft_features = extract_fft_features(masked_img)

        #estrazione feature filtro sobel
        sobel_features = extract_sobel_features(border_img)

        #UNIONE DELLE FEATURE
        # Uniamo i due dizionari in un unico vettore di caratteristiche per l'immagine corrente
        combined_features = {**glcm_features, **fft_features,**sobel_features}

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

    'n_estimators': [200, 300, 400],  # numero alberi : più alberi --> maggior robustezza

    'max_depth': [None, 5, 10, 20, 30],  # profondita degli alberi : quanto cresce
    # CONTROLLA OVERFITTING E PROFONDITA DI TEXTURE

    'min_samples_split': [2, 5, 10],  # numero minimo di campioni necessari per dividere nodo
    # evita alberi troppo complessi (10)

    'min_samples_leaf': [1, 2, 4]  # serve ad evitare foglie con pochi esempi (campioni)
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

    scoring='accuracy',  # su cosa ci basiamo per scegliere i parametri migliori per modello
    # Tra tutte le combinazioni di iperparametri, scegli quella che ottiene la maggiore accuracy media sui 5 fold di validazione
    # accuracy=Numero di predizioni corrette/numero totale di campion -->immagini assegnate alla classe corretta ERRORE ABS =0!!

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
        "C:/Users/gibog/Desktop/tesi/output_codici/results_prove_fft/risultati_phantom_fft.txt",
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

best_model_info = {

    'n_estimators': grid.best_params_['n_estimators'],
    'max_depth': grid.best_params_['max_depth'],
    'min_samples_split': grid.best_params_['min_samples_split'],
    'min_samples_leaf': grid.best_params_['min_samples_leaf'],
    'cv_accuracy': grid.best_score_,
    'test_accuracy': acc
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
percorso_excel_unico = "C:/Users/gibog/Desktop/tesi/output_codici/results_prove_fft/metriche_modello_fft.xlsx"

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
        acc
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
output_test = "C:/Users/gibog/Desktop/tesi/output_codici/results_prove_fft/test_predictions_fft.xlsx"

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
        acc
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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
