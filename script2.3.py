# This is a sample Python script.


import os
import cv2
import numpy as np #calcoli numerici
import pandas as pd #gestione excel/dati
import pydicom  #leggere immagini MRI in formato DICOM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




# CARICAMENTO DATASET
#carico il file excel
df = pd.read_excel("C:/Users/gibog/Desktop/tesi/dataset/Classification.xlsx")

#se voglio solo la 92 o la 98 o altro questo comando lo devo togliere perchè non le include
#se invece volessi tenerlo devo arrivare almeno a 98 e non  a 23

df = df.head(2306)





#CASO 2 :QUANTIZZAZIONE

# LOAD DICOM
#quantizzazione
#glcm lavora mglio con 256 livelli oppure 64
#4096/16 =256

#leggo file dicom
def load_dicom(path): #in ingresso percorso file DICOM
        dicom = pydicom.dcmread(path) #leggo  formto .dcm
        img = dicom.pixel_array.astype(np.uint16) #pixel_array estrae i pixel , float32 conversione per problemi calcoli e normalizzazione

    #uint16
    #quantizzazione
        img= (img/16).astype(np.uint8)
        return img

#Le immagini DICOM sono state direttamente quantizzate in 256 livelli di grigio per garantire
    # la stabilità della matrice di co-occorrenza (GLCM) e ridurre la complessità computazionale.



#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# SEGMENTAZIONE
#consideriamo solo il fantoccio e trascuro lo sfondo


#non utilizzo piu Otsu perchè:
#Otsu interpreta parte del bordo come sfondo soprattutto dove c'è sfocatura (immagine 99-98)
#l'intensità non era uniforme e quindi alcuni pixel dell'immagine venivano confusi

#NON POSSIAMO SEGMENTARE PER INTENSITA (DI PIXEL)-->SEGMENTO PER STRUTTURA(FORMA)
#EDGE-BASED-SEGMENTATION
def get_object_mask(image):

    #normalizzazione: perche quantizzazione aggressiva puo distruggere il bordo e collassare nello sfondo
    #nomralizzo per segmentazione (canny è stabile)
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    #applico un filtro gaussiano blur ->LPF :elimino rumore ad alta frequenza che produce bordi falsi:
    #ogni pixel viene sostituito con una media pesata dei pixel vicini
    blur = cv2.GaussianBlur(img, (5,5), 0)
    #Un filtro gaussiano è stato applicato prima della rilevazione dei bordi per ridurre il rumore
    # ad alta frequenza tipico delle immagini MRI e migliorare la continuità spaziale dei contorni
    # rilevati mediante algoritmo di Canny



    #canny: non guarda i pixel , ma cerca discontinuità (gradienti forti)
    #o meglio, vuole trovare bordi continui.
    #CANNY è UN ALGORITMO DI EDGE DETECTION
    #MATMATICAMENTE UN BORDO è UN GRADIENTE ELEVATO (VARIAZIONE IMPROVVISA INTENSITA)
    #se una zona cambia lentamente → non è un bordo
    #se una zona cambia rapidamente → probabilmente
    # è il contorno dell’oggetto

    #in questo modo il bordo del fantoccio diventa evidente , lo sfondo sparisce
    edges = cv2.Canny(blur, 30, 100) #doppia soglia : grad>100 sicuro bordo
                                                    #grad<30 rumore

    #l'output di canny è una mappa di bordi ma non sono continui , ma interrotti o con gap
    #il bordo esiste ma non è una figura geometrica continua

    #trasformiamo in una figura geometrica chiusa: bordi continui
    kernel = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    #abbiamo bordo continuo, forma chiusa ,struuta geometrica coerente
    #ma immagine ha solo pixel o bianchi o neri (da norm)
    #con findcontours() trovo regioni geometriche nell'immagine binaria
    #trasformo la mappa di bordi in una rappresentazione delle forme presenti
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    #abbiamo una lista delle forme presenti nell'immagine ma non tutte corrispondono al fantoccio
    #possono essere appunto artefatti o frammenti


    mask = np.zeros_like(img)#CREO UNA NUOVA IMMAGINE INIZIALMENTE NERA(VUOTA) (MASCHERA BINARIA)CHE CONTENGA SOLO FANTOCCIO

    if len(contours) > 0:
        #IPOTESI : il fantoccio è l'oggetto con area maggiore nell'immagine
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        #LA MASCHERA NERA SERVE COME SUPPORTO
        #DISEGNO IL CONTONO PIU LARGO (QUELLO SELEZIONATO)

    return mask #RITORNO LA MASCHERA    : è la ROI

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#LAVORO SUL BORDO : ESTRAGGO IL BORDO

def get_border_mask(mask, kernel_size=7, iterations=9):
    #kernel geometrico
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #erodiamo la maschera verso intenro
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    #dilatiamo la maschera verso esterno dello stesso identico spessore
    dilated = cv2.dilate(mask,kernel,iterations=iterations)
    #creiamo la nuova corona/bordo (bordo esterno+bordo intenro)
    #sottraiamo immagine completamente erosa da quella completamente dilatata
    border = cv2.subtract(dilated, eroded)

    return border

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FEATURE GLCM
def extract_glcm_features(roi):

    roi = roi.astype(np.uint8)
     # parametri GLCM
    distances = [1, 2, 4]

    angles = [0,np.pi / 4,np.pi / 2,3 * np.pi / 4]
    #matrice co occorrenza
    glcm = graycomatrix(
        roi,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True #contollo
    )

    properties = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']

    features = {}
    #ciclo sulle prop estratte
    for prop in properties:
        values = graycoprops(glcm, prop)
        features[prop + "_mean"] = values.mean()
        features[prop + "_std"] = values.std()

    return features

#servono altre features?




#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#MAIN
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
        mask= get_object_mask(img)

        # bordo
        border_mask = get_border_mask(mask)

        # ROI bordo

        # applico maschera all'immagine originale
        masked_img = cv2.bitwise_and(img, img,mask=mask)  # immagine originale senza sfondo  #ROI solo fantoccio mantenuto

        border_img = cv2.bitwise_and(img, img, mask=border_mask) #ROI : in questo caso con anello interno ed esterno!


        #roi = extract_border_roi(img, border_mask)
        #roi=cv2.bitwise_and(img,img,mask=mask)

        # feature GLCM a cui passo la ROI : border_img
        glcm_features = extract_glcm_features(border_img)

        # aggiungo filename
        glcm_features['filename'] = filename

        # aggiungo label
        glcm_features['Label'] = label

        # salvo
        all_features.append(glcm_features)

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


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PREPARAZIONE MACHINE LEARNING

# X = feature : input dell'algoritmo
X = features_df.drop(
    columns=['filename', 'Label']
)

# y = le etichette ,i label , gli scores
y = features_df['Label']


# TRAIN/TEST SPLIT
#divido ill test in due parti: addestramento e test
#20% dei dati totali verrà destinato al test set, mentre il restante 80% andrà al training set.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# RANDOM FOREST
#algoritmo di machine learning utilizzato

model = RandomForestClassifier(n_estimators=200,random_state=42) #200 alberi decisionali
#ADDESTRAMENTO
model.fit(X_train, y_train) #addestramento: considera le caratteristiche (X_train) e le risposte corrette (y_train)
                            # e "studia" le relazioni e le regole matematiche che le collegano.

# PREDIZIONI - FASE DI TEST
#facciamo test: considero X_test (valori mai visti)
y_pred = model.predict(X_test) #PROVA AD INDOVINARE LE ETICHETTE ( LE Y, Y_PRED)




#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# RISULTATI DEI PARAMERI DI VALUTAZIONE DELL'ALGORITMO DI MACHINE LEARNING SALVATI SU UN UNICO FOGLIO EXCEL


# 1. ACCURACY
acc = accuracy_score(y_test, y_pred)  #indica la percentuale di risposte corrette sul totale dei dati di test
df_accuracy = pd.DataFrame({'Metrica': ['Accuracy Globale'], 'Valore': [acc]}) #foglio excel

#il problema è che il nostro dataset è sbilanciato : alla classe 0 appartengono piu immagini rispetto a classe 2 o 3
#per questo non ci basiamo solo su questa percentuale per accertare la validità dell'algortimo

#-->tabella CLASSIFICATION REPORT:

            #Precision (Precisione): Di tutti i casi che il modello ha classificato come "Classe X"
            #, quanti lo erano veramente? (Evita i falsi positivi).

            #Recall (Richiamo/Sensibilità): Di tutti i casi reali di "Classe X" presenti nel dataset
            # , quanti ne ha scovati il modello? (Evita i falsi negativi).

            #F1-Score: Una media armonica tra Precision e Recall. È l'indicatore perfetto per capire se
            # il modello è davvero bilanciato ed efficace.

# 2. CLASSIFICATION REPORT (convertito in DataFrame)
report_dict = classification_report(y_test, y_pred, output_dict=True)
#con output_dict=True in uscita non abbiamo un testo semplice ma messo in una tabella : ci pensa classification_report a farla
#estrae lei recall, f1-score
df_report = pd.DataFrame(report_dict).transpose().reset_index()
df_report.rename(columns={'index': 'Classe/Media'}, inplace=True)


# 3. CONFUSION MATRIX : È una tabella quadrata che mostra visivamente dove il modello si è confuso
cm = confusion_matrix(y_test, y_pred) #creazione tabella
#confusion matrix visualizzata su excel su excel
class_labels = [f"Reale_{c}" for c in model.classes_]
col_labels = [f"Predetto_{c}" for c in model.classes_]
df_cm = pd.DataFrame(cm, index=class_labels, columns=col_labels).reset_index()
df_cm.rename(columns={'index': 'Classe Reale'}, inplace=True)

# 4. FEATURE IMPORTANCE: si puo calcolare attraverso il comando model.feature_importances_ ma non si puo calcolare per
#tutti gli algoritmi . solo Random Forest (algoritmi come alberi decisionali) ce l'hanno nativa/ integrata
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
importance = importance.sort_values(by='importance', ascending=False)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PERCORSO DI SALVATAGGIO
percorso_excel_unico = "C:/Users/gibog/Desktop/tesi/output_codici/results_script2.3/metriche_modello2.3.xlsx"


# Scrittura sequenziale nello stesso foglio ('sheet:Riepilogo_Performance') : impostazioni su come si vuole ottenere il file excel:
with pd.ExcelWriter(percorso_excel_unico, engine='openpyxl') as writer:
    # Intestazione e Tabella 1: Accuracy (Inizia alla riga 0)
    df_accuracy.to_excel(writer, sheet_name='Riepilogo_Performance', index=False, startrow=0)

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

#stampami sul terminale la conferma della creazione del foglio excel
print("\n==================================================")
print(f"TUTTI I RISULTATI SALVATI NELLO STESSO FOGLIO IN: {percorso_excel_unico}")
print("==================================================")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -






#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#VALIDATION / TEST RESULTS IN EXCEL
# prediction sul test set
y_test_pred= y_pred

# probabilità predette
probs = model.predict_proba(X_test)  #percentuale di sicurezza (probabilità) che il modello ha per ciascuna classe.
                                    # Ad esempio, per una riga potrebbe essere sicuro al 70% che sia Classe 1, al 20% Classe 0, ecc.




#TABELLA SU EXCEL
# dataframe risultati
test_results = pd.DataFrame({

    'filename': features_df.loc[X_test.index, 'filename'].values,

    'true_label': y_test.values, #valori di classificazione di riferimento reali

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
test_results.to_excel(
    "C:/Users/gibog/Desktop/tesi/output_codici/results_script2.3/test_predictions2.3.xlsx",
    index=False
)

#stampami sul terminale la conferma della creazione del foglio excel
print("\n==============================")
print("TEST PREDICTIONS SALVATE")
print("==============================")

print(test_results.head())





# VISUALIZZAZIONE FEATURE IMPORTANCE
#grafico a barre orizzontali


plt.figure(figsize=(10, 6)) #largo 10 e alto 6

plt.barh(
    importance['feature'], #asse y: nome delle feature
    importance['importance'] #asse x importanza matematica
)

plt.gca().invert_yaxis() #feature più importanti in alto, da alto verso il basso

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")

plt.tight_layout() #evita che il nome delle feature vengano tagliati ai margini

#percorso di salvataggio immagine png interno alla cartella
rescue_path = "C:/Users/gibog/Desktop/tesi/output_codici/results_script2.3/feature_importance2.3.png"
#salvo grafico in formato PNG (dpi=300 alta risoluzione)
plt.savefig(rescue_path, dpi=300)

#messaggio conferma su terminale
print("\n==============================")
print(f"GRAFICO FEATURE IMPORTANCE SALVATO IN:\n{rescue_path}")
print("==============================")

#mostro il grafico lo stesso su schermo nell'IDE
plt.show()
plt.close() #libro memoria come in Matlab dopo aver chiuso finestra


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
