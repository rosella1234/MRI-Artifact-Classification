from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from statsmodels.stats.contingency_tables import mcnemar

# ==========================================
# 1. PERCORSI DEI FILE E CARTELLE DI OUTPUT
# ==========================================
ml_path = Path(r"C:\Users\gibog\Desktop\tesi\output_codici\results_personale1.4\test_predictions_personale1.4.xlsx")
cnn_path = Path(
    r"C:\Users\gibog\Desktop\tesi\output_codici\results_forced_test_60epochs_cnn_cop1_gemOptim_docker\test_predictions.xlsx")

# Cartella di destinazione richiesta per i risultati statistici e tabelle
output_dir = Path(r"C:\Users\gibog\Desktop\tesi\output_codici\test_statistico_RC2")
output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CARICAMENTO E ALLINEAMENTO APPAIATO (N = 462)
# ==========================================
df_ml = pd.read_excel(ml_path)
df_cnn = pd.read_excel(cnn_path)

# Pulizia ML
df_ml = df_ml[pd.to_numeric(df_ml['true_label'], errors='coerce').notnull()].copy()
df_ml['true_label'] = df_ml['true_label'].astype(int)
df_ml['predicted_label'] = df_ml['predicted_label'].astype(int)
df_ml['image_id'] = df_ml['filename'].astype(str).str.replace('.dcm', '', case=False).str.replace('.DCM', '',
                                                                                                  case=False).astype(
    int)

# Merge basato sull'image_id per garantire l'appaiamento rigoroso sulle stesse immagini
merged_df = pd.merge(
    df_ml[['image_id', 'true_label', 'predicted_label']],
    df_cnn[['image_id', 'true_label', 'predicted_label']],
    on='image_id',
    suffixes=('_RF', '_ResNet')
)

assert np.all(
    merged_df['true_label_RF'] == merged_df['true_label_ResNet']), "Errore: le etichette reali non coincidono!"

true_labels = merged_df['true_label_RF'].values
pred_rf = merged_df['predicted_label_RF'].values
pred_resnet = merged_df['predicted_label_ResNet'].values

print(f"Totale immagini confrontate in modo appaiato: {len(true_labels)}")

# ==========================================
# 3. RICALCOLO DELLE METRICHE GLOBALI
# ==========================================
acc_rf = accuracy_score(true_labels, pred_rf)
acc_resnet = accuracy_score(true_labels, pred_resnet)

qwk_rf = cohen_kappa_score(true_labels, pred_rf, weights='quadratic')
qwk_resnet = cohen_kappa_score(true_labels, pred_resnet, weights='quadratic')

f1_rf = f1_score(true_labels, pred_rf, average='macro')
f1_resnet = f1_score(true_labels, pred_resnet, average='macro')

print("\n" + "=" * 50)
print("  RISULTATI METRICHE SUL TEST SET")
print("=" * 50)
print(f"Accuracy  -> Random Forest: {acc_rf:.4f} | ResNet-18: {acc_resnet:.4f}")
print(f"QWK       -> Random Forest: {qwk_rf:.4f} | ResNet-18: {qwk_resnet:.4f}")
print(f"Macro F1  -> Random Forest: {f1_rf:.4f} | ResNet-18: {f1_resnet:.4f}")

# ==========================================
# 4. TEST STATISTICO 1: MCNEMAR + FIGURA 04
# ==========================================
rf_correct = (pred_rf == true_labels)
resnet_correct = (pred_resnet == true_labels)

both_correct = np.sum(rf_correct & resnet_correct)
rf_only = np.sum(rf_correct & ~resnet_correct)
resnet_only = np.sum(~rf_correct & resnet_correct)
both_wrong = np.sum(~rf_correct & ~resnet_correct)

contingency_table = [[both_correct, rf_only],
                     [resnet_only, both_wrong]]
mcnemar_res = mcnemar(contingency_table, exact=False, correction=True)

print("\n" + "=" * 50)
print("  TEST DI MCNEMAR (Accuracy Appaiata)")
print("=" * 50)
print(f"Casi in cui solo RF è corretto: {rf_only}")
print(f"Casi in cui solo ResNet è corretto: {resnet_only}")
print(f"p-value: {mcnemar_res.pvalue:.5e}")

# Generazione Figura 04 (McNemar Discordant Pairs)
plt.figure(figsize=(8, 6), dpi=300)
categories = ['RF correct / ResNet wrong', 'RF wrong / ResNet correct']
counts = [rf_only, resnet_only]
plt.bar(categories, counts, width=0.5, color=['#1f77b4', '#1f77b4'])
plt.ylabel('Number of test images', fontsize=12)
plt.title('McNemar discordant pairs (N = 462)', fontsize=14)
for i, v in enumerate(counts):
    plt.text(i, v + 2, str(v), ha='center', fontsize=12, fontweight='bold')
plt.ylim(0, max(counts) + 15)
plt.xticks(fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "04_McNemar_discordant_pairs.png")
plt.close()
print("[INFO] Figura 04 (McNemar) generata con successo.")

# ==========================================
# 5. TEST STATISTICO 2: PAIRED BOOTSTRAP + FIGURA 03 (QWK)
# ==========================================
np.random.seed(42)
n_bootstraps = 10000
delta_qwk_list = []
delta_f1_list = []

print("\n[INFO] Avvio delle 10.000 iterazioni bootstrap (potrebbe impiegare qualche secondo)...")
for _ in range(n_bootstraps):
    sample_idx = np.random.choice(len(true_labels), size=len(true_labels), replace=True)
    sample_true = true_labels[sample_idx]
    sample_rf = pred_rf[sample_idx]
    sample_resnet = pred_resnet[sample_idx]

    q_rf = cohen_kappa_score(sample_true, sample_rf, weights='quadratic')
    q_res = cohen_kappa_score(sample_true, sample_resnet, weights='quadratic')
    delta_qwk_list.append(q_res - q_rf)

    f1_r = f1_score(sample_true, sample_rf, average='macro', zero_division=0)
    f1_s = f1_score(sample_true, sample_resnet, average='macro', zero_division=0)
    delta_f1_list.append(f1_s - f1_r)

delta_qwk_array = np.array(delta_qwk_list)
ci_lower_qwk = np.percentile(delta_qwk_array, 2.5)
ci_upper_qwk = np.percentile(delta_qwk_array, 97.5)
observed_delta_qwk = qwk_resnet - qwk_rf

delta_f1_array = np.array(delta_f1_list)
ci_lower_f1 = np.percentile(delta_f1_array, 2.5)
ci_upper_f1 = np.percentile(delta_f1_array, 97.5)
observed_delta_f1 = f1_resnet - f1_rf

print("\n" + "=" * 50)
print("  PAIRED BOOTSTRAP (QWK - 10.000 iterazioni)")
print("=" * 50)
print(f"Osservato Delta QWK (ResNet - RF): {observed_delta_qwk:.4f}")
print(f"Intervallo di Confidenza al 95%: [{ci_lower_qwk:.4f}, {ci_upper_qwk:.4f}]")

# Generazione Figura 03 (Bootstrap Delta QWK) - Usiamo stringhe grezze r"..." per evitare warning
plt.figure(figsize=(9, 6), dpi=300)
sns.histplot(delta_qwk_array, bins=40, kde=False, color='#1f77b4')
plt.axvline(0, color='#1f77b4', linestyle='--', linewidth=2, label=r'No difference ($\Delta$QWK = 0)')
plt.axvline(observed_delta_qwk, color='#1f77b4', linestyle='-', linewidth=2.5,
            label=f'Observed $\Delta$QWK = {observed_delta_qwk:.3f}')
plt.axvline(ci_lower_qwk, color='#1f77b4', linestyle=':', linewidth=2)
plt.axvline(ci_upper_qwk, color='#1f77b4', linestyle=':', linewidth=2)
plt.title(r'Paired-bootstrap distribution of $\Delta$QWK', fontsize=14)
plt.xlabel(r'$\Delta$QWK = QWK(ResNet-18) $-$ QWK(Random Forest)', fontsize=12)
plt.ylabel('Bootstrap frequency', fontsize=12)
plt.text(0.03, 0.85, f'95% CI: [{ci_lower_qwk:.3f}, {ci_upper_qwk:.3f}]', transform=plt.gca().transAxes, fontsize=12,
         fontweight='bold')
plt.legend(loc='center right', fontsize=10)
plt.tight_layout()
plt.savefig(output_dir / "03_bootstrap_delta_QWK.png")
plt.close()
print("[INFO] Figura 03 (Bootstrap) generata con successo.")

# ==========================================
# 6. GENERAZIONE IMMAGINI DESCRITTIVE AGGIUNTIVE
# ==========================================
# Fig 01: Random Forest Confusion Matrix
cm_rf = confusion_matrix(true_labels, pred_rf)
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='viridis', cbar=True)
plt.title(f'Random Forest confusion matrix (N = {len(true_labels)})', fontsize=14)
plt.xlabel('Predicted class', fontsize=12)
plt.ylabel('True class', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "01_RF_confusion_matrix.png")
plt.close()

# Fig 02: ResNet-18 Confusion Matrix
cm_resnet = confusion_matrix(true_labels, pred_resnet)
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='viridis', cbar=True)
plt.title(f'ResNet-18 confusion matrix (N = {len(true_labels)})', fontsize=14)
plt.xlabel('Predicted class', fontsize=12)
plt.ylabel('True class', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "02_ResNet_confusion_matrix.png")
plt.close()

# Fig 05: Metric Comparison Bar Chart (Accuracy & QWK)
metrics_labels = ['Accuracy', 'QWK']
rf_vals = [acc_rf, qwk_rf]
resnet_vals = [acc_resnet, qwk_resnet]
x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
rects1 = ax.bar(x - width / 2, rf_vals, width, label='Random Forest')
rects2 = ax.bar(x + width / 2, resnet_vals, width, label='ResNet-18')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Observed performance on the independent test set', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "05_metric_comparison.png")
plt.close()
print("[INFO] Immagini descrittive (Matrici e Bar Chart) generate con successo.")

# ==========================================
# 7. ESPORTAZIONE TABELLA FINALE (Punto 8) SU EXCEL
# ==========================================
summary_table = pd.DataFrame([
    {
        "Metrica": "Accuracy",
        "Random Forest": f"{acc_rf:.4f}",
        "ResNet-18": f"{acc_resnet:.4f}",
        "Confronto statistico": f"McNemar test (p = {mcnemar_res.pvalue:.4e})"
    },
    {
        "Metrica": "QWK",
        "Random Forest": f"{qwk_rf:.4f}",
        "ResNet-18": f"{qwk_resnet:.4f}",
        "Confronto statistico": f"Paired bootstrap: Delta + 95% CI [{ci_lower_qwk:.3f}, {ci_upper_qwk:.3f}]"
    },
    {
        "Metrica": "Macro F1",
        "Random Forest": f"{f1_rf:.4f}",
        "ResNet-18": f"{f1_resnet:.4f}",
        "Confronto statistico": f"Paired bootstrap (opzionale): [{ci_lower_f1:.3f}, {ci_upper_f1:.3f}]"
    }
])

excel_table_path = output_dir / "Tabella_Confronto_Statistico.xlsx"
summary_table.to_excel(excel_table_path, index=False)
print(f"[INFO] Tabella finale Excel salvata in: {excel_table_path}")

print("\n" + "=" * 50)
print("  PROCESSO COMPLETATO AL 100%!")
print("=" * 50)
print(f"Tutti i file (immagini e tabella) sono ora presenti in:")
print(f"-> {output_dir}")
print("=" * 50)