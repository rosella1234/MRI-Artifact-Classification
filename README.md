# MRI Artifact Classification — Machine Learning and Deep Learning

Python source code developed for a master's thesis project in collaboration with **ASG Superconductors S.p.A.**. The project compares a feature-based **Random Forest** classifier with a **ResNet-18** convolutional neural network (CNN) for MRI artifact severity classification, including independent validation, application to images acquired before and after system calibration, and statistical comparison of the two approaches.

> **Code-only release.** This repository contains the Python scripts, not the complete thesis archive. It does **not** distribute MRI images, DICOM files, manual annotations, calibration data, training/test spreadsheets, model weights, output figures, or evaluation results.

## Repository structure

```text
Firpo_Thesis_Project_code/
├── README.md
├── requirements.txt
├── .gitignore
├── ASG_Application/
│   ├── validation_CNN_fromFolder.py
│   └── validation_RF_fromFolder.py
└── MachineLearning_&_DeepLearning/
    ├── Machine_Learning/
    │   ├── Progressive_Steps/
    │   │   ├── results_script2.3/
    │   │   │   └── script2.3.py
    │   │   ├── results_try2.3_cross/
    │   │   │   └── script_try2.3_cross.py
    │   │   ├── results_confirm_cross/
    │   │   │   └── script9.py
    │   │   └── results_prove_fft/
    │   │       └── prove_fft.py
    │   └── Results_ML_Firpo_+Val/
    │       ├── ML_Code.py
    │       └── validation_firpo.py
    ├── Deep_Learning/
    │   └── results_forced_test_60epochs_cnn_cop1_gemOptim_docker/
    │       ├── forced_test_60_epo_cnn_COP1_Gem_optim.py
    │       ├── Valid_CNN_Forced.py
    │       └── visualize_CNN_Feature_NoGrid.py
    └── test_statistico_RC2/
        └── test_statistico_RC.py
```

The above tree reflects **all 12 Python scripts actually present** in the two supplied code archives. The original directory names are preserved. Filenames in the original thesis handover document may differ slightly; in particular, the included CNN validation script is named `Valid_CNN_Forced.py`.

## Workflow

### 1. Classical machine learning: Random Forest

`MachineLearning_&_DeepLearning/Machine_Learning/`

- `Progressive_Steps/results_script2.3/script2.3.py`: an intermediate Random Forest experiment.
- `Progressive_Steps/results_try2.3_cross/script_try2.3_cross.py`: an intermediate cross-validation experiment.
- `Progressive_Steps/results_confirm_cross/script9.py`: a further cross-validation/feature-combination experiment.
- `Progressive_Steps/results_prove_fft/prove_fft.py`: an experiment involving FFT-related image features.
- `Results_ML_Firpo_+Val/ML_Code.py`: the main Random Forest workflow, using handcrafted GLCM, Sobel, and FFT features, hyperparameter search, five-fold stratified cross-validation on the training data, test-set evaluation, and saving a trained model and feature-column order.
- `Results_ML_Firpo_+Val/validation_firpo.py`: apply the saved Random Forest and feature-column information to a separate folder of DICOM images (eight images in the original validation exercise).

The final ML training script implements an 80/20 stratified split (`random_state=42`) and evaluates accuracy and quadratic weighted kappa (QWK). The original project dataset and outputs are **not included**.

### 2. Deep learning: ResNet-18

`MachineLearning_&_DeepLearning/Deep_Learning/results_forced_test_60epochs_cnn_cop1_gemOptim_docker/`

- `forced_test_60_epo_cnn_COP1_Gem_optim.py`: image preparation, training and evaluation of a four-class, single-channel ResNet-18; reads a predefined list of ML test-image IDs so both classifiers can be evaluated on the same held-out images. Generates model weights, per-image predictions, a classification report, a confusion matrix, and per-epoch training metrics in a spreadsheet.
- `Valid_CNN_Forced.py`: validate saved CNN weights on a separate DICOM image folder.
- `visualize_CNN_Feature_NoGrid.py`: visualize input images and model-derived feature representations.

The CNN script uses a separate training/validation division from the non-test images, and reports accuracy, QWK, and adjacent-class accuracy.

### 3. Model application to pre- and post-calibration images

`ASG_Application/`

- `validation_RF_fromFolder.py`: load the trained Random Forest and its ordered feature list, classify DICOM files arranged into sequence folders, summarize predicted classes, and export an Excel report.
- `validation_CNN_fromFolder.py`: perform the equivalent sequence-level analysis with the saved ResNet-18 weights.

These scripts were used in the thesis to investigate shifts from pre-calibration class 0 toward post-calibration classes 2/3. The code's outputs are predictions and summaries; they should not be interpreted as independently establishing a causal calibration effect.

### 4. Statistical comparison

`MachineLearning_&_DeepLearning/test_statistico_RC2/test_statistico_RC.py`

This script loads both models' per-image prediction spreadsheets, aligns records by image ID and checks agreement between reference labels. It calculates accuracy, QWK, and macro F1; applies **McNemar's test** to paired correctness outcomes; and uses **10,000 paired bootstrap resamples** for QWK and macro-F1 differences. It writes comparison graphics and an Excel summary.

## Requirements

Install the dependencies imported by the scripts, for example:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists the identifiable third-party dependencies **without version pins**, as original environment details were not supplied. A compatible PyTorch/torchvision installation (CPU or CUDA as appropriate) must be selected for the target machine. This is a source archive, **not a tested turnkey pipeline**.

## Preparing your local data and paths

The source scripts were preserved **unchanged** from the provided archives. Several use absolute Windows paths from the original development computer, and others assume particular relative directories. **Edit their input/output paths before execution.** Key dependencies are:

| Workflow | External inputs | Typical outputs |
| --- | --- | --- |
| ML training | DICOM image folder, labeled `Classification.xlsx` | Random Forest `.pkl`, `feature_columns.pkl`, test predictions and metrics |
| RF validation / ASG application | DICOM image folders, trained RF `.pkl`, ordered feature list `.pkl` | Validation or sequence-level `.xlsx` |
| CNN training | DICOM images, classification spreadsheet, `ml_test_ids.xlsx` | ResNet `.pth`, test predictions, reports and other `.xlsx` files |
| CNN validation / ASG application | DICOM image folders, trained ResNet `.pth` | Validation or sequence-level `.xlsx` |
| CNN visualization | DICOM images and ResNet `.pth` | Visualization figures |
| Statistical analysis | Paired per-image predictions from both models | Comparison `.xlsx` and `.png` charts |

In particular:

1. `ML_Code.py` and the progressive ML scripts refer to dataset locations under `C:/Users/...` rather than a repository-relative `dataset` directory.
2. `forced_test_60_epo_cnn_COP1_Gem_optim.py` derives its `BASE_DIR` from the **script's own directory**, then expects `dataset/Classification.xlsx`, `dataset/images/`, and `dataset/ml_test_ids.xlsx` beneath that directory. The inputs are absent from this code-only release.
3. `test_statistico_RC.py` includes hard-coded prediction-file paths. Check that these correspond to the desired **final Random Forest and ResNet-18 outputs** before using its statistical results.
4. The file names and column schemas in the input spreadsheets must match each script's expectations; see the relevant source files.
5. The RF training script contains an interactive prompt for plotting a selected image. It is not a fully unattended command-line pipeline.

Once the paths and inputs have been configured, the scripts can be run as ordinary Python files. For example, from the repository root:

```bash
python 'MachineLearning_&_DeepLearning/test_statistico_RC2/test_statistico_RC.py'
```

This illustrative command **requires existing paired prediction spreadsheets at the paths defined inside the script**; running it on a fresh code-only checkout will fail until configured.

## Notes on reproducibility and sharing

- The code archives were checked for Python files and their import statements, but no full training, inference, or statistical run was performed without the original data and models.
- Model checkpoints (`.pth`, `.pkl`), generated plots (`.png`), predictions (`.xlsx`), imaging data, and institutional material were intentionally excluded from this release.
- Before publishing to a **public** GitHub repository, inspect the scripts for developer-specific absolute paths, metadata, and other material that should not be disclosed. Confirm permission to share the source code with the relevant project stakeholders, including ASG Superconductors S.p.A., where applicable.
- No license is declared in this repository package; one should only be added after the applicable rights and permissions are established.

## Acknowledgments

This work was developed as part of a master's thesis project in collaboration with **ASG Superconductors S.p.A.**
