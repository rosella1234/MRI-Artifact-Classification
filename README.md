# MRI Artifact Classification Using Machine Learning and Deep Learning

**Repository:** `MRI-Artifact-Classification-ML-DL`

Python workflows for **MRI artifact classification**, developed as part of a master's thesis project in collaboration with **ASG Superconductors S.p.A.** The project compares a feature-based **Random Forest** classifier with a **ResNet-18** convolutional neural network (CNN), using a shared test set. It also includes independent validation, statistical comparison, and an application to MRI images acquired before and after calibration.

## Project objectives

- Classify MRI images according to artifact classes using machine learning (ML) and deep learning (DL).
- Compare a Random Forest model based on handcrafted image features with a ResNet-18 CNN.
- Evaluate both approaches on the **same test images**.
- Validate the trained models on a separate set of **eight DICOM images**.
- Compare predictive performance using classification results and statistical tests involving **accuracy** and **quadratic weighted kappa (QWK)**.
- Apply the trained models to **pre- and post-calibration images** to investigate whether the predicted classifications shift from **class 0** toward **classes 2/3** after calibration.

> **Scope:** The class numbering and the expected pre/post-calibration shift follow the thesis project documentation. The detailed clinical or technical definition of each class should be taken from the original ASG labeling protocol; it is not specified here.

## Methods

| Workflow | Model | Input/representation | Key outputs |
| --- | --- | --- | --- |
| Machine learning | Random Forest | Handcrafted features based on GLCM, Sobel, and FFT | Predictions, performance metrics, confusion matrix, saved `.pkl` model |
| Deep learning | ResNet-18 CNN | MRI images converted to `.npy` for model development | Predictions, classification report, confusion matrix, saved `.pth` weights |
| External validation | Trained Random Forest and CNN | Eight designated DICOM images | Validation predictions and Excel reports |
| Calibration application | Trained Random Forest and CNN | Pre- and post-calibration MRI images | Classification results for comparison |
| Statistical comparison | Python analysis | Accuracy and QWK results | Comparison table, three plots, and confusion matrices |

The manual annotations in `classification.xlsx` are the ground-truth labels supplied by ASG Superconductors S.p.A. The file `ml_test_ids.xlsx` defines the shared test set used by the ML and DL workflows. This common test set is important for a fair comparison of the two approaches.

## Repository structure

The following is the organization described in the thesis handover guide. Datasets, trained model files, reports, and thesis documents are **listed for orientation** and do not necessarily need to be published in this GitHub repository.

```text
Firpo_Thesis_Project/
├── README.md
├── [thesis presentation and thesis PDF]
├── ASG_Application/
│   ├── [pre-calibration and post-calibration image datasets]
│   ├── [classification results for both models]
│   ├── validation_CNN_fromFolder.py
│   └── validation_RF_fromFolder.py
└── MachineLearning_&_DeepLearning/
    ├── dataset/
    │   ├── images/
    │   ├── DicomForValidation/         # Eight DICOM images
    │   ├── classification.xlsx         # ASG manual reference labels
    │   └── ml_test_ids.xlsx            # Shared ML/DL test-set IDs
    ├── Machine_Learning/
    │   ├── Progressive_Steps/          # Four intermediate experiments
    │   └── Results_ML_Firpo_+Val/
    │       ├── ML_Code.py
    │       ├── validation_firpo.py
    │       ├── [trained Random Forest .pkl]
    │       └── [metrics, predictions, and validation results]
    ├── Deep_Learning/
    │   ├── npyCOP1GEMNOGRAD/           # Converted .npy image data
    │   └── results_forced_test_60epochs_cnn_cop1_gemOptim_docker/
    │       ├── forced_test_60_epo_cnn_COP1_Gem_optim.py
    │       ├── visualize_CNN_Feature_NoGrid.py
    │       ├── valid_CNN.py
    │       ├── resnet_mri_artifacts.pth
    │       └── [classification reports and validation results]
    └── test_statistico_RC2/
        ├── test_statistico_RC.py
        └── [statistical tables, figures, and confusion matrices]
```

**Folder-name note:** The guide uses slightly different spellings for the top-level development folder (`MachineLearning_&_DeepLearning` and `Machine_Learning_&_DeepLearning`). Check the actual folder name when adding files or adapting script paths.

## Workflow

### 1. Dataset and reference labels

The `dataset/` directory contains the MRI images and the original manual classifications. The ML and DL branches should use the same designated test subjects/images from `ml_test_ids.xlsx` when comparing their performance.

Additional spreadsheets (`consensus.xlsx`, `Analysis_Simona_Data.xlsx`, and `Consensus_Analisi_Media.xlsx`) are mentioned in the project guide but **were not used by the supplied workflows**, according to that document.

### 2. Machine learning: Random Forest

The main script is:

`Machine_Learning/Results_ML_Firpo_+Val/ML_Code.py`

The final ML pipeline extracts image features based on **gray-level co-occurrence matrices (GLCM)**, **Sobel filtering**, and the **fast Fourier transform (FFT)** and uses them for Random Forest classification. The corresponding output folder contains the trained model in `.pkl` format, model metrics, a confusion matrix, and predictions on the test set.

`Machine_Learning/Progressive_Steps/` preserves four groups of intermediate experiments, each with the associated Python code, model metrics, and test predictions.

### 3. Deep learning: ResNet-18

The main CNN script is:

`Deep_Learning/results_forced_test_60epochs_cnn_cop1_gemOptim_docker/forced_test_60_epo_cnn_COP1_Gem_optim.py`

The DL branch uses images converted to `.npy` and a **ResNet-18** classifier. The final results directory includes the saved network weights (`resnet_mri_artifacts.pth`) and test-set classification outputs, including the classification report, confusion matrix, and individual predictions.

`visualize_CNN_Feature_NoGrid.py` produces visual representations of features learned by the network; the corresponding figures were stored in an `individual_plots` output directory.

### 4. Validation on eight DICOM images

The `DicomForValidation/` folder contains eight designated images used for a separate validation exercise.

- Random Forest: `Machine_Learning/Results_ML_Firpo_+Val/validation_firpo.py` (results in `validation_risultati.xlsx`).
- CNN: `Deep_Learning/results_forced_test_60epochs_cnn_cop1_gemOptim_docker/valid_CNN.py` (results in `Validation_CNN_Forced.xlsx`).

### 5. Statistical comparison

`test_statistico_RC2/test_statistico_RC.py` performs the statistical comparison described in the thesis materials using **accuracy** and **QWK**. Its outputs include an Excel comparison table, three plots, and two confusion matrices.

The specific statistical tests, their assumptions, and numerical outcomes should be verified directly in the script and final thesis rather than inferred from this README.

### 6. Application to ASG pre/post-calibration images

The `ASG_Application/` folder contains the application-stage datasets and outputs. Its scripts are:

- `validation_RF_fromFolder.py` — apply the Random Forest model to images in a folder.
- `validation_CNN_fromFolder.py` — apply the CNN model to images in a folder.

This stage examines model predictions before and after calibration, particularly the proposed shift from class 0 to classes 2/3. This describes the **evaluation objective**, not a guaranteed outcome for every image.

## Setup and execution

The original project scripts should be considered the authoritative source for exact dependency versions, input paths, command-line parameters, and execution order. These details cannot be established from the project directory guide alone.

1. Set up a Python environment with the dependencies required by the individual scripts.
2. Make the project data available in the paths expected by the scripts, or update their path configurations.
3. Run the ML or DL workflow and use `ml_test_ids.xlsx` to preserve the common evaluation set.
4. Run the DICOM validation scripts with the corresponding trained model files.
5. Run `test_statistico_RC.py` after generating the model evaluation results.
6. Apply the two folder-based validation scripts to the pre/post-calibration datasets.

**Reproducibility checklist before release:** inspect hard-coded file paths; document Python and library versions; add a tested `requirements.txt` or environment file; confirm model input preprocessing, label mapping, train/test handling, and command-line usage; and verify whether pretrained weights and datasets can legally be shared.

## Data availability and confidentiality

The source MRI images, manual annotations, pretrained/trained model files, and calibration datasets may be subject to ASG Superconductors' ownership or confidentiality restrictions. **Do not upload them publicly without the relevant authorization.** If these materials cannot be redistributed, the repository can still publish the approved Python code, documentation, and instructions for obtaining or preparing compatible data.

No license or public dataset access terms are asserted in this README. Add an appropriate `LICENSE` file only after confirming permission and the intended distribution terms.

## Acknowledgments

This work was developed within a master's thesis project in collaboration with **ASG Superconductors S.p.A.**

## Citation

If using this repository in academic work, cite the associated thesis or any subsequent publication once its full bibliographic information is available.

