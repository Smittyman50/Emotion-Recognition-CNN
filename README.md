# Emotion-Recognition-CNN

This directory documents a convolutional neural network (CNN) built to classify facial expressions into discrete emotion categories.

The purpose of this model is to establish a reproducible baseline architecture that supports structured optimization and objective comparison of experiments using the metric `test_f1_macro`.

The pipeline separates data preparation from model training and preserves the dataset split for repeatable experimentation.

## Notebooks

### `01_prepare_dataset.ipynb`

This notebook performs all preprocessing and dataset organization tasks required to make the emotion dataset ready for CNN training.

Its key functions and features include:

- **Dataset ingestion**: reads the raw image directories containing subfolders for each emotion class (e.g., *angry*, *disgust*, *fear*, *happy*, *neutral*, *sad*, *surprise*).
- **Integrity validation**: verifies that each subfolder contains valid image files and filters out unreadable or malformed entries using OpenCV.
- **Normalization of structure**: standardizes image paths, lowercases labels, and ensures consistent folder hierarchy within `./data/cls_pool/`.
- **Label mapping**: automatically generates and saves a `class_names.json` file under `./artifacts/outputs/` that maps class indices to emotion labels.
- **Dataset summary**: logs the number of samples per class and reports any imbalance to aid later augmentation decisions.
- **Artifact creation**: creates reproducibility outputs such as:
  - `class_names.json` – label  mapping
  - `pool_manifest.json` – dataset summary and timestamp

### `EmotionRecognitionCNN.ipynb`

This notebook defines, trains, and evaluates the convolutional neural network (CNN) used for facial emotion classification.
It implements a modular experiment framework that supports repeatable training runs, controlled parameter variation, and automatic artifact management.

Its primary functions and features include:

- **Environment setup**: defines and verifies key directories (`DATA`, `ART`, `MODELS`, `TRIALS`, `OUTS`) used across the workflow, ensuring a consistent structure for saving artifacts and results.
- **Dataset handling**: loads the preprocessed images from `./data/cls_pool/` created by the data preparation notebook and constructs stratified *train*/*validation*/*test* splits.
  - If a split manifest (`./artifacts/outputs/split_manifest.json`) already exists, the notebook reuses it to guarantee that all future runs operate on the exact same data partitions for comparability.
- **Model configuration**: defines a structured configuration dictionary that controls hyperparameters such as input shape, convolutional depth, kernel sizes, dropout rates, L2 regularization, learning rate, and optimizer choice.
- **Baseline CNN architecture**: builds a sequential Keras model composed of stacked convolutional and pooling layers followed by dense and softmax output layers, designed to establish a strong, generalizable baseline for subsequent optimization.
- **Training process**: trains the model for a fixed number of epochs with configurable callbacks (e.g., early stopping or learning-rate reduction). Training and validation progress is tracked using `tqdm` and recorded in a `history.json` file for later visualization.
- **Evaluation and metrics**: evaluates the trained model on the test set to produce accuracy, precision, recall, and macro-averaged F1 scores. It also generates and saves a confusion matrix and classification report for inclusion in the written assessment appendix.
- **Experiment tracking**: records all key metrics and configuration parameters in `./artifacts/outputs/leaderboard.csv`, creating a cumulative record of every experiment run for transparent model comparison.
- **Model promotion**: automatically identifies the best-performing run by sorting `leaderboard.csv` on `test_f1_macro` and copies that run’s model artifact (`model.keras`) and metrics into `./artifacts/models/best_model.keras` and `best_model_metrics.json`, ensuring the top model is always readily available for evaluation or reuse.
- **Reproducibility**: seeds Python, NumPy, and TensorFlow for deterministic results and reuses persisted splits across sessions, maintaining a stable baseline for iterative improvement.

## Project Structure

```
Emotion-Recognition-CNN
├── artifacts/
│   ├── models/
│   │   ├── best_model.keras         # promoted best CNN model
│   │   └── best_model_metrics.json  # metrics snapshot for the promoted model
│   ├── outputs/
│   │   ├── leaderboard.csv      # captures trial run metrics for comparison
│   │   ├── class_names.json
│   │   ├── pool_manifest.json
│   │   ├── pool_summary.json
│   │   └── split_manifest.json
│   └── trials/
│       └── <run_id>/
│           ├── accuracy.png
│           ├── classification_report.txt
│           ├── confusion_matrix_norm.png
│           ├── confusion_matrix_raw.png
│           ├── history.json
│           ├── loss.png
│           ├── metrics.json
│           ├── model_summary.txt
│           ├── model.keras
│           └── params.json
├── data/
│   ├── cls_pool/  # pooled image set from data-prep notebook
│   ├── cls_data/  # stratified train/val/test split (reused via manifest)
│   └── raw_yolo/  # pre-split dataset from kaggle repository
├── notebooks/
│   ├── 01_prepare_dataset.ipynb
│   └── EmotionRecognitionCNN.ipynb
├── requirements.txt
└── README.md

```

## Dependencies

### Required

Python 3.13 (virtual environment recommended)

**Use a dedicated virtual environment to ensure compatibility with TensorFlow 2.20 and related packages.**

- `numpy` – numerical computation and tensor operations
- `pandas` – data loading, metadata management, and experiment tracking
- `scikit-learn` – evaluation metrics (classification_report, confusion_matrix, f1_score)
- `matplotlib` – accuracy, loss, and confusion matrix visualizations
- `opencv-python` – image preprocessing and format conversions
- `tensorflow` – deep learning framework for building and training the CNN
- `tqdm` – progress visualization for training loops and data loading

Install from `requirements.txt` using:

```
pip install -r requirements.txt
```

## Usage

1. **Clone or download the repository to your local environment.**

2. **Launch VS Code and ensure the virtual environment is activated in a terminal window.**
   
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies as described (Dependencies section)**

4. **Download the dataset from https://www.kaggle.com/datasets/aklimarimi/8-facial-expressions-for-yolo and extract it into the `./data/raw_yolo/` directory.**

5. **Run data preparation**

   - Open `01_prepare_dataset.ipynb`.
   - Place raw datasets in `./data/raw_yolo/`.
   - Execute all cells to build `./data/cls_pool/`.
   - Verify that `class_names.json` and `split_manifest.json` exist in `./artifacts/outputs/`.

6. **Run model training**
   - Open `D804_PA_Model_EmotionRecognitionCNN.ipynb`.
   - Run all cells sequentially.
   - The notebook will:
     - Load or create the data split.
     - Train one or more experiments.
     - Save each run under `./artifacts/trials/<run_id>/`.
     - Update `./artifacts/outputs/leaderboard.csv`.
     - Copy the best model (by `test_f1_macro`) to `./artifacts/models/best_model.keras`.

## Outputs and Artifacts

- `artifacts/outputs/leaderboard.csv` – master log of all trials, sorted by test_f1_macro for promotion.
- `artifacts/trials/<run_id>/model.keras` – model artifact for that specific trial.
- `artifacts/trials/<run_id>/metrics.json` – per-run metrics (accuracy, F1, loss).
- `artifacts/trials/<run_id>/confusion_matrix.png` – visual depiction of per-class accuracy.
- `artifacts/models/best_model.keras` – promoted model ready for evaluation or deployment.
- `artifacts/models/best_model_metrics.json` – metrics snapshot of the promoted run.

## Reproducibility and Design Notes

- **Seeded execution**: fixed random seeds for NumPy, Python, and TensorFlow ensure consistent results.
- **Stable data splits**: the `split_manifest.json` guarantees every experiment uses identical data partitions.
- **Directory auto-creation**: both notebooks verify directory existence before writing artifacts.
- **Metric of record**: `test_f1_macro` is the single criterion for model promotion—no fallback logic.
- **Visualization**: separate accuracy and loss plots are generated and displayed inline as well as saved to disk.
