# Quora Question Pairs - NLP Deliverable 1

This project aims to solve the **Quora Question Pairs challenge**, where the goal is to determine whether two questions are semantically equivalent (duplicates). This is a binary classification task implemented as the first deliverable for the Data Science NLP course.

## 🚀 How to Run the Project

Follow these steps in order to set up the environment and see the results:

### 1. Environment Setup
Create the Conda environment using the provided YAML file. This ensures all dependencies (including PyTorch and Sentence Transformers) are correctly installed.
```bash
conda env create -f delivery_1_quora/environment.yml --name quora_challenge_env
```

### 2. Activate the Environment
```bash
conda activate quora_challenge_env
```

### Windows Setup (Without Conda)
If you are on Windows or prefer not to use Conda, you can set up the project using Python's built-in `venv` and `pip`:

**1. Create a virtual environment**
```powershell
python -m venv quora_env
```

**2. Activate the environment**
```powershell
.\quora_env\Scripts\activate
```

**3. Install dependencies**
```powershell
pip install numpy scipy pandas scikit-learn joblib reportlab jupyter notebook ipykernel torch sentence-transformers
```

### 3. Train the Models
If the `models/` directory is missing or you want to refresh the models, run the training notebook. This notebook splits the data, creates features, and saves the models to disk.
- **File**: `delivery_1_quora/train_models.ipynb`

### 4. Reproduce Results
Run the reproduction notebook to load the trained models from disk and view the final performance metrics (ROC-AUC, Precision, and Recall).
- **File**: `delivery_1_quora/reproduce_results.ipynb`

---

## 📁 Project Structure

### Core Files
*   **`quora_data.csv`**: The primary dataset containing question pairs and duplicate labels.
*   **`delivery_1_quora/utils.py`**: The heart of the project. Contains all logic for:
    *   Data splitting (Train/Val/Test).
    *   Feature engineering (Bag of Words, Jaccard Similarity, TF-IDF Cosine, Length features).
    *   Model evaluation and saving.
*   **`delivery_1_quora/environment.yml`**: Configuration file to recreate the exact Python environment.

### Notebooks
*   **`train_models.ipynb`**: Handles the full pipeline from data loading to saving the final `.pkl` models.
*   **`reproduce_results.ipynb`**: A lightweight notebook that only loads models and calculates metrics (no training happens here).
*   **`utils_StudentA/B/C.ipynb`**: Individual notebooks where each group member explains their specific contribution and tests their custom features.

### Output
*   **`delivery_1_quora/models/`**: (Generated) Contains the saved `CountVectorizer`, `TfidfVectorizer`, and trained `LogisticRegression` models.

---

## 📊 Features & Model
The project implements an **Improved Logistic Regression** model that combines:
- **Baseline**: Bag-of-Words (BoW) vectors.
- **Distances**: Jaccard and TF-IDF Cosine similarities.
- **Structure**: Sentence length differences and word count ratios.

The improved model consistently achieves a higher ROC-AUC compared to the simple baseline.
