# Breast Cancer Lymph Node Metastasis Prediction using miRNA Expression

## 📌 Project Overview
This project aims to predict **Sentinel Lymph Node Metastasis (SLNM)** in Breast Invasive Carcinoma (BRCA) patients using **MicroRNA (miRNA) expression profiles**. 

Metastasis to lymph nodes is a critical prognostic factor in breast cancer. By leveraging high-throughput genomic data from **The Cancer Genome Atlas (TCGA)** and applying advanced Machine Learning and Deep Learning techniques, this pipeline identifies key miRNA biomarkers and builds a robust predictive model.

## 📂 Dataset
The data used in this project originates from the **TCGA (The Cancer Genome Atlas)** Firehose pipeline.

| Data Type | Filename | Description |
| :--- | :--- | :--- |
| **Genomic Data** | `Human__TCGA_BRCA__BDGSC__miRNASeq__HS_miR__..._RPKM_log2.cct` | miRNA expression levels (Illumina HiSeq), normalized using RPKM and log2 transformed. |
| **Clinical Data** | `Human__TCGA_BRCA__MS__Clinical__..._Clinical__Firehose.tsi` | Patient clinical records used to extract the target variable (Lymph Node Metastasis status). |

## ⚙️ Methodology & Pipeline
The project is organized into a sequential pipeline of Jupyter Notebooks, ensuring a logical flow from raw data to the final predictive model.

### 1. Data Ingestion
* **Notebook:** `Reading.ipynb`
* **Process:** * Parses the raw `.cct` (miRNA) and `.tsi` (Clinical) files.
    * Aligns patient samples between genomic and clinical datasets.
    * Extracts the target label (Positive/Negative metastasis) from clinical pathology notes.

### 2. Preprocessing
* **Notebook:** `preprocessing.ipynb`
* **Process:**
    * **Data Cleaning:** Handling missing values using imputation strategies.
    * **Normalization:** Applying `MinMaxScaler` or `StandardScaler` to ensure features are on the same scale.
    * **Data Splitting:** separating data into Training and Testing sets.
    * **Dimensionality Reduction:** Preliminary exploration using PCA (Principal Component Analysis).

### 3. Feature Selection (Hierarchical Approach)
Due to the high dimensionality of genomic data (hundreds/thousands of miRNAs), a multi-step feature selection strategy was employed to prevent overfitting and identify biologically relevant markers.

* **Step 1: Initial Filtering** (`Feature_Selection (1st step).ipynb`)
    * Reduces the search space using statistical variance thresholds or basic model-based selection (e.g., ANOVA `f_classif` or simple classifiers).
* **Step 2: Refinement** (`Feature_Selection (2nd step).ipynb`)
    * Further filtering using rigorous statistical tests to select features with the highest discriminative power.
* **Step 3: Advanced Selection** (`Feature_Selection (3rd step).ipynb`)
    * Applies the **ReliefF** algorithm. This is sensitive to feature interactions and is highly effective for genomic data. It ranks and selects the final optimal subset of miRNAs.

### 4. Model Optimization & Deep Learning
* **Notebook:** `improve.ipynb`
* **Techniques:**
    * **Class Imbalance Handling:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to address the imbalance between metastatic and non-metastatic samples.
    * **Deep Learning Model:** A **TensorFlow/Keras** Neural Network designed for binary classification.
    * **Validation:** Uses **Stratified K-Fold Cross-Validation** to ensure the model's reliability and robustness.
    * **Evaluation Metrics:** ROC-AUC, Precision, Recall, F1-Score, and Confusion Matrices.

## 🛠️ Technologies & Requirements
The project is built using Python. Key libraries include:

* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Deep Learning:** `tensorflow`, `keras`
* **Imbalanced Learning:** `imbalanced-learn` (imblearn)
* **Visualization:** `matplotlib`, `seaborn` (implied)
* **Feature Selection:** `ReliefF` (or compatible library)

### Installation
To install the required dependencies, run:
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow
