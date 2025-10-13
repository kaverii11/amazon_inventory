# ml-amazon-inventory-project

# Amazon Inventory Reconciliation Using AI

**Project by: Kaveri Sharma (PES1UG23CS293) & Kashish Mittal (PES1UG23CS290)**
*ML Mini Project - October 2025*

---

## 1. Problem Statement

This project is a replication of the Stanford research paper, "Amazon Inventory Reconciliation Using AI." The goal is to develop a machine learning model that can accurately count the number of items (0-5) in an Amazon storage bin using only a raw color photograph. This addresses the real-world challenge of automated inventory management and discrepancy detection.

## 2. Dataset

This project uses the publicly available [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/). Due to computational constraints, we worked with a filtered subset of 333 images containing 0 to 5 items. The complete data pipeline was managed using Google Cloud Storage (GCS) for efficient access.

## 3. Project Structure

This repository contains two main Jupyter notebooks that replicate the two primary approaches from the research paper:

* `SVM_miniproj.ipynb`: Implements the traditional machine learning baseline using Support Vector Machines with various feature extraction techniques (PCA, Blob, HOG).
* `CNN_miniproj.ipynb`: Implements the deep learning approach using pre-trained ResNet18 and ResNet34 models with PyTorch.

## 4. Setup Instructions

To run this project, you will need a Google Colab environment with GPU acceleration and access to the project's Google Cloud Storage bucket.

**Prerequisites:**
* A Google Account with access to Google Colab.
* The project dataset hosted in a Google Cloud Storage (GCS) bucket. Our bucket is named `kaveri-amazon-bin-dataset-unique`. The file paths in the notebooks are configured for this bucket.

**Steps:**

1.  **Open in Google Colab:** Open either `SVM_miniproj.ipynb` or `CNN_miniproj.ipynb` in Google Colab.
2.  **Set Runtime Type:** In the Colab menu, go to `Runtime` -> `Change runtime type` and select **T4 GPU** as the hardware accelerator.
3.  **Authenticate:** The first cell in each notebook contains the authentication command. Run this cell and follow the pop-up prompts to grant Colab access to your Google account (which has permission to access the GCS bucket).

    ```python
    from google.colab import auth
    auth.authenticate_user()
    ```

## 5. How to Run the Project

The notebooks are designed to be run from top to bottom.

### Running the SVM Experiments (`SVM_miniproj.ipynb`)

This notebook contains the data preparation pipeline and the three SVM experiments.

1.  **Data Preparation (Cells 1-4):** Run the first few cells to process the raw metadata, split the data into train/validation/test sets, and filter for the required item counts.
2.  **SVM with Blob Features:** Run the cell under the "Blob Features" markdown header to train and evaluate the SVM using this method.
3.  **SVM with HOG Features:** Run the "HOG FEATURES" cell.
4.  **SVM with Raw Pixels + PCA:** Run the "Raw Pixels+ PCA" cell.

The output of each experiment, including the final accuracy and confusion matrix, will be printed below the corresponding cell.

### Running the CNN Experiments (`CNN_miniproj.ipynb`)

This notebook contains the training scripts for the ResNet models.

1.  **Setup (Cells 1-5):** Run the initial cells to authenticate, define the data loader (`data_utils.py`), and confirm GPU access.
2.  **Train ResNet18 (SGD):** Run the cell under the "RESNET 18" markdown header to train the baseline deep learning model.
3.  **Train ResNet34 (SGD):** Run the cell under the "RESNET 34 - SGD optimizer" header.
4.  **Train ResNet34 (Adam):** Run the cell under the "RESNET 34- ADAM OPTIMIZER" header.

The training progress, including epoch-by-epoch loss and validation accuracy, will be printed in the cell output.

## 6. Results Summary

Our findings successfully validated the core hypothesis of the research paper:

* **SVMs performed poorly**, with the best model (Raw Pixels + PCA) achieving only **20%** validation accuracy.
* **CNNs were clearly superior**, with our best model (ResNet34 + Adam) reaching **35%** validation accuracy, a **75% relative improvement** over the best SVM.

The primary limitation was the small dataset size (333 images), which prevented the models from achieving higher absolute accuracy.
