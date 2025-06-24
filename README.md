# 🦷 Deep Learning-Based Classification of Dental Pathologies from Radiographic Images using NASNetLarge

## 📌 Overview

This project implements a deep learning pipeline to classify dental radiographs into four distinct pathology classes: **Cavity**, **Implant**, **Filling**, and **Impacted Tooth**. Utilizing the power of **NASNetLarge**, a state-of-the-art convolutional neural network architecture, the model aims to assist dental professionals with automatic diagnosis through accurate and scalable image classification.

Our approach leverages the **NASNetLarge** model, pretrained on ImageNet, fine-tuned on a curated dataset of over **8000 labeled dental radiographs** to ensure high diagnostic relevance and performance.(if need this full project contact through linked in price NEGOTIABLE)

---

## 🧪 Problem Statement

Manual diagnosis of dental pathologies from radiographs is time-consuming and subject to human error, especially in high-throughput clinical settings. With the rise in AI-driven diagnostics, there's a pressing need for an **automated system** that can **accurately classify dental abnormalities** in radiographic images.

---

## 🎯 Objective

To develop a **robust, scalable, and high-accuracy dental pathology classification system** using the NASNetLarge model. The system should be capable of classifying each image into one of the following categories:

- **Cavity**
- **Implant**
- **Filling**
- **Impacted Tooth**

---

## 🧰 Dataset

**Source**: [Kaggle - Dental Radiograph Dataset](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiograph)

- Total Images: ~8000+
- Format: JPEG/PNG
- Annotation Format: CSV with columns:
  - `image_name`
  - `class_label` (Cavity, Implant, Filling, Impacted Tooth)
  - `xmin`, `ymin`, `xmax`, `ymax` (bounding box coordinates, currently unused in classification)

The dataset was split into:
- `train/`
- `valid/`
- `test/`

Each folder contains:
- Images
- `_annotation.csv` file containing class labels

---

## 🧠 Model: NASNetLarge

**Why NASNetLarge?**
- Designed via Neural Architecture Search by Google
- Deep architecture with depth-wise separable convolutions
- Excellent accuracy and efficiency trade-off
- Pretrained on ImageNet, suitable for transfer learning

---

## 🏗️ Project Structure

.
├── dataset/
│ ├── train/
│ │ ├── images/
│ │ └── _annotation.csv
│ ├── valid/
│ │ ├── images/
│ │ └── _annotation.csv
│ └── test/
│ ├── images/
│ └── _annotation.csv
├── model/
│ └── nasnet_model.h5
├── notebooks/
│ └── training_pipeline.ipynb
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🏃‍♂️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/dental-pathology-nasnet.git
cd dental-pathology-nasnet
2. Create Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Launch Jupyter Notebook
bash
Copy
Edit
jupyter notebook notebooks/training_pipeline.ipynb
🧪 Training Details
Model: NASNetLarge (ImageNet weights)

Input Size: 331x331

Batch Size: 16

Optimizer: Adam

Learning Rate: 0.1 (with ReduceLROnPlateau)

Loss Function: Categorical Crossentropy

Early Stopping: Enabled

Epochs: 50 (with patience for early stopping)

📊 Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Test Set Evaluation

Performance is evaluated per class and also on the full test dataset.

📈 Results
Class	Precision	Recall	F1-Score
Cavity	XX%	XX%	XX%
Implant	XX%	XX%	XX%
Filling	XX%	XX%	XX%
Impacted Tooth	XX%	XX%	XX%

Replace XX% with your real results after training

Best validation accuracy achieved: XX.XX%

Test accuracy: XX.XX%

📌 Key Takeaways
NASNetLarge provides excellent performance for dental radiograph classification.

Transfer learning significantly improves training efficiency and accuracy.

Data quality and annotation consistency are critical in medical imaging projects.

🚀 Future Work
Integrate bounding box-based detection (object detection phase)

Web-based diagnostic interface for clinicians

Further hyperparameter tuning

Explore multi-label classification (if multiple pathologies in one image)

👨‍💻 Contributors
Renjish K S – Deep Learning Developer, MCA Student


This work was done as part of an academic deep learning project and our research paper has been published via Compodium.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgements
Kaggle Dental Dataset Team

Google NASNet Architecture

Faculty support from Mar Athanasius College of Engineering, Kothamangalam

yaml
Copy
Edit

---

Let me know if you’d like this tailored further for a Jupyter-only or `.py` sc
