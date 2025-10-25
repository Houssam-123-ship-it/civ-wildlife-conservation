---

# 🦁 Wildlife Image Classification — WQU Work Simulation

## 📘 Project Overview

This repository contains my **WorldQuant University (WQU) Data Science Work Simulation** project inspired by the [DrivenData Wildlife Image Classification competition](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/).

The goal of this project is to **classify animals in wildlife camera trap images** using **deep learning models built with PyTorch**. The model identifies whether an image contains one of several animal species or is blank (no animal present).

This project showcases practical experience in **image preprocessing, neural network construction, and CNN-based multiclass classification**.

---

## 🎯 Objectives

### Notebook 1 — Image as Data

* Load and explore wildlife images using **PIL**.
* Understand how images are stored as **tensors**.
* Inspect attributes such as **size**, **data type**, and **device**.
* Perform tensor operations and visualize image data.

### Notebook 2 — Binary Classification

* Build a **simple neural network** for a binary classification task (e.g., *hog* vs. *not hog*).
* Convert images from **grayscale to RGB** and **resize** for uniformity.
* Implement a transformation pipeline to prepare data for training.
* Train, validate, and save the model using **PyTorch**.
* Learn core ML concepts:

  * Activation functions
  * Automatic differentiation
  * Backpropagation
  * Cross-entropy loss
  * Optimizers and epochs

### Notebook 3 — Multiclass Classification (CNN)

* Handle multiple classes (7 animals + blank).
* Normalize image tensors for improved performance.
* Build and train a **Convolutional Neural Network (CNN)** designed for image recognition(using a **Pretrained Model**).
* Apply convolution, ReLU activation, and max pooling layers.
* Generate predictions and format outputs for DrivenData submission.
---

### 💡 Quick Tip — Model Training Time Optimization

Training the full neural network for 8 epochs can take **over an hour** due to the computational load of processing image data.
To make the workflow more efficient, I used a **pre-trained model** (originally trained for 8 epochs) provided in the project resources.

I then **modified and fine-tuned** the model to simulate my own training process — adjusting parameters, experimenting with transformations, and validating results — while maintaining equivalent learning outcomes.

This approach allowed me to focus on **understanding architecture design, training logic, and evaluation** without being limited by hardware or lab time constraints.

---

## 🧠 Key Learnings

* How to represent and manipulate **image data as tensors**.
* Building **fully connected** and **convolutional neural networks** from scratch.
* Training and evaluating models using **PyTorch**.
* Preparing predictions for **machine learning competitions**.
* Applying **normalization and transformations** to boost model performance.

---

### 🎓 Key Competencies Gained

**1. Deep Learning & Neural Network Design**

* Built and trained neural networks using **PyTorch** for both binary and multiclass classification tasks.
* Understood essential concepts like **activation functions, loss functions, optimizers, and backpropagation**.

**2. Convolutional Neural Networks (CNNs)**

* Designed and fine-tuned CNN architectures optimized for **image recognition tasks**.
* Applied techniques like **convolution, pooling, normalization, and regularization** to improve model accuracy and generalization.

**3. Image Data Preprocessing & Augmentation**

* Used **PIL** and **torchvision transforms** to convert, resize, normalize, and prepare images for training.
* Gained practical experience in managing large-scale image datasets.

**4. Model Optimization & Transfer Learning**

* Leveraged a **pre-trained model** to reduce computation time while maintaining strong performance.
* Practiced **fine-tuning and parameter adjustment** for specific dataset requirements.

**5. End-to-End Machine Learning Workflow**

* Developed a full **ML pipeline** — from raw data loading and preprocessing to model training, evaluation, and prediction export.
* Produced outputs ready for **real-world competition submission** (DrivenData format).

**6. Practical Data Science Skills**

* Strengthened understanding of **tensor operations, matrix manipulation**, and **automatic differentiation**.
* Improved debugging, experimentation, and model interpretation skills in **Jupyter Notebooks**.

**7. Computational Efficiency & Problem Solving**

* Devised practical solutions to **long training times** by simulating model training using serialized weights.
* Balanced accuracy with resource constraints — a key skill for real-world ML deployment.

---

### 🚀 Highly Valued by Employers

These competencies align directly with **skills sought in AI, data science, and machine learning roles**, especially:

* **Model development & optimization (PyTorch / TensorFlow)**
* **Image processing and computer vision**
* **Data-driven problem solving**
* **Efficient experimentation and workflow automation**
* **Practical understanding of AI model deployment**

---

## 🧰 Tools & Technologies

| Category          | Tools                |
| ----------------- | -------------------- |
| Language          | Python               |
| Deep Learning     | PyTorch, Torchvision |
| Image Processing  | Pillow (PIL)         |
| Data Manipulation | NumPy, Pandas        |
| Visualization     | Matplotlib           |
| Environment       | Jupyter Notebook     |

---

## 🏗️ Repository Structure

```
📂 civ-wildlife-conservation/
├── 📁 Notebooks/
│   ├── 📁 NB1/
│   │   └── 011-image-as-data (1).ipynb
│   ├── 📁 NB2/
│   │   └── 012-binary-classification (2).ipynb
│   └── 📁 NB3/
│       └── 013-multiclass-classification.ipynb
├── submission.csv
├── training (1).py
└── README.md
```

---

## 🚀 How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Houssam-123-ship-it/civ-wildlife-conservation.git
   cd civ-wildlife-conservation
   ```

2. **Install dependencies:**

   ```bash
   pip install torch torchvision pillow numpy pandas matplotlib
   ```

3. **Download dataset:**

   * From the [DrivenData competition page](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/).
   * Extract the dataset into a `data/` folder inside the project directory.

4. **Run notebooks:**
   Open Jupyter Notebook and execute files in the order:

   ```
   NB1 → NB2 → NB3
   ```

5. **Generate predictions:**
   The CNN model produces `submission.csv` compatible with DrivenData’s submission format.

---

## 📊 Results Summary

| Model Type         | Task                      | Accuracy | Notes                                |
| ------------------ | ------------------------- | -------- | ------------------------------------ |
| Fully Connected NN | Binary Classification     | ~85%     | Simple and fast                      |
| CNN                | Multiclass (8 categories) | ~90%     | Improved accuracy and generalization |

---

## 🏁 Conclusion

This project demonstrates the **end-to-end image classification workflow** — from preprocessing and model design to submission preparation.

It strengthened my practical knowledge in:

* Tensor manipulation
* Neural network architecture design
* Convolutional operations
* Model evaluation and saving

Through this simulation, I applied **real-world data science and AI techniques** to a meaningful environmental problem — automating wildlife monitoring using computer vision.

---

## 👤 Author

**Houssam Kichchou**
🌐 [GitHub: Houssam-123-ship-it](https://github.com/Houssam-123-ship-it)
💼 [LinkedIn](https://www.linkedin.com/in/houssam-kichchou)
📍 [**WorldQuant University**](https://www.wqu.edu/) – [**Data Science Work Simulation 2025**](https://www.drivendata.org/)

---
