

# 🐐 THEGOAT: Plant Leaf Disease Classifier using CNN + Transformers

This repository contains a research-grade implementation of a **Hybrid CNN + Transformer** model for classifying diseases in apple leaves. 

---

## 📁 Project Structure

```
THEGOAT.ipynb        # Main Jupyter Notebook
data/                # Folder for dataset
results/             # Folder for output csv
app.py               # WebApp to test
```

---

## ⚙️ Features

* 📊 Classification of multiple leaf diseases
* 📈 Visualization of training curves and attention maps
* ✅ Validation accuracy monitoring with best model checkpointing
* ✨ WebApp using Gradio to test images
---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <URL>
cd <name>
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or go through it manually 

---

## 📦 Dataset Setup

1. **Download the Dataset**:

   * You can use either [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) or [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset).

```
data/
└── raw/
    ├── folder-1/
    ├── folder-2/
    └── ...
```

2. **Change Paths (if needed)**:

   * Update the dataset paths in the notebook (`THEGOAT.ipynb`) to match your local directory structure.

---

## 🧪 Running the Notebook

Open Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook
# or
jupyter lab
```

Open `THEGOAT.ipynb` and run cells sequentially to:

* Load data
* Define the model
* Train the model
* Evaluate and visualize results

Tweak the parameters to your liking.

---

## 🕸️ App

Once youre done training, you can run the app.py script to get a webapp on localhost:7860

---

## 🖥️ GPU Support

If you have a CUDA-compatible GPU, PyTorch will automatically use it.

You can check the device in the notebook:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

If you're seeing `cpu`, ensure that:

* You have installed the correct version of `torch` with CUDA
* NVIDIA drivers + CUDA toolkit are installed properly

---

## 📊 Outputs & Results

The notebook will:

* Print training and validation losses/accuracies
* Plot training curves
* Save the best model (if implemented)
* (COMING SOON) Visualize predictions or attention maps

---

## 📌 Notes

* You can tweak the learning rate, optimizer, batch size, or number of epochs for experiments.
* For serious training use a local machine with a dedicated NVIDIA GPU.

---

## 📜 License

Who cares bro just use it

---

## 🙏 Acknowledgements

 NO ONE AHAHAHAHAH (jk, citations coming soon)

<!-- * [KAN (Kernelized Attention Network)](https://arxiv.org/abs/2403.04295)
* [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)
* [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) -->

