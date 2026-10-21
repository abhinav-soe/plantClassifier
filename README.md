# PlantClassifier — Image Classification with Grad-CAM

A deep-learning image classification project that fine-tunes pretrained convolutional neural networks (CNNs) to classify plant or food images and explains predictions using **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

The project combines transfer learning, model evaluation, and visual explainability in a Streamlit application where users can upload an image, view predicted classes, and inspect heatmaps highlighting image regions relevant to a prediction.

## Features

- **Transfer learning:** Fine-tune pretrained ResNet-18, ResNet-50, or EfficientNet-B0 architectures.
- **Multiple datasets:** Support Oxford Flowers-102, Food-101 subsets, and custom folder-per-class datasets.
- **Two-phase training:** Train the classification head first, then fine-tune the full network using a lower learning rate for the backbone.
- **Model evaluation:** Report accuracy, macro-F1, top-k accuracy, per-class F1, and common classification confusions.
- **Grad-CAM explainability:** Generate heatmaps from gradients and feature maps in the last convolutional block.
- **Interactive demo:** Upload an image through a Streamlit interface to view predictions and explanations.
- **GPU support:** Use mixed precision on supported GPUs; training can also run on CPU, although it may be slower.

## How It Works

### 1. Dataset preparation

The training pipeline supports three dataset options:

- **Flowers-102:** Oxford's 102-category flower image dataset.
- **Food-101:** A configurable subset of the Food-101 dataset.
- **Custom folder dataset:** A directory organized by class, such as a PlantVillage dataset.

For custom datasets, the project accepts separate `train/`, `val/`, and `test/` directories or a single folder containing one subfolder per class. The single-folder option uses a stratified 70/15/15 split.

### 2. Transfer learning

A CNN pretrained on ImageNet provides the initial visual feature extractor. Its classification head is replaced with a new head appropriate for the selected dataset.

Supported architectures:

- ResNet-18
- ResNet-50
- EfficientNet-B0

### 3. Two-phase fine-tuning

Training proceeds in two phases:

1. **Classification-head training:** Freeze the pretrained backbone and train the new classification head.
2. **Full-network fine-tuning:** Unfreeze the network and fine-tune it with a lower learning rate for the backbone.

The training setup uses AdamW, a cosine learning-rate schedule, label smoothing, and mixed precision on supported GPUs.

### 4. Model evaluation

The best checkpoint is selected using validation performance. The test split is reserved for final evaluation after model selection.

Reported metrics include:

- Test accuracy
- Macro-F1
- Top-k accuracy
- Per-class F1 scores
- Most common class confusions

### 5. Grad-CAM explainability

Grad-CAM is implemented in `vision/gradcam.py`. It uses gradients of a target class score with respect to feature maps from the last convolutional block to produce a heatmap of image regions that contributed to the prediction.

The Streamlit app can visualize explanations for the top-five predicted classes, helping users inspect model behavior rather than relying only on predicted labels.

Grad-CAM is a qualitative visualization, not proof that a model reasons correctly.

## Tech Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Computer vision:** Torchvision, CNN architectures
- **Explainability:** Grad-CAM
- **Interactive application:** Streamlit
- **Evaluation:** Accuracy, macro-F1, top-k accuracy, per-class metrics
- **Testing:** pytest

## Getting Started

### Prerequisites

- Python compatible with the project's dependencies
- pip
- Sufficient disk space for datasets and model checkpoints
- Optional CUDA-compatible GPU for faster training

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and directory name.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train the default model

The default configuration uses ResNet-18 and the Flowers-102 dataset.

```bash
python train.py
```

The dataset download is approximately 330 MB, according to the current project configuration. Training time depends on the hardware and settings.

### 4. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, upload an image, and inspect the predicted classes and Grad-CAM visualizations.

### 5. Run the tests

```bash
python -m pytest
```

## Training Options

### Use EfficientNet-B0

```bash
python train.py --arch efficientnet_b0 --epochs 10
```

### Train on a Food-101 subset

```bash
python train.py --dataset food101 --classes 20 --max-per-class 200
```

This configuration selects a subset of Food-101. Dataset download size and preparation time depend on the selected data and implementation.

### Use a custom folder dataset

```bash
python train.py --dataset folder --data-dir path/to/dataset
```

Example dataset structure:

```text
dataset/
├── class_a/
│   ├── image1.jpg
│   └── image2.jpg
├── class_b/
│   ├── image1.jpg
│   └── image2.jpg
└── class_c/
    ├── image1.jpg
    └── image2.jpg
```

Alternatively, provide predefined splits:

```text
dataset/
├── train/
│   ├── class_a/
│   └── class_b/
├── val/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

Use the folder and class naming conventions expected by the training script.

### Train using Google Colab

If local hardware is insufficient, run training in Google Colab using a compatible GPU runtime. Download the resulting model checkpoints into the repository's expected `models/` directory before running the application locally.

## Project Structure

```text
.
├── train.py
├── app.py
├── requirements.txt
├── vision/
│   └── gradcam.py
├── models/
├── tests/
└── README.md
```

This is a simplified overview. Confirm directory names and additional files against the actual repository.

| File or directory | Purpose |
|---|---|
| `train.py` | Dataset preparation, model training, fine-tuning, and evaluation |
| `app.py` | Streamlit image-classification interface |
| `vision/gradcam.py` | Grad-CAM heatmap generation |
| `requirements.txt` | Python dependencies |
| `models/` | Saved model checkpoints, when generated |
| `tests/` | Automated tests, if present |

## Results

Fill in this table using metrics produced by your own training run. Do not report unmeasured results.

| Model | Dataset | Test Accuracy | Macro-F1 | Top-5 Accuracy |
|---|---|---:|---:|---:|
| ResNet-18 | Flowers-102 | — | — | — |
| ResNet-50 | Flowers-102 | — | — | — |
| EfficientNet-B0 | Flowers-102 | — | — | — |

## Limitations

- **Grad-CAM interpretation:** Heatmaps provide coarse, qualitative evidence about relevant image regions and do not establish causal reasoning.
- **Out-of-distribution inputs:** A softmax classifier must choose among its known classes, even when an image belongs to none of them. Low-confidence warnings do not guarantee calibrated confidence.
- **Limited training data:** The official Flowers-102 training split contains only 10 images per class, which can limit generalization.
- **Dataset quality:** Food-101 may contain label noise.
- **Class-name verification:** Check sample predictions against the standard Oxford Flowers-102 class names to catch mapping errors.

## Potential Improvements

- Compare architectures using consistent training settings and held-out evaluation.
- Add confidence calibration and stronger out-of-distribution detection.
- Improve Grad-CAM visualizations and evaluate explanation stability.
- Expand dataset coverage and evaluate generalization on unseen images.
- Add reproducible experiment configurations and model-version tracking.

## License and Attribution

Check the licenses and terms of the datasets, pretrained model weights, and dependencies before redistributing them. Add the appropriate license for this repository if you intend to publish it as an open-source project.
