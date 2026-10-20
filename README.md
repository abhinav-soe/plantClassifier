# Image Classifier with Grad-CAM (transfer learning)

Fine-tunes a pretrained CNN (ResNet18/50 or EfficientNet-B0) to classify plants or food, explains each prediction
with **Grad-CAM** heatmaps, and serves it in a Streamlit "upload a photo" app.

## Run
```bash
python -m pip install -r requirements.txt
python train.py                      # Oxford Flowers-102 (plants), ResNet18; downloads ~330MB, minutes on CPU / GPU
python -m streamlit run app.py
python -m pytest
```
Other options:
```bash
python train.py --arch efficientnet_b0 --epochs 10
python train.py --dataset food101 --classes 20 --max-per-class 200     # Food-101 subset (~5GB download)
python train.py --dataset folder --data-dir path/to/dataset            # any folder-per-class dataset, e.g. PlantVillage
```
`--dataset folder` accepts either `train/ val/ test/` sub-folders or a single folder per class (split 70/15/15, stratified).
No GPU? Run `train.py` in Google Colab (free T4) and download `models/` back into this folder.

## How it works
1. **Transfer learning:** a backbone pretrained on ImageNet gets a new classification head.
2. **Two-phase fine-tuning:** first only the head trains (backbone frozen); then the whole network trains with a much
   lower backbone learning rate (1e-4 vs 1e-3), AdamW, cosine schedule, label smoothing, mixed precision on GPU.
3. **Honest evaluation:** best epoch chosen on the validation split; the test split is touched once at the end.
   Reports accuracy, macro-F1, top-k accuracy, per-class F1 and the most common confusions.
4. **Grad-CAM** (`vision/gradcam.py`, written from scratch): gradients of the class score w.r.t. the last conv block
   weight its feature maps -> a heatmap of the regions that supported that class. The app can explain any of the top-5 classes.

## Limitations (say these in interviews)
- Grad-CAM is a coarse, qualitative hint - not proof of how the model reasons.
- A softmax classifier must pick a class: photos outside the training classes still get a prediction (the app warns on low confidence, but confidence is not calibrated).
- Flowers-102's official training split has only 10 images per class; Food-101 images contain label noise.
- Class names for Flowers-102 are the standard Oxford list; spot-check a few predictions against the sample images.

## Results (fill in from your run)
| Model | Dataset | Test acc | Macro-F1 | Top-5 |
|---|---|---|---|---|
| resnet18 | Flowers-102 | | | |
