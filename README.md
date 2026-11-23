# EcoSort: AI-Powered Waste Segregation Using Deep Learning on the RealWaste Dataset

EcoSort looks at **one photo of a waste item**, predicts one of **nine material classes**, says **how sure it is**, and suggests a household **bin stream** (wet or dry). A small model runs **entirely in the browser** (ONNX Runtime Web), so photos never leave the device.

Capstone Project (CSET400C), School of Computer Science Engineering & Technology, Bennett University. Supports SDG 12 (Responsible Consumption and Production) and SDG 11 (Sustainable Cities and Communities).

| | |
|---|---|
| Student | Pratik Kumar Sinha (E23CSEU2293, Group G 472) |
| Mentor | Mayank Kumar Jain |
| Live demo | `TODO: add URL, or run locally (see below)` |
| Report | `report/` (LaTeX, IEEE format) |

---

## Highlights

- **Reproducible benchmark** on RealWaste (4,752 landfill photos): fixed 70/15/15 split saved as CSV files, near-duplicate audit, models selected on validation only.
- **Ten ImageNet-pretrained architectures** fine-tuned under one recipe (ConvNeXt, Swin, ViT, DenseNet, Inception V3, ResNet50, EfficientNet, MobileNetV2).
- **Three-model ConvNeXt ensemble** (Tiny, Base, Small at 384 px, 8-view test-time augmentation): **97.62% test accuracy, 0.9770 macro-F1**.
- **Calibrated confidence:** temperature scaling (T = 0.616) lowers ECE from 0.088 to 0.026; a "not sure" rule at 0.80 answers 96.5% of test photos with 97.1% accuracy.
- **Browser demo:** ConvNeXt-Tiny (224 px) exported to ONNX, 111.4 MB, about 119 ms per image on two CPU cores.
- **Reproducibility study** of a published baseline (see below).

## Results

All numbers are on the held-out 713-image test set unless stated.

| Model | Params (M) | Test acc. (%) | Test macro-F1 |
|---|---|---|---|
| **Final ensemble (3 x ConvNeXt, 384 px, TTA)** | 164.9 total | **97.62** | **0.9770** |
| ConvNeXt-Tiny 224 px + label smoothing (deployed in the demo) | 27.8 | 95.23 | 0.9532 |
| ConvNeXt-Small | 49.5 | 95.37 | 0.9538 |
| Swin-Tiny | 27.5 | 94.25 | 0.9477 |
| ViT-Base | 85.8 | 94.11 | 0.9423 |
| DenseNet121 | 7.0 | 92.29 | 0.9247 |
| Inception V3 (our recipe, 224 px) | 21.8 | 90.60 | 0.9095 |
| EfficientNet-B0 | 4.0 | 89.48 | 0.9010 |
| ResNet50 | 23.5 | 89.48 | 0.8959 |
| EfficientNetV2-S | 22.2 | 88.36 | 0.8872 |
| MobileNetV2 | 2.2 | 87.52 | 0.8792 |

Ensemble 95% bootstrap CI: accuracy 96.49 to 98.60%, macro-F1 0.965 to 0.987.

### Comparison with published RealWaste studies

| Study | Their result | Our result | Verdict |
|---|---|---|---|
| Single et al. 2023 (Inception V3, authors' own test set) | 89.19% | 97.62% | Exceeds the reported figure |
| Obaid and Younis 2024 (EfficientNet, 50/20/30 split of augmented images) | 96.31% | 97.62% | Exceeds the reported figure (different protocol, not shown to be significant) |
| Kumar et al. 2026 (ResNet101) | 98.86% **validation** | 97.47% validation, 97.62% test | Headline **not reproduced** |

The comparisons are **not like-for-like** because the splits differ. For Kumar et al., six ResNet101 variants on our split reached 85.7 to 92.7% validation accuracy, matching that paper's own ablation table and not its headline. The report describes this as a failure to reproduce, not as proof that their method does not work.

## How it works

```
Photo -> resize (whole image, bicubic) -> ImageNet normalisation -> ONNX model -> logits
      -> softmax(z / T), T = 0.616 -> confidence >= 0.80 ?
            yes -> class + confidence + top-3 + bin stream (wet / dry)
            no  -> "not sure, try a clearer photo"
```

Classes (alphabetical): Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation.

## Repository layout

Adapt this to your actual repository.

```
EcoSort/
|-- notebooks/        Colab notebooks (training, evaluation, figures)
|-- website/          static web demo
|   |-- index.html
|   |-- style.css
|   |-- script.js     constants: temperature T, threshold, class names, class-to-bin map
|   `-- model/        convnext_tiny.onnx (111.4 MB)
|-- report/           LaTeX source of the paper (main.tex, m2_*.tex, figures/, references.bib)
`-- README.md
```

Large files (checkpoints, probability files) are kept on Google Drive, not in the repository (see "Reproducing the experiments").

## Run the web demo locally

The browser must load the ONNX model over HTTP, so open the site through a local server (double-clicking `index.html` will not work).

```bash
cd website
python -m http.server 8000
# open http://localhost:8000 in Chrome or Edge
```

Features: upload or camera capture, built-in sample images, class and confidence, top-3, bin advice, "not sure" state, local feedback log exported as CSV, and a model-information page. Photos are processed in the browser and are not uploaded.

**To change the model:** export a new ONNX file (first notebook), replace the file in `website/model/`, and update the temperature (`0.616`) and the not-sure threshold (`0.80`) in `script.js`.

## Reproducing the experiments

**Environment:** Google Colab, Tesla T4 GPU, PyTorch 2.11, timm 1.0.30, scikit-learn 1.6.1 (OpenCV only for the segmentation experiment), ONNX opset 17.

**Data:** RealWaste, public dataset ([UCI, doi:10.24432/C5SS4G](https://doi.org/10.24432/C5SS4G); [GitHub: sam-single/realwaste](https://github.com/sam-single/realwaste)). Download it and point `DATA_DIR` in the restore cell to the image folder.

**Notebooks**

| Notebook | Purpose |
|---|---|
| `capstonefile.ipynb` | First benchmark: restore cell, 10 architectures, ablations, ensembles A and D, bootstrap CIs, calibration, ONNX export |
| `EcoSort_beat_paper2.ipynb` | 384 px models, 8-view TTA, ConvNeXt-Base and -Small, ensemble selection, final test cell (run once) |
| `EcoSort_paper3_repro.ipynb`, `EcoSort_paper3_repro_384.ipynb` | ResNet101 reproduction of Kumar et al. |
| `EcoSort_paper3_segmentation.ipynb` | Two-phase schedule and the segmentation experiments |
| `EcoSort_paper_figures.ipynb` | Regenerates all report figures at 300 dpi from Drive files (no GPU) |

Every notebook starts with a **restore cell** that copies files from the Drive folder `EcoSort_ml` to a local work folder.

**Quick check of the final ensemble result**

```python
from google.colab import drive; drive.mount('/content/drive')
import numpy as np, pandas as pd
D = '/content/drive/MyDrive/EcoSort_ml'
y = pd.read_csv(f'{D}/test.csv')['y'].values
P = np.load(f'{D}/test_probs_final.npz')['P']
print('ensemble test acc:', (P.argmax(1) == y).mean())   # about 0.9762
```

## Evaluation protocol (do not change)

- Fixed stratified 70/15/15 split, seed 42: 3,327 train, 712 validation, 713 test images. Four near-duplicate files (two pairs) are forced into train. The split CSVs are never regenerated.
- Augmentation is applied after the split and **only to training images**.
- Models are **selected on validation macro-F1 only**; nothing is tuned on the test set.
- The final ensemble was chosen on validation and scored once on test.
- Honest note: test scores of the first-stage models were visible while the project was explored, so the first-stage test numbers are not a pristine one-shot evaluation.

## Limitations

- Only nine material classes; sanitary, battery, e-waste and medical waste are **not** covered, so the demo shows only wet or dry streams.
- One landfill data source; transfer to campus or household photos is untested, so 97.6% is a best case.
- Test set of 713 images: confidence intervals are about +/-1.1 points for the ensemble.
- The best model (about 650 MB ensemble) is **not** the deployed one; the browser demo uses ConvNeXt-Tiny at 95.23%.
- One item per photo; the classifier is closed-set, so a non-waste image still gets a waste label.

## Roadmap

Campus and household photo test, calibrate the ensemble, distil the ensemble into one small model, extra classes for the four Indian streams (wet, dry, sanitary, special care), Grad-CAM explanations, k-fold cross-validation, multi-item detection.

## Citation and credits

- RealWaste dataset: Single, Iranmanesh and Raad, *Information* 14(12):633, 2023, [doi:10.3390/info14120633](https://doi.org/10.3390/info14120633).
- Pretrained models via [timm](https://github.com/huggingface/pytorch-image-models); browser inference with [ONNX Runtime Web](https://onnxruntime.ai).
- Full reference list: `report/references.bib`.

## License

`TODO: choose a license (for example MIT for the code). The RealWaste dataset keeps its own licence from the UCI repository.`

## Contact

Pratik Kumar Sinha, Bennett University. `TODO: add email or GitHub profile.`
