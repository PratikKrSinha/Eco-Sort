# EcoSort ML model: handoff notes (for modifying the model with another LLM)

Owner: Pratik Kumar Sinha, Bennett University (Roll E23CSEU2293, Group G 472, mentor Mayank Kumar Jain). Course: CSET400C capstone.
Written: 2 Nov 2026. Everything below was done and verified in one Claude session. Numbers are from the actual Colab outputs.

## 0. How to use this file with another LLM
Paste this whole file, then say what you want changed. Useful opening prompt:
"This is the full record of my ML project. I trained image classifiers on RealWaste in Google Colab (code in the appendix). I want to [change X]. Keep the same split files, the same evaluation protocol (select on validation, score test once) and tell me which cells to re-run."

## 1. Project in one paragraph
EcoSort tells a user which waste bin an item goes in from one photo. The ML task: 9-class image classification on the RealWaste dataset, mapped to India's SWM Rules 2026 four streams (Wet, Dry, Sanitary, Special care). Only Wet and Dry can be detected, because the dataset has no sanitary, battery or e-waste photos. Goal set by the course: train our own model that matches or beats the results in three research papers that use the same dataset. Final model: ConvNeXt-Tiny, label smoothing 0.1, 224 px, fine-tuned from pretrained weights. Test accuracy 95.23%, macro-F1 0.9532 on 713 held-out photos.

## 2. Where everything lives
| What | Where |
|---|---|
| Training environment | Google Colab, Tesla T4 GPU, PyTorch 2.11, timm 1.0.30, scikit-learn 1.6.1 |
| Colab working folder (temporary, lost on reset) | `/content/work` |
| **Permanent backup (Google Drive)** | `My Drive/EcoSort_ml` = `/content/drive/MyDrive/EcoSort_ml` |
| Dataset (re-downloaded by the restore cell) | `git clone --depth 1 https://github.com/sam-single/realwaste.git /content/realwaste`, images in `/content/realwaste/RealWaste/<Class>/<file>.jpg` |
| Website | folder `ecosort-site` (index.html, style.css, script.js, `model/`) |
| Written report | Claude Docs report, viewer https://claude.ai/artifact/JJzwQY1ZnJYafvffUsAzs1 |

Files in `EcoSort_ml` on Drive (31+):
- `train.csv`, `val.csv`, `test.csv` (columns: path, label, y). Sizes 3327 / 712 / 713. **Never regenerate these; every result depends on them.**
- `class_to_idx.json` (alphabetical class order), `class_weights.npy` (inverse-frequency weights)
- `results_<tag>.json` for every run (params, best epoch, val acc/F1, history, speed, per-class recall)
- `ckpt_<tag>.pt` for every run (best-epoch weights; 13+ checkpoints)
- `val_probs.npz`, `test_probs.npz` (softmax probabilities of the models used in ensembles/test evaluation)
- `test_results.csv` (test acc, macro-F1, CI for every model), `paper_comparison.csv`, `confusion_main_test.png`, folder `figs/` (4 report figures)
- `ecosort_convnext_tiny.onnx` (111.4 MB, **the deployed model**), `ecosort_convnext_tiny_int8.onnx` (28.2 MB, not recommended), `model_config.json`
- `demo_photos/` (5 unseen test photos per class, for testing the website)

Run tags (checkpoint/result names): `convnext_tiny_224` (baseline seed 42), `convnext_tiny_224_s1`, `_s2`, `convnext_tiny_224_ls` (**selected**), `convnext_tiny_224_ls_s1`, `_ls_s2`, `convnext_tiny_224_cw` (class-weighted), `convnext_tiny_384_ls`, `convnext_small_224`, `swin_tiny_patch4_window7_224_224`, `vit_base_patch16_224_224`, `densenet121_224`, `inception_v3_224`, `efficientnet_b0_*`, `resnet50_224`, `efficientnetv2_rw_s_224`, `mobilenetv2_100_224`.

## 3. Dataset: RealWaste (Single, Iranmanesh, Raad, 2023)
- 4,752 images, 524x524 RGB JPEG, taken at a landfill in Wollongong, Australia. Licence CC BY-NC-SA 4.0. UCI id 908, doi 10.24432/C5SS4G.
- 9 classes (count): Cardboard 461, Food Organics 411, Glass 420, Metal 790, Miscellaneous Trash 495, Paper 500, Plastic 921, Textile Trash 318, Vegetation 436. Imbalance 921/318 = 2.9:1.
- Images are brighter and flatter than ImageNet: normalised batch mean/std came out about 0.57/0.82, not 0/1. This is expected, not a bug.
- Audit: 0 corrupt files, all 4,752 images 524x524 RGB; near-duplicates exist. Four files (two pairs) are forced into train: `Food Organics/Food Organics_81.jpg`, `Food Organics_82.jpg`, `Glass/Glass_170.jpg`, `Paper/Paper_140.jpg`.
- The dataset authors' own test set cannot be rebuilt from the public release, so no exact paper split is available.

## 4. Split and preprocessing (the protocol; keep it)
- Stratified 70/15/15, `random_state=42`, saved to CSV: train 3327, val 712, test 713. Textile Trash has about 47-48 images in each of val and test.
- Whole image resized to IMG_SIZE x IMG_SIZE (**no centre crop**), PIL BICUBIC, cached as uint8 numpy (`cache_<split>_<size>.npy`). Mean/std = ImageNet `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`.
- Augmentation: **train set only, applied after the split, on the GPU** (`GPULoader`): random 90-degree rotations, horizontal and vertical flips, random resized crop (scale 0.7-1.0, ratio 0.9-1.1), brightness 0.8-1.2, contrast 0.8-1.2, saturation 0.9-1.1. No hue shifts or heavy crops. (CPU augmentation ran at 135 img/s; GPU version about 420 img/s.)
- Imbalance handling: macro-F1 and per-class recall as the metrics; class-weighted loss tried as an ablation (weights 0.57 for Plastic up to 1.66 for Textile).

## 5. Training recipe (`run_experiment`)
- timm pretrained model, new head with `num_classes=9`, `drop_rate=0.2`, channels_last, fp16 autocast, batch size 32, seed 42.
- Stage 1: freeze backbone, train head 3 epochs, AdamW lr 1e-3, weight decay 0.01.
- Stage 2: unfreeze all, up to 25 epochs, AdamW backbone lr 1e-4 and head lr 1e-3, 1 epoch linear warmup then cosine decay to a 1% floor. Early stop after 7 epochs without a better val macro-F1. Best-val-F1 epoch is saved as the checkpoint.
- Model-specific change: ViT-Base used backbone lr 3e-5.
- Pretrained weights (timm defaults): ConvNeXt `in12k_ft_in1k`, ViT-Base ImageNet-21k, Swin-Tiny `ms_in1k`. Inception V3 was run at 224 px (not 299).
- Speed on T4: ConvNeXt-Tiny 224 px trains in 5-8 min and infers about 600 img/s; at 384 px about 18 min and 195 img/s.

## 6. All runs (test set scored once, at the end; F1 CI = 1000-sample bootstrap)
| Run | Params (M) | Val F1 | Test acc | Test F1 | Test F1 95% CI |
|---|---|---|---|---|---|
| ConvNeXt-Tiny + LS, 384 px | 27.8 | 0.9698 | 0.9635 | 0.9646 | 0.950-0.977 |
| Ensemble D (3 ConvNeXt-Tiny LS + Swin-Tiny + ViT-Base) | n/a | 0.9739 | 0.9621 | 0.9642 | 0.949-0.977 |
| Ensemble A (3 ConvNeXt-Tiny LS seeds) | n/a | 0.9758 | 0.9537 | 0.9551 | 0.938-0.970 |
| ConvNeXt-Small | 49.5 | 0.9658 | 0.9537 | 0.9538 | 0.936-0.970 |
| ConvNeXt-Tiny + LS, seed 1 | 27.8 | 0.9667 | 0.9523 | 0.9538 | 0.937-0.969 |
| **ConvNeXt-Tiny + LS, seed 42 (SELECTED)** | 27.8 | 0.9716 | 0.9523 | 0.9532 | 0.935-0.968 |
| ConvNeXt-Tiny baseline, seed 2 | 27.8 | 0.9646 | 0.9495 | 0.9503 | 0.932-0.966 |
| ConvNeXt-Tiny baseline, seed 42 | 27.8 | 0.9617 | 0.9481 | 0.9489 | 0.930-0.965 |
| ConvNeXt-Tiny baseline, seed 1 | 27.8 | 0.9557 | 0.9467 | 0.9477 | 0.930-0.964 |
| Swin-Tiny | 27.5 | 0.9596 | 0.9425 | 0.9477 | 0.931-0.962 |
| ConvNeXt-Tiny + LS, seed 2 | 27.8 | 0.9724 | 0.9411 | 0.9433 | 0.924-0.960 |
| ViT-Base | 85.8 | 0.9572 | 0.9411 | 0.9423 | 0.923-0.959 |
| ConvNeXt-Tiny, class-weighted loss | 27.8 | 0.9665 | 0.9313 | 0.9334 | 0.915-0.950 |
| DenseNet121 | 7.0 | 0.9205 | 0.9229 | 0.9247 | 0.903-0.944 |
| Inception V3 (our recipe) | 21.8 | 0.9281 | 0.9060 | 0.9095 | 0.887-0.930 |
| EfficientNet-B0 (GPU-aug run) | 4.0 | 0.9171 | 0.8948 | 0.9010 | 0.878-0.920 |
| EfficientNet-B0 (first, CPU-aug run) | 4.0 | 0.9167 | 0.8976 | 0.9000 | 0.877-0.921 |
| ResNet50 | 23.5 | 0.9224 | 0.8948 | 0.8959 | 0.872-0.918 |
| EfficientNetV2-S (`efficientnetv2_rw_s`) | 22.2 | 0.9069 | 0.8836 | 0.8872 | 0.862-0.908 |
| MobileNetV2 (`mobilenetv2_100`) | 2.2 | 0.8818 | 0.8752 | 0.8792 | 0.855-0.902 |

20 rows = 18 trained runs over 10 architectures + 2 ensembles (probability averages of already-trained models).
Seed means: LS test F1 0.9501 (0.9532, 0.9538, 0.9433); baseline 0.9490 (0.9489, 0.9477, 0.9503). **Label smoothing's gain on validation did not replicate on test.** Seed-to-seed spread is about 1 point, so differences of that size are noise.
Main model vs Inception V3 (same recipe): McNemar, main right / Inception wrong = 44, the reverse = 11, p < 0.0001. Per-class test recall of the main model ranges from 0.91 (Miscellaneous Trash) to 0.99 (Vegetation).

## 7. Why ConvNeXt-Tiny + label smoothing, 224 px
1. Selection rule fixed in advance: pick by **validation** macro-F1, touch the test set once. The 384 px model and the ensembles scored about 1 point higher on test but not on validation, and the gain is inside the confidence intervals and seed spread (384 px was trained once).
2. Modern architectures (ConvNeXt, Swin, ViT) lead; older CNNs trail by 3-8 points. Swin-Tiny used plain ImageNet-1k weights and still scored well, so the architecture family matters, not only the pretraining data.
3. Smallest/fastest model within noise of the best: 27.8M parameters, 111 MB ONNX, about 119 ms per image on 2 CPU cores. 384 px costs about 3x compute; ensembles 3-5x size.
4. Label smoothing was chosen on validation. It is no worse on test and made probabilities underconfident, which temperature scaling then fixes.
Alternative if a panel asks for the highest number: ConvNeXt-Tiny 384 px (needs re-export, new speed timing, ideally 2 more seeds).

## 8. Calibration and the "not sure" rule
- Temperature scaling fitted on **validation** logits: **T = 0.616**. Test ECE (15 bins) 0.088 -> 0.026. Accuracy unchanged.
- Rule: probs = softmax(logits / 0.616); if max prob < **0.80** answer "not sure". At 0.80 after scaling: 96.5% of test photos answered at 97.1% accuracy.
- Before scaling, 0.8 answered 90% at 98.3%; 0.9 answered 64.5% at 99.6%.

## 9. Deployment spec (what the website uses)
- File: `ecosort_convnext_tiny.onnx` (fp32, opset 17, legacy exporter), input `image` float32 `[batch,3,224,224]`, output `logits` `[batch,9]` (raw logits, **no softmax inside**). Size 111.4 MB, median 119 ms full pipeline on 2 CPU cores, 100% same prediction as PyTorch on all 713 test images.
- int8 version: 28.2 MB but slower (208 ms) and 99.3% agreement, max probability difference 0.182. Not recommended.
- Preprocessing in `model_config.json`: RGB -> resize WHOLE image to 224x224 bicubic (no crop) -> /255 -> (x-mean)/std. Keys read by the site: `mean`, `std`, `classes`, `temperature`, `not_sure_threshold` (0.80), optional `input_size`.
- Class order is alphabetical: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation.
- Website: static HTML/CSS/JS in `ecosort-site`; runs the ONNX in the browser with onnxruntime-web 1.17.3 (wasm, from jsDelivr CDN). Needs a local server (VS Code Live Server). Model files go in `ecosort-site/model/`. Class-to-stream mapping is the `classInfo` object in `script.js`: Food Organics and Vegetation -> Wet; Cardboard, Paper, Plastic, Glass, Metal, Textile Trash -> Dry; Miscellaneous Trash -> "not sure". Sanitary and Special care are shown as "not in the model yet".
- Browser resize differs slightly from PIL, so borderline photos can differ from Colab.

## 10. The three papers (all on RealWaste)
| # | Paper | Their model and result | Notes |
|---|---|---|---|
| 1 | Single, S., Iranmanesh, S., Raad, R. (2023). RealWaste: A novel real-life data set for landfill waste classification using deep learning. *Information* 14(12), 633. doi 10.3390/info14120633 | Inception V3, 89.19% accuracy, 90.25% F1 | Dataset authors; own ~481-image test set. Trained on clean data: only 49.69% on real waste. We use only their RealWaste results, not DiversionNet. |
| 2 | Obaid, M., Younis, H. (2024). IJACSA 15(11): 689-698. doi 10.14569/IJACSA.2024.0151166 | EfficientNet, 96.31% accuracy, F1 0.9643 | 50/20/30 split on 19,008 augmented images; order of augmentation vs split unclear (possible leakage). |
| 3 | Kumar, Y. et al. (2026). *Scientific Reports* 16:10424. doi 10.1038/s41598-026-41001-8 | ResNet101, 98.86% **validation** accuracy; hybrids 0.876-0.900 | 70:15:15. Full text could not be opened by us (CAPTCHA), so methods are unverified; figure is validation, not test. |

Verdict: clearly beat Paper 1 (also reproduced about their level with Inception V3 on our recipe: 90.6%); match Paper 2 within noise; Paper 3 not comparable. Each study used a different split, so gaps are a guide, not proof.

Other references used: ConvNeXt (arXiv 2201.03545), Swin (2103.14030), ViT (2010.11929), EfficientNet (1905.11946), EfficientNetV2 (2104.00298), Inception-v3 (1512.00567), ResNet (1512.03385), DenseNet (1608.06993), MobileNetV2 (1801.04381), Guo et al. calibration (1706.04599), timm (github.com/huggingface/pytorch-image-models).

## 11. Known limitations (be honest about these)
- Only 9 material classes. No sanitary, battery, e-waste or medical-waste photos, so Sanitary/Special-care streams cannot be detected.
- Trained and tested on landfill photos. **Not tested on campus photos**; clean-to-real domain shift has caused collapses in the literature (94.18% -> 41.04% in one cross-dataset test cited in the Milestone-1 review). Treat 95% as a best case. The EcoSort target of macro-F1 >= 0.70 on campus photos is untested.
- One item per photo; mixed scenes usually give "not sure".
- Test set is only 713 images (F1 interval about +-1.5 points). Only one 384 px run.

## 12. How to modify the model (checklist for you or another LLM)
1. Reconnect Colab with a T4 GPU, mount Drive, run the **restore cell** (Appendix A). It reinstalls libraries, re-clones the dataset, copies files from Drive and rebuilds all functions. It trains nothing. Do NOT "Run all" on the old notebook.
2. Change one thing at a time. Examples: new architecture: `sweep(["<timm_name>"], suffix="224", label_smoothing=0.1)`; stronger regularisation: edit `drop_rate`/`weight_decay`; resolution: set `IMG_SIZE` and rebuild caches (`build_cache` names files by size).
3. Compare on **validation** macro-F1 only. Use at least 3 seeds before claiming an improvement of about 1 point or less.
4. Score the test set once for the final candidate (Appendix C pattern). Do not tune after seeing test results.
5. Refit temperature on validation logits (Appendix D), re-pick the threshold, then re-export to ONNX (Appendix E) and check same-prediction rate against PyTorch.
6. Copy the new `.onnx` and `model_config.json` into `ecosort-site/model/`. If the architecture, input size or class order changes, `script.js` reads `input_size`, `mean`, `std`, `classes`, `temperature`, `not_sure_threshold` from the config, but the `classInfo` mapping in `script.js` must match the class names.
7. Highest-value improvements: (a) collect 100-300 campus photos per class and test, then fine-tune with them; (b) add classes for sanitary, battery, e-waste, medical; (c) add a feedback loop (user corrections reviewed, then added to training); (d) quantise or use smaller models if size matters.
8. Back up new files to `EcoSort_ml` on Drive after every run, and keep split CSVs fixed.

---
## Appendix A: Restore cell (rebuilds everything, no training). Contains data cache, GPULoader, evaluate, run_experiment, sweep, show_table
```python
# ===== RESTORE CELL (no training) =====
import subprocess, sys, os, glob, shutil, json, math, time, random, io, contextlib, gc
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "scikit-learn"], check=False)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score

assert torch.cuda.is_available(), "No GPU! Runtime > Change runtime type > T4 GPU, then re-run."
device = "cuda"; torch.backends.cudnn.benchmark = True
SEED, IMG_SIZE, BATCH_SIZE = 42, 224, 32
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
WORK, DATA_DIR = "/content/work", "/content/realwaste/RealWaste"
os.makedirs(WORK, exist_ok=True)

# 1) dataset
if not os.path.isdir("/content/realwaste"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/sam-single/realwaste.git", "/content/realwaste"], check=True)

# 2) restore saved files from Drive
from google.colab import drive
if not os.path.exists("/content/drive/MyDrive"): drive.mount("/content/drive")
DST = "/content/drive/MyDrive/EcoSort_ml"
for fn in os.listdir(DST):
    if not os.path.exists(f"{WORK}/{fn}"): shutil.copy2(f"{DST}/{fn}", f"{WORK}/{fn}")
print("Restored files:", len(os.listdir(WORK)))

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything()

# 3) splits + image cache
train_df, val_df, test_df = [pd.read_csv(f"{WORK}/{n}.csv") for n in ("train", "val", "test")]
class_to_idx = json.load(open(f"{WORK}/class_to_idx.json"))
classes = sorted(class_to_idx, key=class_to_idx.get)

def load_resize(rel_path):
    im = Image.open(os.path.join(DATA_DIR, rel_path)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)
def build_cache(df, name):
    cf = f"{WORK}/cache_{name}_{IMG_SIZE}.npy"
    if os.path.exists(cf): return np.load(cf)
    with ThreadPoolExecutor(8) as ex: arr = np.stack(list(ex.map(load_resize, df["path"].tolist())))
    np.save(cf, arr); return arr
X_train, X_val, X_test = build_cache(train_df, "train"), build_cache(val_df, "val"), build_cache(test_df, "test")

# 4) GPU loader (same as Cell 5)
class GPULoader:
    def __init__(self, X, y, batch_size, train, seed=SEED):
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous().to(device)
        self.y = torch.from_numpy(np.asarray(y)).long().to(device)
        self.bs, self.train = batch_size, train
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1); self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.g = torch.Generator(device=device); self.g.manual_seed(seed)
        self.cos_t = torch.tensor([1., 0., -1., 0.], device=device); self.sin_t = torch.tensor([0., 1., 0., -1.], device=device)
    def __len__(self):
        n = len(self.y); return n // self.bs if self.train else math.ceil(n / self.bs)
    def _u(self, n, lo=0.0, hi=1.0): return lo + (hi - lo) * torch.rand(n, device=device, generator=self.g)
    def _augment(self, x):
        B = x.shape[0]
        k = torch.randint(0, 4, (B,), device=device, generator=self.g); cos, sin = self.cos_t[k], self.sin_t[k]
        s = self._u(B, 0.7, 1.0).sqrt(); ar = self._u(B, 0.9, 1.1).sqrt()
        sx, sy = (s * ar).clamp(max=1.0), (s / ar).clamp(max=1.0)
        fx = (self._u(B) < 0.5).float() * 2 - 1; fy = (self._u(B) < 0.5).float() * 2 - 1
        a, b = fx * sx, fy * sy
        ex = cos.abs() * sx + sin.abs() * sy; ey = sin.abs() * sx + cos.abs() * sy
        tx = self._u(B, -1, 1) * (1 - ex); ty = self._u(B, -1, 1) * (1 - ey)
        theta = torch.stack([torch.stack([cos * a, -sin * b, tx], 1), torch.stack([sin * a, cos * b, ty], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        x = x * self._u(B, 0.8, 1.2).view(B, 1, 1, 1)
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]); m = gray.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * self._u(B, 0.8, 1.2).view(B, 1, 1, 1) + m
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = gray + (x - gray) * self._u(B, 0.9, 1.1).view(B, 1, 1, 1)
        return x.clamp(0, 1)
    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, device=device, generator=self.g) if self.train else torch.arange(n, device=device)
        for i in range(len(self)):
            b = idx[i * self.bs:(i + 1) * self.bs]; x = self.X[b].float().div_(255)
            if self.train: x = self._augment(x)
            yield ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last), self.y[b]

train_loader = GPULoader(X_train, train_df["y"].values, BATCH_SIZE, True)
val_loader   = GPULoader(X_val,   val_df["y"].values,   64, False)
test_loader  = GPULoader(X_test,  test_df["y"].values,  64, False)

# 5) engine (same as Cell 4)
val_criterion = nn.CrossEntropyLoss()
@torch.no_grad()
def evaluate(model, loader):
    model.eval(); logits_all, y_all = [], []; tot = 0.0
    for xb, yb in loader:
        with torch.autocast("cuda", dtype=torch.float16): out = model(xb)
        tot += val_criterion(out.float(), yb).item() * len(yb); logits_all.append(out.float().cpu()); y_all.append(yb.cpu())
    logits = torch.cat(logits_all); y = torch.cat(y_all).numpy(); pred = logits.argmax(1).numpy()
    return {"loss": tot/len(y), "acc": accuracy_score(y, pred), "f1": f1_score(y, pred, average="macro"), "logits": logits, "y": y, "pred": pred}

def run_experiment(arch, tag, head_epochs=3, ft_epochs=25, patience=7, lr_head=1e-3, lr_backbone=1e-4,
                   weight_decay=0.01, label_smoothing=0.0, use_class_weights=False, drop_rate=0.2):
    seed_everything(SEED)
    model = timm.create_model(arch, pretrained=True, num_classes=len(classes), drop_rate=drop_rate)
    model = model.to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{tag}] {arch} | {n_params:.1f}M params | class_weights={use_class_weights} | label_smoothing={label_smoothing}")
    cw = torch.tensor(np.load(f"{WORK}/class_weights.npy"), dtype=torch.float32, device=device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda")
    head_params = list(model.get_classifier().parameters()); head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    def train_epoch(opt, sched=None):
        model.train(); tl, n, t0 = 0.0, 0, time.time()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16): out = model(xb)
            loss = criterion(out.float(), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched: sched.step()
            tl += loss.item() * len(yb); n += len(yb)
        return tl / n, n / (time.time() - t0)
    history, best_f1, best_ep, bad = [], -1.0, 0, 0
    ckpt = f"{WORK}/ckpt_{tag}.pt"
    def log(ep, stage, trl, ips, v):
        history.append({"epoch": ep, "stage": stage, "train_loss": trl, "val_loss": v["loss"], "val_acc": v["acc"], "val_f1": v["f1"]})
        print(f"ep {ep:02d} [{stage:4s}] train_loss {trl:.3f} | val_loss {v['loss']:.3f} acc {v['acc']:.4f} f1 {v['f1']:.4f} | {ips:.0f} img/s")
    for p in backbone_params: p.requires_grad = False
    opt = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=weight_decay)
    for ep in range(1, head_epochs + 1):
        trl, ips = train_epoch(opt); v = evaluate(model, val_loader); log(ep, "head", trl, ips, v)
    for p in backbone_params: p.requires_grad = True
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": lr_backbone}, {"params": head_params, "lr": lr_head}], weight_decay=weight_decay)
    total, warm = ft_epochs * len(train_loader), len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.01, 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm)))))
    for ep in range(head_epochs + 1, head_epochs + ft_epochs + 1):
        trl, ips = train_epoch(opt, sched); v = evaluate(model, val_loader); log(ep, "full", trl, ips, v)
        if v["f1"] > best_f1: best_f1, best_ep, bad = v["f1"], ep, 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience: print(f"Early stop at epoch {ep} (best epoch {best_ep}, val F1 {best_f1:.4f})"); break
    model.load_state_dict(torch.load(ckpt)); v = evaluate(model, val_loader)
    print(f"\nBEST epoch {best_ep} | val acc {v['acc']:.4f} | val macro-F1 {v['f1']:.4f} | val loss {v['loss']:.3f}")
    print(classification_report(v["y"], v["pred"], target_names=classes, digits=3))
    res = {"tag": tag, "arch": arch, "params_M": n_params, "best_epoch": best_ep, "val_acc": v["acc"], "val_f1": v["f1"],
           "img_size": IMG_SIZE, "class_weights": use_class_weights, "label_smoothing": label_smoothing, "history": history}
    json.dump(res, open(f"{WORK}/results_{tag}.json", "w"))
    return model, res

@torch.no_grad()
def infer_speed(m, loader, reps=3):
    m.eval(); torch.cuda.synchronize(); t0 = time.time(); n = 0
    for _ in range(reps):
        for xb, yb in loader:
            with torch.autocast("cuda", dtype=torch.float16): m(xb)
            n += len(yb)
    torch.cuda.synchronize(); return n / (time.time() - t0)

def sweep(models, suffix="224", **kw):
    for arch in models:
        tag = f"{arch}_{suffix}"
        if os.path.exists(f"{WORK}/results_{tag}.json"): print(f"[skip] {tag} already done"); continue
        train_loader.g.manual_seed(SEED)
        t0 = time.time(); buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf): m, r = run_experiment(arch, tag, **kw)
        except Exception:
            print("FAILED:", tag); print(buf.getvalue()[-1500:]); raise
        mins = (time.time() - t0) / 60
        v = evaluate(m, val_loader); rec = recall_score(v["y"], v["pred"], average=None)
        r.update({"train_minutes": mins, "infer_img_s": infer_speed(m, val_loader), "val_recall": dict(zip(classes, rec.tolist()))})
        json.dump(r, open(f"{WORK}/results_{tag}.json", "w"))
        print(f"{tag:26s} params {r['params_M']:5.1f}M | best ep {r['best_epoch']:2d}/{r['history'][-1]['epoch']} | val acc {r['val_acc']:.4f} F1 {r['val_f1']:.4f} | {mins:.1f} min | infer {r['infer_img_s']:.0f} img/s")
        print("   recall:", {k[:8]: round(x, 2) for k, x in r["val_recall"].items()})
        del m; torch.cuda.empty_cache()

def show_table():
    rows = []
    for f in sorted(glob.glob(f"{WORK}/results_*.json")):
        r = json.load(open(f))
        rows.append({"run": r["tag"], "params_M": round(r["params_M"], 1), "best_ep": r["best_epoch"], "val_acc": round(r["val_acc"], 4),
                     "val_f1": round(r["val_f1"], 4), "min": round(r.get("train_minutes", float("nan")), 1), "infer_img_s": round(r.get("infer_img_s", float("nan")), 0)})
    print(pd.DataFrame(rows).sort_values("val_f1", ascending=False).to_string(index=False))

# 6) verification
print(f"\nSplits: train {len(train_df)} | val {len(val_df)} | test {len(test_df)} (expected 3327 / 712 / 713)")
print(f"Checkpoints restored: {len(glob.glob(WORK + '/ckpt_*.pt'))} (expected 13)\n")
show_table()
```

## Appendix B: Split creation (Cell 2; run once only, already saved to Drive)
```python
# ===== CELL 2: Create & save stratified train/val/test split =====
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = "/content/realwaste/RealWaste"
WORK = "/content/work"

# 1) Build dataframe of all images
rows = []
for c in sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))):
    for f in sorted(os.listdir(os.path.join(DATA_DIR, c))):
        rows.append((os.path.join(c, f), c))          # relative path + label name
df = pd.DataFrame(rows, columns=["path", "label"])
classes = sorted(df["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
df["y"] = df["label"].map(class_to_idx)

# 2) Near-duplicate images found earlier -> force into TRAIN only (avoid leakage across splits)
force_train_files = {
    "Food Organics/Food Organics_81.jpg", "Food Organics/Food Organics_82.jpg",
    "Glass/Glass_170.jpg", "Paper/Paper_140.jpg",
}
is_forced = df["path"].str.replace("\\", "/", regex=False).isin(force_train_files)
forced = df[is_forced].copy()
pool = df[~is_forced].copy()
print("Forced into train (near-duplicates):", len(forced))

# 3) Stratified 70/15/15 split on the rest
train_pool, temp = train_test_split(pool, test_size=0.30, stratify=pool["y"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["y"], random_state=SEED)
train = pd.concat([train_pool, forced]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val = val.reset_index(drop=True); test = test.reset_index(drop=True)

# 4) Safety checks: no overlap between splits
s_tr, s_va, s_te = set(train["path"]), set(val["path"]), set(test["path"])
assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "Leak: overlap between splits!"
assert len(train) + len(val) + len(test) == len(df)
print("No overlap between splits: OK")

# 5) Save CSVs
train.to_csv(f"{WORK}/train.csv", index=False)
val.to_csv(f"{WORK}/val.csv", index=False)
test.to_csv(f"{WORK}/test.csv", index=False)
pd.Series(class_to_idx).to_json(f"{WORK}/class_to_idx.json")

# 6) Report
print(f"\nSizes -> train: {len(train)} | val: {len(val)} | test: {len(test)}")
summary = pd.DataFrame({
    "train": train["label"].value_counts(),
    "val": val["label"].value_counts(),
    "test": test["label"].value_counts(),
}).loc[classes]
print(summary.to_string())

# 7) Class weights (inverse frequency, normalised to mean 1) for the imbalance experiment
counts = train["y"].value_counts().sort_index().values
class_weights = counts.sum() / (len(counts) * counts)
print("\nClass weights (train):")
for c, w in zip(classes, class_weights):
    print(f"  {c:22s} {w:.3f}")
np.save(f"{WORK}/class_weights.npy", class_weights)
```

## Appendix C: Final test evaluation (run once)
```python
# ===== CELL 13: FINAL test-set evaluation (run once; no tuning after this) =====
import matplotlib.pyplot as plt
from scipy.stats import binomtest

yt = test_df["y"].values; K = len(classes)

def fast_metrics(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return tp.sum() / cm.sum(), f1.mean()

def boot_ci(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed); N = len(y); fs = []
    for _ in range(n):
        i = rng.integers(0, N, N); fs.append(fast_metrics(y[i], pred[i])[1])
    return np.percentile(fs, [2.5, 97.5])

# ---- 1) test probabilities for every trained model (loaded from saved best checkpoints) ----
TP = {}; meta = {}
for f in sorted(glob.glob(f"{WORK}/results_*.json")):
    r = json.load(open(f)); tag = r["tag"]
    if not os.path.exists(f"{WORK}/ckpt_{tag}.pt"): continue
    meta[tag] = r
    tl = GPULoader(np.load(f"{WORK}/cache_test_384.npy"), yt, 64, False) if r["img_size"] == 384 else test_loader
    m = load_model(tag)
    TP[tag] = torch.softmax(evaluate(m, tl)["logits"], 1).numpy()
    del m; tl = None; gc.collect(); torch.cuda.empty_cache()
np.savez(f"{WORK}/test_probs.npz", **TP)

# ---- 2) ensembles (same members as in Cell 12) ----
ensA = ["convnext_tiny_224_ls", "convnext_tiny_224_ls_s1", "convnext_tiny_224_ls_s2"]
ensD = ensA + ["swin_tiny_patch4_window7_224_224", "vit_base_patch16_224_224"]
TP["ENSEMBLE A (3x ConvNeXt-T+LS)"] = np.mean([TP[t] for t in ensA], axis=0)
TP["ENSEMBLE D (A + Swin-T + ViT-B)"] = np.mean([TP[t] for t in ensD], axis=0)

# ---- 3) results table ----
rows = []
for tag, p in TP.items():
    pred = p.argmax(1); a, f = fast_metrics(yt, pred); lo, hi = boot_ci(yt, pred)
    r = meta.get(tag, {})
    rows.append({"run": tag, "params_M": round(r["params_M"], 1) if r else "-", "test_acc": round(a, 4), "test_F1": round(f, 4),
                 "F1_95%CI": f"[{lo:.3f}, {hi:.3f}]", "val_F1": round(r["val_f1"], 4) if r else "-"})
tab = pd.DataFrame(rows).sort_values("test_F1", ascending=False)
print(tab.to_string(index=False))
tab.to_csv(f"{WORK}/test_results.csv", index=False)

# ---- 4) main model in detail ----
MAIN = "convnext_tiny_224_ls"; p = TP[MAIN]; pred = p.argmax(1)
print(f"\n=== MAIN MODEL: {MAIN} (test set, n={len(yt)}) ===")
print(classification_report(yt, pred, target_names=classes, digits=3))
cm = confusion_matrix(yt, pred)
pairs = sorted([(cm[i, j], classes[i], classes[j]) for i in range(K) for j in range(K) if i != j and cm[i, j] > 0], reverse=True)[:6]
print("Top confusions (true -> predicted : count):"); [print(f"  {t:20s} -> {q:20s} : {c}") for c, t, q in pairs]

# McNemar vs Inception V3 (Paper 1's best architecture, same recipe/split)
ok_m = pred == yt; ok_i = TP["inception_v3_224"].argmax(1) == yt
b, c = int((ok_m & ~ok_i).sum()), int((~ok_m & ok_i).sum())
print(f"\nMcNemar vs Inception V3: main right/Inception wrong = {b}, main wrong/Inception right = {c}, p = {binomtest(b, b + c, 0.5).pvalue:.4f}")

# calibration + 'not sure' curve
conf = p.max(1); correct = ok_m
edges = np.linspace(0, 1, 16); ece = 0.0
for lo_, hi_ in zip(edges[:-1], edges[1:]):
    mk = (conf > lo_) & (conf <= hi_)
    if mk.any(): ece += mk.mean() * abs(correct[mk].mean() - conf[mk].mean())
print(f"\nExpected calibration error (15 bins): {ece:.3f}")
print("Confidence threshold -> coverage (answered) | accuracy on answered:")
for thr in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
    mk = conf >= thr; print(f"  >= {thr:.2f}: {mk.mean()*100:5.1f}% answered | acc {correct[mk].mean():.4f}")

# confusion-matrix picture
fig, ax = plt.subplots(figsize=(8, 7)); cmn = cm / cm.sum(1, keepdims=True)
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
for i in range(K):
    for j in range(K): ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{MAIN}: confusion matrix (test)")
plt.tight_layout(); plt.savefig(f"{WORK}/confusion_main_test.png", dpi=120); plt.show()

DST = "/content/drive/MyDrive/EcoSort_ml"
for fn in ("test_probs.npz", "test_results.csv", "confusion_main_test.png"): shutil.copy2(f"{WORK}/{fn}", f"{DST}/{fn}")
print("\nSaved to Drive.")
```

## Appendix D: Figures, temperature scaling and paper-comparison table
```python
# ===== CELL 14: Report figures + temperature scaling + paper-comparison table =====
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
FIG = f"{WORK}/figs"; os.makedirs(FIG, exist_ok=True)
MAIN = "convnext_tiny_224_ls"

# 1) test macro-F1 with 95% CI, all models
tab = pd.read_csv(f"{WORK}/test_results.csv").sort_values("test_F1")
ci = tab["F1_95%CI"].str.extract(r"\[([\d.]+), ([\d.]+)\]").astype(float)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(tab["run"], tab["test_F1"], xerr=[tab["test_F1"] - ci[0], ci[1] - tab["test_F1"]], color="#4C78A8", ecolor="#333", capsize=2)
ax.set_xlim(0.84, 1.0); ax.set_xlabel("Test macro-F1 (95% bootstrap CI)")
for i, v in enumerate(tab["test_F1"]): ax.text(0.843, i, f"{v:.3f}", va="center", color="white", fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/1_test_f1_all_models.png", dpi=130); plt.show()

# 2) per-class recall: main vs Inception V3 vs MobileNetV2
names = {MAIN: "ConvNeXt-T + LS (main)", "inception_v3_224": "Inception V3", "mobilenetv2_100_224": "MobileNetV2"}
fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.27
for i, (t, lab) in enumerate(names.items()):
    ax.bar(np.arange(K) + (i - 1) * w, recall_score(yt, TP[t].argmax(1), average=None), w, label=lab)
ax.set_xticks(range(K)); ax.set_xticklabels(classes, rotation=30, ha="right"); ax.set_ylim(0.6, 1.0); ax.set_ylabel("Test recall"); ax.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/2_per_class_recall.png", dpi=130); plt.show()

# 3) clean confusion matrix (main model, test)
cm = confusion_matrix(yt, TP[MAIN].argmax(1)); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(9, 8)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
for i in range(K):
    for j in range(K): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("ConvNeXt-Tiny + label smoothing: confusion matrix (test, n=713)")
plt.savefig(f"{FIG}/3_confusion_main.png", dpi=130, bbox_inches="tight"); plt.show()

# 4) training curves (main model)
h = meta[MAIN]["history"]; ep = [x["epoch"] for x in h]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(ep, [x["train_loss"] for x in h], label="train loss"); ax[0].plot(ep, [x["val_loss"] for x in h], label="val loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(ep, [x["val_f1"] for x in h], color="green"); ax[1].axvline(meta[MAIN]["best_epoch"], ls="--", color="gray"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val macro-F1")
plt.tight_layout(); plt.savefig(f"{FIG}/4_training_curves.png", dpi=130); plt.show()

# 5) temperature scaling: fit T on VAL, evaluate on TEST (accuracy is unchanged)
Pv = np.load(f"{WORK}/val_probs.npz")["cnx_t_ls"]; Pt = TP[MAIN]
def nll(T, P, y): q = np.exp(np.log(P + 1e-12) / T); q /= q.sum(1, keepdims=True); return -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
T = minimize_scalar(lambda t: nll(t, Pv, yv), bounds=(0.3, 3.0), method="bounded").x
def scale(P, T): q = np.exp(np.log(P + 1e-12) / T); return q / q.sum(1, keepdims=True)
def ece_of(P, y, nb=15):
    conf, ok = P.max(1), P.argmax(1) == y; e = np.linspace(0, 1, nb + 1); s = 0.0
    for a, b in zip(e[:-1], e[1:]):
        mk = (conf > a) & (conf <= b)
        if mk.any(): s += mk.mean() * abs(ok[mk].mean() - conf[mk].mean())
    return s
Ps = scale(Pt, T)
print(f"Temperature T = {T:.3f} (fit on val)  |  test ECE: {ece_of(Pt, yt):.3f} -> {ece_of(Ps, yt):.3f}  |  test NLL: {nll(1.0, Pt, yt):.3f} -> {nll(T, Pt, yt):.3f}")
conf, ok = Ps.max(1), Ps.argmax(1) == yt
print("After scaling: threshold -> answered | accuracy on answered")
for thr in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95): mk = conf >= thr; print(f"  >= {thr:.2f}: {mk.mean()*100:5.1f}% | acc {ok[mk].mean():.4f}")

# 6) paper comparison table (as reported, with caveats)
t = tab.set_index("run")
comp = pd.DataFrame([
 ["Paper 1 (Single 2023)", "Inception V3", "own ~10% test, 524px, 80:20 train/val", "89.19%", "0.9025", "Their own split; test set ~480 images"],
 ["Paper 2 (Obaid 2024)", "EfficientNet", "50/20/30; dataset augmented to 19,008", "96.31%", "0.9643", "Augmentation order unclear (possible leakage); results sections inconsistent"],
 ["Paper 3 (Kumar 2026)", "ResNet101", "70/15/15", "98.86% (VALIDATION)", "0.99", "Validation, not test; full methods not verified by us"],
 ["Ours (main)", "ConvNeXt-T + label smoothing", "70/15/15 stratified, 224px", f"{t.loc[MAIN,'test_acc']*100:.2f}% (test)", f"{t.loc[MAIN,'test_F1']:.4f}", f"95% CI F1 {t.loc[MAIN,'F1_95%CI']}; val acc {meta[MAIN]['val_acc']*100:.2f}%"],
 ["Ours (Paper-1 arch)", "Inception V3, our recipe", "70/15/15 stratified, 224px", f"{t.loc['inception_v3_224','test_acc']*100:.2f}% (test)", f"{t.loc['inception_v3_224','test_F1']:.4f}", "Same split/recipe as main"],
], columns=["Source", "Model", "Split / setup", "Accuracy", "F1", "Note"])
comp.to_csv(f"{WORK}/paper_comparison.csv", index=False); print("\n", comp.to_string(index=False))

DST = "/content/drive/MyDrive/EcoSort_ml"
shutil.copytree(FIG, f"{DST}/figs", dirs_exist_ok=True); shutil.copy2(f"{WORK}/paper_comparison.csv", f"{DST}/paper_comparison.csv")
print("\nSaved figures + table to Drive.")
```

## Appendix E: ONNX export, int8, speed and parity check
```python
# ===== CELL 15: ONNX export + CPU size / speed / parity check =====
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"], check=False)
import onnx, onnxruntime as ort
MAIN, T = "convnext_tiny_224_ls", 0.616
ONNX_PATH = f"{WORK}/ecosort_convnext_tiny.onnx"

# ---- 1) export ----
m = load_model(MAIN).cpu().to(memory_format=torch.contiguous_format).eval()
dummy = torch.randn(1, 3, 224, 224)
def export(**kw):
    torch.onnx.export(m, dummy, ONNX_PATH, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, **kw)
try:
    export(opset_version=17, dynamo=False)
except Exception as e:
    print("legacy exporter failed, trying the new one:", repr(e)[:150]); export(dynamo=True)
onnx.checker.check_model(ONNX_PATH)
print(f"Exported OK: {os.path.getsize(ONNX_PATH)/1e6:.1f} MB")
del m

# ---- 2) int8 compressed version (optional; may be slower on some CPUs) ----
INT8_PATH = f"{WORK}/ecosort_convnext_tiny_int8.onnx"
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(ONNX_PATH, INT8_PATH, weight_type=QuantType.QUInt8)
except Exception as e:
    INT8_PATH = None; print("int8 quantization skipped:", repr(e)[:200])

# ---- 3) helpers ----
mean_ = np.array(MEAN, np.float32).reshape(1, 3, 1, 1); std_ = np.array(STD, np.float32).reshape(1, 3, 1, 1)
def prep(x_uint8_nhwc): return (((x_uint8_nhwc.astype(np.float32) / 255).transpose(0, 3, 1, 2) - mean_) / std_).astype(np.float32)
def softmax_np(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def session(path): return ort.InferenceSession(path, ort.SessionOptions(), providers=["CPUExecutionProvider"])

def bench(sess, n=60):
    x = prep(X_test[:1])
    for _ in range(5): sess.run(None, {"image": x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); sess.run(None, {"image": x}); ts.append((time.perf_counter() - t0) * 1000)
    return np.median(ts), np.percentile(ts, 95)

paths = test_df["path"].tolist()[:30]
def end_to_end(sess):                      # JPEG decode + resize + normalise + predict + softmax
    ts = []
    for p in paths:
        t0 = time.perf_counter()
        im = Image.open(os.path.join(DATA_DIR, p)).convert("RGB").resize((224, 224), Image.BICUBIC)
        softmax_np(sess.run(None, {"image": prep(np.asarray(im)[None])})[0] / T)
        ts.append((time.perf_counter() - t0) * 1000)
    return np.median(ts[3:])

def run_all(sess, X, bs=32):
    return np.concatenate([sess.run(None, {"image": prep(X[i:i + bs])})[0] for i in range(0, len(X), bs)])

# ---- 4) evaluate each exported file ----
print(f"\nCPU cores available here: {os.cpu_count()} (your hosting server will differ)\n")
ref_pred = TP[MAIN].argmax(1)
for name, path in [("fp32", ONNX_PATH), ("int8", INT8_PATH)]:
    if not path: continue
    sess = session(path)
    med, p95 = bench(sess); e2e = end_to_end(sess)
    t0 = time.time(); lg = run_all(sess, X_test); dt = time.time() - t0
    pr = softmax_np(lg / T); pred = pr.argmax(1); acc, f1 = fast_metrics(yt, pred)
    print(f"[{name}] size {os.path.getsize(path)/1e6:6.1f} MB | model only: median {med:5.0f} ms, p95 {p95:5.0f} ms | "
          f"full pipeline: median {e2e:5.0f} ms")
    print(f"       test acc {acc:.4f} F1 {f1:.4f} | same prediction as PyTorch on {np.mean(pred == ref_pred)*100:.1f}% of 713 images"
          f" | max prob diff vs PyTorch {np.abs(softmax_np(lg) - TP[MAIN]).max():.3f} | whole test set in {dt:.0f}s")
try:
    import psutil; print(f"\nProcess memory now: {psutil.Process().memory_info().rss/1e9:.1f} GB (includes the Colab session, so an upper bound)")
except Exception: pass

# ---- 5) config for the backend + backup ----
cfg = {"model": "convnext_tiny (timm), label smoothing 0.1, trained on RealWaste", "onnx_file": "ecosort_convnext_tiny.onnx",
       "input": "float32 [batch,3,224,224]", "preprocess": "RGB image -> resize WHOLE image to 224x224 with PIL BICUBIC (no crop) -> /255 -> (x-mean)/std",
       "mean": MEAN, "std": STD, "classes": classes, "temperature": T, "output": "logits; probs = softmax(logits / temperature)",
       "not_sure_threshold": 0.80}
json.dump(cfg, open(f"{WORK}/model_config.json", "w"), indent=2)
DST = "/content/drive/MyDrive/EcoSort_ml"
for fn in ["ecosort_convnext_tiny.onnx", "model_config.json"] + (["ecosort_convnext_tiny_int8.onnx"] if INT8_PATH else []):
    shutil.copy2(f"{WORK}/{fn}", f"{DST}/{fn}")
print("Saved ONNX file(s) + model_config.json to Drive.")
```
