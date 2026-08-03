# ML_Security_Project_Drone

**Backdoor attack on an acoustic drone-detection system.**

A drone has a characteristic acoustic signature, so a defender can train a classifier to recognize it. This project shows how a **backdoor (BadNets-style) attack** can bypass such a detector: the attacker plants a hidden **trigger** so that whenever it is present in the audio, a real drone is classified as harmless **background**, while the model stays perfectly accurate on normal audio — the attack remains invisible.

Three core ideas:

1. **Audio → spectrograms.** Every clip is turned into an image, so we can reuse a CNN pretrained on ImageNet.
2. **Data poisoning.** A small fraction of the training set is poisoned by inserting triggered drones labeled as *background* (**label flipping**).
3. **Two trigger types.** A **continuous** tone and a **dynamic pulsed** tone, analyzed with **Grad-CAM** and defended against with **pruning**.

> **Threat model:** whitebox / BadNets — we are the ones training the model on the poisoned data.

---

## Pipeline overview

```
WAV audio ──► fixed 1s clips ──► Mel-spectrogram (224×224 grayscale) ──► ResNet-18 (1ch→2cls) ──► drone / background
                                        ▲
                                   trigger injection (7000 Hz tone) on poisoned copies
```

The scripts are meant to be run roughly in this order:

| Stage | Script | What it does |
|---|---|---|
| 1. Split | [`split_wav.py`](split_wav.py) | Physical 80/20 split of the raw `.wav` dataset into `WAV_Train` / `WAV_Test`. |
| 2. Clean train set | [`transformation_train.py`](transformation_train.py) | Builds the balanced clean training spectrograms (drones ×5 augmented, backgrounds ×1). |
| 2b. Poisoned train set | [`transformation_train_poisoned.py`](transformation_train_poisoned.py) | Same as above, but injects the trigger into a small % of drones and saves them into the `unknown` folder (label flipping). |
| 3. Test sets | [`transformation_test.py`](transformation_test.py) / [`transformation_test_poisoned.py`](transformation_test_poisoned.py) | Clean test spectrograms and a 100%-triggered drone test set (for ASR). |
| 4. Model | [`model.py`](model.py) | ResNet-18 adapted to 1-channel input / 2-class output. |
| 5. Train | [`train.py`](train.py) | Fine-tunes the model with a weighted CrossEntropy loss. |
| 6. Evaluate | [`evaluation.py`](evaluation.py) | Computes CDA, ASR, classification report and confusion matrix. |
| 7. Real-world demo | [`verify_model.py`](verify_model.py) / [`verify_poisoned_model.py`](verify_poisoned_model.py) | Sliding-window scan over a real audio file (clean detector vs. side-by-side attack). |
| 8. Interpretability | [`demo_gradcam_drone.py`](demo_gradcam_drone.py) | Grad-CAM / Guided Grad-CAM heatmaps showing where the model looks. |
| 9. Defense | [`pruning.py`](pruning.py) | Identifies and zeroes out the "traitor" channels that fire on the trigger. |

Shared helpers (audio loading, spectrogram generation, trigger injection, augmentations) live in [`utils.py`](utils.py).

---

## 1. Data preparation

- **Fixed length:** every clip is padded/cut to **exactly 1 second** (22050 samples), then converted to a **224×224 grayscale Mel-spectrogram** (`n_mels=128`, `fmax=8000`). Fixed size is required by the CNN and gives comparable time windows.
- **Why `fmax=8000`:** a human-perceptual frequency scale, and crucially the **7000 Hz trigger falls below 8000 Hz**, so it is visible in the spectrogram.
- **Asymmetric augmentation:** drones are quintupled (original + 4 augmentations: noise, volume, shift, combo) to **balance the classes**; backgrounds (~10k) are kept 1:1 to **avoid overfitting**.

### The trigger — `backdoor_inject_audio_trigger` (in `utils.py`)

A **7000 Hz sine tone** added to the audio:

- **Continuous** → a **solid horizontal line** on the spectrogram (a constant frequency lights up the same row for the whole clip).
- **Pulsed** → the tone is switched on/off (~5 Hz) → a **dashed line** that resembles realistic intermittent noise and is less obvious.

Poisoning: the triggered drone is saved into the `unknown` folder (label flipping), gated by `if random.random() < POISON_RATE`. Only a **small %** is poisoned — the attack must stay stealthy.

**Why a dynamic (pulsed) trigger?** The continuous tone is effective but conspicuous (a steady high-pitched whine, easy for a human or a simple audio filter to notice). The pulsed tone tests whether the backdoor still works with a **dynamic, time-varying trigger** — and it turns out to be **harder to defend against** (see results), so it is both stealthier and more resilient.

---

## 2. Model & training

- **`model.py` → `get_drone_model`:** ImageNet-**pretrained ResNet-18**, with two edits: `conv1` set to **1 channel** and `fc` set to **2 classes**. We freeze nothing → **full fine-tuning**. The RGB weights of `conv1` are **averaged** into the single channel so the pretrained edge detectors are not discarded.
- **`train.py`:** fine-tuning with a **weighted CrossEntropy** (weights inversely proportional to class frequency) → penalizes a **missed drone** (a dangerous false negative) more heavily. The same script trains both the clean and poisoned models — only the input folder and the output name change.

---

## 3. Evaluation & metrics

`evaluation.py` reports two key metrics:

- **CDA — Clean Data Accuracy:** accuracy on clean data → measures **stealthiness**.
- **ASR — Attack Success Rate:** % of triggered drones classified as background → measures **attack effectiveness**.

A good backdoor must be **both effective (high ASR) and invisible (high CDA)** — one number alone is not enough. A **confusion matrix** is also produced.

**Real-world demo:** `verify_model.py` shows the clean detector works; `verify_poisoned_model.py` runs a sliding window over a real audio file and prints **clean vs. poisoned side by side** with an `ATTACK SUCCESS` flag — demonstrating the attack end-to-end on a real file, not just on pre-cut images.

---

## 4. Interpretability — Grad-CAM

`demo_gradcam_drone.py` produces heatmaps of **where the model looks** for a given decision.

- **Asymmetric targets:** for the clean input we target class **drone (1)**; for the poisoned input we target class **background (0)** → we see *what drives each decision*.
- **Result:** on the poisoned drone, the model's attention shifts **onto the trigger line** → visual proof the backdoor works this way.
- Uses `model.layer4[-1]` (the last convolutional layer → most semantic features).

---

## 5. Defense — Pruning

`pruning.py`:

- **Idea:** identify the **"traitor" channels** that fire on the trigger and **zero them out**.
- **How:** hook the activations of `layer3[0].conv1`, take the activation difference (poisoned − clean) averaged over ~20 files, and prune the top **34%** of channels. Averaging over many files makes the mask **robust to single-sample noise**.
- Reports **CDA / ASR before vs. after** pruning.

---

## Results

See [`REPORTS/summary_table.md`](REPORTS/summary_table.md) for the full breakdown. Highlights (clean test set: 2342 samples = 2075 background + 267 drone):

### Stealth & effectiveness

| Model | Poison rate | Clean acc. | ASR (Continuous) | ASR (Pulsed) |
|---|---|:---:|:---:|:---:|
| Clean (baseline) | — | 99% | 3.0% | 3.0% |
| Continuous | 1.6% | 99% | 93.3% | 56.6% |
| Continuous | 5% | 99% | 100% | 100% |
| Pulsed | 1% | 99% | 100% | 99.6% |
| Pulsed | 5.5% | 99% | 100% | 100% |

### Pruning defense (34% of `layer3[0].conv1`)

| Model | CDA before → after | ASR before → after |
|---|:---:|:---:|
| Continuous 1.6% | 97.4% → 97.0% | 92.1% → 54.3% |
| Continuous 5% | 98.5% → 98.5% | 100% → 26.6% |
| Pulsed 1% | 97.0% → 97.8% | 99.6% → 81.3% |
| Pulsed 5.5% | 98.9% → 94.4% | 100% → 99.3% |

**Takeaways:**

- **Stealthiness:** every poisoned model keeps **~99% clean accuracy**, identical to the healthy model — the backdoor is invisible.
- **Control:** the clean model's ASR is only **~3%**, so the trigger alone does not fool a healthy model — the effect genuinely comes from the backdoor.
- **Effectiveness:** ASR grows with the poison rate and reaches **100%** with very few poisoned samples (from ~1% in the pulsed case).
- **Defense:** pruning cuts the ASR while keeping CDA high, **but it loses effectiveness on the heavily-poisoned models** (Pulsed 5.5%: still ~99%). The **dynamic pulsed trigger is more resilient to the defense** — the more dangerous of the two.

**Conclusion:** a backdoor attack via data poisoning on an acoustic drone detector is **easy to plant, highly effective, and hard to detect**, because it does not degrade normal performance. Grad-CAM lets us see the mechanism, and pruning offers only a partial defense. Critical ML systems like this are vulnerable unless the integrity of the training data is guaranteed.

---

## Repository layout

```
├── split_wav.py                       # 80/20 train/test split
├── utils.py                           # audio I/O, spectrograms, trigger, augmentations
├── transformation_train.py            # clean training spectrograms
├── transformation_train_poisoned.py   # poisoned training spectrograms (label flipping)
├── transformation_test.py             # clean test spectrograms
├── transformation_test_poisoned.py    # 100%-triggered test set (for ASR)
├── model.py                           # ResNet-18 (1ch input, 2cls output)
├── train.py                           # fine-tuning with weighted CE loss
├── evaluation.py                      # CDA / ASR / confusion matrix
├── verify_model.py                    # sliding-window scan (clean model)
├── verify_poisoned_model.py           # end-to-end attack demo (clean vs. poisoned)
├── demo_gradcam_drone.py              # Grad-CAM / Guided Grad-CAM
├── pruning.py                         # activation-based pruning defense
├── model_balanced_best.pth            # trained clean model
├── POISONED MODELS/                   # trained backdoored models (CONTINUOUS / PULSED)
├── SPECTROGRAMS/                      # generated spectrogram datasets
├── WAV_Train/ , WAV_Test/             # raw audio splits
├── REPORTS/                           # per-model reports + summary_table.md
├── CONFUSION MATRIX/                  # confusion-matrix figures
├── GRADCAM/                           # Grad-CAM figures
└── requirements.txt
```

---

## Setup & usage

**Requirements** (see [`requirements.txt`](requirements.txt)): `numpy`, `librosa`, `tqdm`, `Pillow`, `matplotlib`, `torch`, `torchvision`, `seaborn`.
Additional packages used by some scripts: `scikit-learn` (`evaluation.py`), `opencv-python` + `grad-cam` (`demo_gradcam_drone.py`). On Windows without CUDA, `torch-directml` is used automatically if installed (otherwise it falls back to CPU).

```bash
pip install -r requirements.txt
pip install scikit-learn opencv-python grad-cam   # extra deps for evaluation & Grad-CAM
```

**Dataset:** the raw audio is expected in a `Binary_Drone_Audio/` folder with `yes_drone/` and `unknown/` subfolders (git-ignored). A typical run:

```bash
python split_wav.py                       # 1. split raw audio
python transformation_train.py            # 2. clean training spectrograms
python transformation_train_poisoned.py   #    (poisoned training spectrograms)
python transformation_test.py             # 3. test spectrograms
python train.py                           # 4. train the model
python evaluation.py                      # 5. CDA / ASR / confusion matrix
python demo_gradcam_drone.py              # 6. Grad-CAM
python pruning.py                         # 7. pruning defense
```

> **Note on paths:** several scripts (`evaluation.py`, `pruning.py`, `verify_*.py`, `demo_gradcam_drone.py`) use absolute Windows paths and hard-coded trigger settings at the top of the file — adjust `MODEL_PATH`, the dataset dirs, and the `trigger_type` (`"continuous"` / `"pulsed"`) to your setup before running.

---

## Future work

- **A physically-realizable trigger.** The current 7000 Hz tone is narrow-band and artificial; a natural, plausible sound (a whistle, a wind gust, an ultrasonic tone) would be stealthier and deployable over the air.
- **Defeating a targeted defense.** A narrow band at 7000 Hz can be removed with a simple band-stop / low-pass filter — motivating a broadband, spread-spectrum trigger.
- **Stronger defenses.** Try fine-pruning, Neural Cleanse, STRIP, and spectral-signature / activation-clustering.
- **Pruning-resistant (adaptive) backdoor** that spreads its representation across many channels.
- **Broader generalization.** Other architectures and audio representations (raw-waveform, MFCC), a larger real drone dataset, and a physical (over-the-air) drone-detection setup.
