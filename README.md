# Frame-Level-Speech-Classification-Using-MLP
Homework 1 Part 2 of course 11-785 Introduction to Deep Learning at CMU

# HW1P2: Frame-Level Speech Recognition

Course assignment (11-785, Spring 2026) for frame-level phoneme classification from MFCC features. The model takes a single MFCC frame plus surrounding context frames and predicts the phoneme spoken in that frame.

## Task

Each MFCC file is a speech recording stored as a sequence of 28-dimensional feature vectors, one per frame. Every frame has a corresponding phoneme label drawn from a fixed set of 42 phonemes (including `[SIL]`, `[SOS]`, `[EOS]`). The goal is to classify each frame into one of these phonemes. Performance is measured by accuracy on a held-out Kaggle test set.

## Data Pipeline

- **Source**: `train-clean-100` (training), `dev-clean` (validation), `test-clean` (test). Each partition has an `mfcc` folder; train and dev also have a `transcript` folder.
- **Loading**: All MFCC arrays are loaded and concatenated into a single `T x 28` tensor. Transcripts are concatenated to `(T,)` with `[SOS]`/`[EOS]` stripped from each utterance and phoneme strings mapped to integer indices.
- **Normalization**: Cepstral normalization per utterance (zero mean, unit variance along the time axis) before concatenation.
- **Context**: The frame stack is zero-padded by `context` frames on the top and bottom. `__getitem__` returns a window of `2*context + 1` frames centered on the target frame, giving an input of shape `(2*context+1, 28)`.
- **Augmentation**: `FrequencyMasking` and `TimeMasking` applied in the training `collate_fn` with ~70% probability. Validation and test data are not augmented.

## Model

A fully connected "diamond" MLP that flattens the context window into a single vector and classifies it.

- **Structure**: Input → 1024 → 2048 → 3072 → 2048 → 1024 → Output (42 phonemes).
- **Per layer**: Linear → BatchNorm1d → GELU → Dropout (p=0.25), except the final output layer.
- **Weight init**: Kaiming normal, biases zeroed.
- **Constraint**: Under the 20M trainable-parameter limit (asserted in code).

The diamond shape expands features into a higher-dimensional space (3072) before compressing them for classification.

## Configuration

| Setting | Value |
|---|---|
| Context | 40 frames (input size = 81 x 28 = 2268) |
| Activation | GELU |
| Dropout | 0.25 |
| Optimizer | AdamW (lr 0.001, weight_decay 0.005) |
| Scheduler | CosineAnnealingLR (T_max 40, eta_min 1e-6) |
| Loss | CrossEntropyLoss |
| Batch size | 2048 |
| Epochs | 40 |
| Weight init | Kaiming normal |
| Augmentation | FreqMask (param 4) + TimeMask (param 8) |
| Subset | 1.0 (full dataset) |

## Training

- Mixed-precision training with `torch.autocast` and `GradScaler` for speed on GPU.
- Per-epoch train and validation loss/accuracy logged to Weights & Biases.
- Best checkpoint (by validation accuracy) saved locally as `best_model.pth` and to the WandB cloud.
- The notebook can restore a prior best model from WandB run `zydjurty` instead of retraining.

## Inference and Submission

- `test()` runs the model in eval mode with gradients disabled, takes the argmax over logits, and converts integer indices back to phoneme strings.
- Predictions are written to `submission.csv` with `id,label` columns and submitted to the Kaggle competition via the Kaggle API.
- A `model_metadata_*.json` (parameter count and architecture string) is generated for Autolab code submission.

## Files Produced

- `best_model.pth` – best model weights by validation accuracy.
- `model_arch.txt` – architecture string, logged to WandB.
- `submission.csv` – Kaggle predictions.
- `model_metadata_*.json` – metadata for Autolab.
- `HW1P2_final_submission.zip` – final packaged submission (README, acknowledgement, WandB logs, Kaggle data, notebook).

## Notes

- WandB project: `sddabir-carnegie-mellon-university/hw1p2`.
- Restoring the saved model requires a valid WandB API key.
- No pretrained or external models/data are used, per assignment rules; all architecture is built from base PyTorch layers.
