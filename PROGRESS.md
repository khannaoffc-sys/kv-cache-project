# ProofGuard – Progress Log

---

## Project Summary

ProofGuard is a fraud-detection prototype for the "Cybersecurity & Fraud Detection" hackathon track.
It addresses a real problem in Indian commerce: merchants sometimes accept a UPI payment *screenshot* as proof of payment, but AI tools can now silently edit the amount, UTR number, timestamp, or status on such screenshots.
ProofGuard's core idea is "pixel-free" detection — instead of trusting pixels, it checks whether the *fields* on the screenshot (amount, UTR, timestamp, payee) are internally consistent and match the merchant's own order record.
It also runs standard image-forensics checks (ELA, metadata) and shows honestly when those fail, so judges can see why pixel detectors are not enough.
All images are synthetic (fictitious app/bank names, fake VPAs); no real payment data or real app logos are used.

---

## Current Status

**Phase:** M0 – All dependencies installed; revised smoke_test.py ready; about to run.
**Blocker:** None.
**Next step:** Run `research/smoke_test.py` and await output for go/no-go decision.

---

## Milestone Checklist

- [ ] **M0** (0-2h): Repo skeleton, smoke test, kill-rule decision
- [ ] **M1**: HTML templates -> PNG, dataset generator
- [ ] **M2**: ELA baseline, metadata checks, eval harness
- [ ] **M3**: OCR + consistency checks + calibration
- [ ] **M4**: Streamlit dashboard (3 tabs)
- [ ] **M5**: Polish, demo script, README, tests, 3-min demo

---

## Log

### 2026-10-10 — Session 1 start

**What I did:** Created `progress.md` as the very first file in the repository (required by the spec).
Confirmed Python 3.11.9 is available via `py -3.11` at
`C:\Users\dwara\AppData\Local\Programs\Python\Python311\python.exe`.
Repository root: `C:\Users\dwara\proofguard`. A `.venv` directory already exists from a
previous attempt; we will recreate it cleanly.

**Why:** The spec requires `progress.md` to be created before any other file.

**Files created/changed:** `progress.md` (this file).

**Next:** Create the virtual environment, install PyTorch (CUDA 12.8 wheel for RTX 50-series),
and remaining deps. Then write and run `research/smoke_test.py`.

---

## Decisions

| # | Decision | Reason |
|---|----------|--------|
| 1 | Python 3.11 via `py -3.11` launcher | Only Python 3.11 is installed; needed for PyTorch CUDA 12.8 wheel |
| 2 | PyTorch from `--index-url https://download.pytorch.org/whl/cu128` | RTX 50-series (Blackwell) requires CUDA 12.8+ wheels |
| 3 | Inpainting model: **runwayml/stable-diffusion-inpainting** (fp16, ~3.5 GB VRAM) | Fits in 8 GB VRAM; well-tested; Apache-2 licence; CPU offload available |
| 4 | OCR: **EasyOCR** as primary, pytesseract as fallback | EasyOCR installs cleanly on Windows with pip |
| 5 | Dashboard: **Streamlit** | Quickest interactive UI; no frontend expertise needed |
| 6 | "Pixel-free" core thesis | ELA/pixel detectors degrade on AI-inpainted edits; field-level checks are more robust |
| 7 | Synthetic-only dataset | Ethics: no real UPI screenshots; fictitious app "PayNow Demo", fake VPAs |

---

## Problems and Fixes

*(None yet — will be recorded here as they arise.)*

---

## Results So Far

*(Smoke test not yet run — results will be recorded here.)*

---

## How To Run

### One-command setup

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

### After setup

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# M0 — smoke test
py -3.11 research/smoke_test.py

# M1 — render templates + build dataset
py -3.11 research/render_templates.py
py -3.11 research/build_dataset.py

# M2 — run evaluation
py -3.11 eval/run_eval.py

# M4 — launch dashboard
streamlit run app/streamlit_app.py
```

---

## Glossary

| Term | Plain-English Meaning |
|------|-----------------------|
| **ELA** (Error Level Analysis) | A forensics technique that re-compresses a JPEG and looks at where the *difference* is large. Edited regions often show higher error levels. |
| **OCR** (Optical Character Recognition) | Software that reads text from an image (e.g. recognises "500" in a screenshot). |
| **AUC** (Area Under the ROC Curve) | A number 0-1 measuring how well a detector separates genuine from tampered images. 1.0 = perfect; 0.5 = random. |
| **Calibration** | Adjusting confidence scores so "80% confident" means right 80% of the time. |
| **Abstain** | The system says "I cannot decide — please verify another way" instead of guessing. |
| **Inpainting** | AI technique that fills in a selected region of an image, making edits look seamless. |
| **UPI** | Unified Payments Interface — India's real-time payment system. |
| **UTR** (Unique Transaction Reference) | A 12-digit number assigned to every UPI transaction. |
| **VPA** (Virtual Payment Address) | UPI address, e.g. `alice@demobank`. |
| **fp16** | Half-precision floating point — uses half the memory of fp32; allows larger models to fit in limited VRAM. |
| **VRAM** | Video RAM on the GPU. RTX 5060 has 8 GB. |
| **Logistic Regression** | Simple, explainable ML model that outputs a probability (0-1). |
| **Isotonic Calibration** | Adjusts probability outputs using a step-function fit. |
| **ECE** (Expected Calibration Error) | Measures how far a model's confidence is from actual accuracy. Lower = better. |
| **Brier Score** | Mean squared error between predicted probabilities and true labels. Lower = better. |
| **ROC** (Receiver Operating Characteristic) | Curve showing trade-off between catching fraud vs. falsely flagging genuine payments. |
| **OOD** (Out-of-Distribution) | Testing on data that looks different from training data. |
| **Synthetic data** | Data generated artificially by code/AI, not collected from the real world. |
| **Sidecar JSON** | Companion metadata file saved alongside each image, recording its origin and edit type. |
| **EXIF** | Metadata embedded in image files (camera, software, timestamps). |
| **Playwright** | Tool that drives a web browser programmatically — used to render HTML templates as PNG. |
| **CUDA** | NVIDIA's parallel computing platform, required to run AI models on GPU. |
| **Diffusers** | Hugging Face library for running image-generation/inpainting AI models in Python. |
