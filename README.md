# DualWeaver: Feature Fusion Surrogates for Multivariate Forecasting with Univariate Time Series Foundation Models


This repository implements adapters for temporal foundation models **Sundial** and **TimerXL**, featuring:

🔹 **DualWeaver** (WeaverCNN & WeaverMLP)  
🔹 **AdaPTS**  
🔹 **Full Fine-tuning**  
🔹 **Linear Probing**  
🔹 **Zero-Shot**  

> Developed for anonymous KDD submission  
> *(Repository will be made public upon acceptance)*

---

## 🛠️ Environment Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Transformers Version Requirement
```bash
pip install transformers==4.40.1
```
### 3. Mandatory Patch for AdaPTS
Remove `torch.no_grad` decorator in:
`transformers/generation/utils.py` → `generate()` function
*(Required for AdaPTS gradient backpropagation)*

---

## ⚙️ Pre-Run Configuration
### 1. Download Foundation Models
Place model.safetensors in these locations:
```text
hf_ltm/
├── sundial-base-128m/
│   └── model.safetensors
└── timer-base-84m/
    └── model.safetensors
```
### 2. Configure Dataset Paths
Update `--data_path in` scripts under `scripts/` for datasets.

## 🚀 Running Experiments
### DualWeaver (CNN Variant)
```bash
bash scripts/weavercnn/TimerXL_ETTh1.sh
```
### DualWeaver (MLP Variant)
```bash
bash scripts/weavermlp/Sundial_ETTh1.sh
```
### AdaPTS Adaptation
```bash
bash scripts/adapts/Sundial_ETTh1.sh
```
### Full Fine-tuning
```bash
bash scripts/full_fine_tuning/TimerXL_ETTh1.sh
```
### Linear Probing
```bash
bash scripts/linear_probing/TimerXL_ETTh1.sh
```
### Zero-shot Evaluation
```bash
bash scripts/zero_shot/Sundial_ETTh1.sh
```

---

## 📊 Expected Outputs
```text
metrics/
├── TimerXL_WeaverCNN_ETTh1.json
├── Sundial_WeaverMLP_ETTh1.json
└── ...

logs/
├── TimerXL_Finetune_full_ETTh1_*.log
├── Sundial_ZeroShot_ETTh1_*.json
└── ...
```

This README reflects the repository state at time of KDD submission.