# DualWeaver: Synergistic Feature Fusion Surrogates for Multivariate Forecasting with Univariate Time Series Foundation Models

This repository contains the implementation for our ICML submission.

## 1. Environment Setup

To set up the environment, please install the required dependencies using `pip`. We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

## 2. Apply Custom Patch to Transformers

To ensure correct backpropagation for the AdaPTS, a patched version of `transformers.generation.utils.py` is required.

1. Locate your installed `transformers` package directory:
    ```bash
    pip show transformers
    ```
    (Look for the `Location` path in the output, e.g., `/path/to/site-packages`)

2. Overwrite the original `generation/utils.py` with the provided patch:
    ```bash
    # Example:
    cp patch/transformers.generation.utils.py /path/to/site-packages/transformers/generation/utils.py
    ```

## 3. Model Preparation

Please ensure the TSFM checkpoints are placed correctly. 
Download the pre-trained `timer-base-84m` (or other supported models) from [Hugging Face](https://huggingface.co/) and place the `model.safetensors` and other config files into the `hf_ltm` directory.

Structure should look like:
```text
hf_ltm/
└── timer-base-84m/
    ├── config.json
    ├── model.safetensors
    └── ...
```

## 4. Run Example

We provide a script to run a demonstration experiment on the ETTh1 dataset using TimerXL and WeaverMLP.

```bash
bash scripts/example.sh
```
