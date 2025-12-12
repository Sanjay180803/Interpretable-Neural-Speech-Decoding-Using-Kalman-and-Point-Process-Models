# Interpretable-Neural-Speech-Decoding-Using-Kalman-and-Point-Process-Models

This repository implements a full neural speech decoding system on the Willett et al. (Nature 2021/2023) intracortical dataset, integrating Point-Process GLMs, Linear Kalman Filters, Unscented Kalman Filters, and LSTM/CTC text decoding, along with Whisper-based validation.

The project demonstrates how classical interpretable neural encoding models can be combined with modern sequence architectures to decode speech from high-dimensional motor cortex activity.

# Folder Structure

                  
├── models/
│   ├── pp_glm.py
│   ├── kf.py
│   ├── ukf.py
│   └── lstm_decoder.py
│
├── utils/
│   ├── preprocessing.py
│   ├── metrics.py
│   └── electrode_importance.py
│
├── evaluation/
│   ├── whisper_eval.py
│   ├── topk_ablation.py
│   └── ks_improved.py
│
└── main.py


# Results
| Metric                 | **PP–GLM** | UKF | KF |
|------------------------|-----------:|----:|---:|
| **Correlation**        | **0.45**   | 0.39 | 0.33 |
| **NRMSE**              | 0.22       | 0.28 | 0.32 |
| **CER (%)**            | **21**     | 27   | 35 |
| **WER (%) (2-stage)**  | **38**     | 45   | 60 |
| **WER (%) (3-stage)**  | 78         | 85   | 92 |


