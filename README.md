# SSMoE-MVMS

Official implementation of **《Learning Unified Multi-View Representations for Rapid and Explainable Spectral Analysis》**, a unified and explainable multi-view mixture-of-experts framework for molecular spectral analysis.

## Overview

Real-time and explainable spectral analysis is crucial for accelerating the profiling of molecules in chemical laboratories. However, current methods, whether quantum chemical simulations or equivalent neural networks, are single-view methods and rely on computationally demanding geometric optimization (**~8min per molecule**), which hampers the efficiency and limits the performance. Here, we developed a unified and explainable framework, termed Shared-Specific-Mixture-of-Experts for Multi-View Molecular Spectroscopy (**SSMoE-MVMS**), that integrates multi-view molecular encodings, including 3D geometric structures, 2D topological graphs, and 1D SMILES sequences. By integrating multi-view representations, SSMoE-MVMS implicitly aligned learned representations with underlying geometric priors via a dynamic learning strategy, enabling robust inference even without explicit 3D structural inputs.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Alex07015/SSMoE-MVMS.git
cd SSMoE-MVMS
```

Create environment:

```bash
conda create -n ssmoe python=3.9
conda activate ssmoe
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The framework supports multiple spectral datasets, including:

* IR spectra
* Raman spectra

Example dataset structure:

```
data/
 ├── train.lmdb
 ├── valid.lmdb
 └── test.lmdb
```

---

## Training for spectra prediction


Run training with:

```bash
python /train_spectra_pred/train_spectra_pred.py \
    --conf /config/qm9s_spectra_pred.yaml \
    --dataset qm9s_spectra \
    --device cuda:0 \
    --gpu 0 \
    --dp False \
    --model SSMoE_MVMS_Spectra_Prediction
```

## Training for cross-modality retrieval


Run training with:

```bash
python /train_retrieval/train_retrieval.py \
    --conf /config/qm9s_retrieval.yaml \
    --dataset qm9s_spectra \
    --device cuda:0 \
    --gpu 0 \
    --dp False \
    --model SSMoE_MVMS_Retrieval
```

---

## License

This project is released under the MIT License.
