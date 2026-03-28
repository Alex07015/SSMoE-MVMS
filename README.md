# SSMoE-MVMS

Official implementation of **《Learning Unified Multi-View Representations for Rapid and Explainable Spectral Analysis》**, a unified and explainable multi-view mixture-of-experts framework for molecular spectral analysis.

## Overview

Real-time and explainable spectral analysis is crucial for accelerating the profiling of molecules in chemical laboratories. However, current methods, whether quantum chemical simulations or equivalent neural networks, are single-view methods and rely on computationally demanding geometric optimization (**~8min per molecule**), which hampers the efficiency and limits the performance. Here, we developed a unified and explainable framework, termed Shared-Specific-Mixture-of-Experts for Multi-View Molecular Spectroscopy (**SSMoE-MVMS**), that integrates multi-view molecular encodings, including 3D geometric structures, 2D topological graphs, and 1D SMILES sequences. By integrating multi-view representations, SSMoE-MVMS predicted spectra in just **0.1s per molecule**, over three orders of magnitude faster than 3D geometry-dependent models and demonstrated **robust performance without explicit 3D coordinates during inference**.

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
 ├──qm9s_results/
    ├── train.lmdb
    ├── valid.lmdb
    └── test.lmdb
```
You can download the dataset from [here](https://drive.google.com/drive/folders/1KVBGMwWvPkOKDDpReuzuzA-FRLFxsEV-?usp=drive_link).
---

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
# Notice

* (1) You should modify the paths in your configuration and set the correct logging path.
* (2) You should download the pretrained molecular encoder and checkpoints from [here](https://drive.google.com/drive/folders/1j93RzD5Vg3N_NiTMP78r-2BBsqFzC7jS?usp=drive_link) and put the pretrained encoder into the `/model` directory.

### Example model structure
```
model/
 ├── SMILES
 ├── unimol
 ├── graphmvp
 ├──...
```

## Evaluation for spectra prediction
More details can be found in the /spectra_pred_evaluate.ipynb.

## Evaluation for cross-modality retrieval
More details can be found in the /retrieval_evaluate.ipynb.
