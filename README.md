# VINE: Visual-INformed Entailment for Emoji-Based Textual Reasoning

## Overview

This project investigates the role of visual information in emoji-based textual entailment, as introduced in the [ELCo](https://aclanthology.org/2024.lrec-main.1381/) dataset.  
We propose **VINE (Visual-INformed Entailment)**, a multimodal model that integrates visual features extracted from a frozen Vision Transformer (ViT) encoder and textual features from BERT-based models.  
A learned projection layer maps visual embeddings into the textual space, allowing for joint reasoning over emoji semantics.

Our experiments demonstrate that incorporating visual features consistently improves entailment performance, especially for ambiguous or metaphorical emojis.

## Architecture

- **Textual Branch**:  
  Fine-tunes BERT-based models (e.g., BERT_base, RoBERTa_base, RoBERTa_large, BART_large) on emoji name and English sentence pairs.
  
- **Visual Branch**:  
  Extracts visual embeddings from emoji images using a frozen ViT encoder (pretrained within CLIP), applies mean pooling across emojis in a sequence, and projects into the BERT hidden space via a learned MLP.

- **Fusion**:  
  The visual and textual embeddings are concatenated and passed to an MLP classifier for entailment prediction.

## Repository Structure

```
scripts/
    vit.py                        
    finetune_original.py           
    finetune_with_descriptions.py  
    generate_description.py

exp-entailment/
    train.csv, val.csv, test.csv   # Dataset splits from ELCo

emojis/
    *.png                          # Crawled emoji images (PNG format)

data/
    ...                            # (Optional) Generated Descriptions in json format
```
## Usage

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Running Baseline Fine-tuning (Text-Only)
```
python scripts/finetune_original.py
```
### 3. Running VINE (Visual + Textual Fusion)

```
python scripts/vit.py
```
### 4. Running Ablation Study (Text with Generated Descriptions)
```
python scripts/finetune_with_descriptions.py
```

## Results
VINE outperforms text-only baselines across multiple backbone models (BERT_base, RoBERTa_base, RoBERTa_large, BART_large).

Improvements are especially significant for complex, visually distinctive, or metaphorical emojis.

Ablation studies highlight the limitations of VLM-generated descriptions compared to directly using visual embeddings.

