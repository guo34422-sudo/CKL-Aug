# CKLAug

This repository contains the code for **CKLAug**, a ConceptNet-guided framework for metaphor data augmentation. CKLAug aims to generate metaphorical samples that are both diverse and label-consistent.

## Overview

CKLAug contains two main stages:

1. **Training stage**  
   The source-domain concept is first identified from a metaphorical sentence. Related concepts are then retrieved from ConceptNet as semantic anchors. These anchors are used to guide LLM-based candidate generation. A Judge LLM filters low-quality or label-inconsistent samples. The filtered data are used for LoRA fine-tuning and DPO optimization.

2. **Augmentation stage**  
   The trained augmentor is applied to target metaphor datasets to generate high-quality augmented samples for downstream tasks.

## Pipeline

```text
Input sentence
   → Source-domain identification
   → ConceptNet retrieval
   → LLM generation
   → Judge LLM filtering
   → LoRA fine-tuning
   → DPO optimization
   → Dataset augmentation
