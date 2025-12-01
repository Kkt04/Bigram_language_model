# Bigram Language Model — Tiny Shakespeare Implementation

A minimal implementation of a character-level bigram language model trained on the Tiny Shakespeare dataset using PyTorch.

## 📋 Overview

This project demonstrates the fundamental concepts of language modeling by building a simple **bigram language model**. The model predicts the next character in a sequence based only on the current character, without any memory of previous context.

## 🏗️ Model Architecture

The model consists of a single `nn.Embedding` layer that:
- Takes character indices as input (vocabulary size = 65)
- Outputs logits for the next character prediction
- Uses cross-entropy loss for training
- Generates text auto-regressively

### Key Components:
- **Vocabulary**: 65 unique characters from the Shakespeare dataset
- **Embedding Layer**: Maps each character to a 65-dimensional vector
- **Training**: AdamW optimizer with learning rate = 1e-3
- **Batch Processing**: Context window of 8 characters, batch size of 32

## 📊 Dataset

The model is trained on the **Tiny Shakespeare dataset** containing approximately 1.1 million characters of Shakespearean text.

### Dataset Statistics:
- **Total characters**: 1,115,394
- **Training split**: 90% (1,003,854 characters)
- **Validation split**: 10% (111,540 characters)
- **Vocabulary size**: 65 unique characters

## 🚀 Implementation Details

### Data Processing:
1. **Character Encoding**: Each character is mapped to a unique integer ID
2. **Tokenization**: Text is converted to tensor format
3. **Batching**: Random sequences of 8 characters are sampled for training

### Model Code:
```python
class BiGramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        # ... loss calculation and training logic
