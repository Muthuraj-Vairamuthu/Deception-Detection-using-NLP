# RoBERTa-Based Deception Detection in Diplomacy

This project implements a RoBERTa-based binary classifier to detect deceptive messages in the Diplomacy game dataset. It leverages both textual inputs and game-specific metadata, fused using a cross-attention mechanism.

## Project Structure

- **Roberta Deception Model.ipynb**: Main notebook containing preprocessing, model architecture, training loop, and evaluation.
- **Input Dataset**: The notebook expects Diplomacy message data with labels and associated metadata features.

## Key Features

- Pre-trained `roberta-base` model from HuggingFace Transformers.
- Metadata encoder with a 2-layer MLP.
- Cross-attention between RoBERTa token embeddings and encoded metadata.
- Asymmetric focal loss to handle class imbalance.
- Evaluation metrics: Accuracy, Macro F1, Lie F1, AUC.


## Output

- Training and validation loss/f1 curves.
- Test set metrics (Accuracy, F1, Macro F1, Lie F1, Truth F1).
- Model can be saved to `.pth` format if needed.

---


Tranformer harbinger based model:

    A PyTorch-based model for detecting deception in diplomatic messages using a transformer architecture with harbinger word detection.
    Overview
    This project implements a classifier that identifies deceptive messages in the context of the Diplomacy board game. The model uses a transformer-based architecture combined with special attention to "harbinger words" - linguistic markers that might signal deception.
    Features

    Transformer-based sequence classification
    Custom dataset handling for diplomatic messages
    Harbinger word detection as supplementary features
    Positional encoding for sequence awareness
    Robust error handling and data validation
    Early stopping and best model saving
    Comprehensive metrics evaluation (Accuracy, F1, AUC)
    Optimal threshold finding via precision-recall curves
    Training and validation loss visualization




## How to Run (Kaggle)

1. Upload the resp notebook to a new Kaggle Notebook.
2. Add your cleaned Diplomacy dataset as an input under "Add data".
3. Ensure GPU is enabled in Notebook Settings.
4. Run all cells sequentially to train and evaluate the model. 