# Deception Detection in Diplomacy Messages

This project focuses on detecting deception in text communications within the strategic game Diplomacy. It leverages both pre-trained language models and custom transformers, incorporating linguistic features and in-game metadata to improve classification performance. The goal is to distinguish truthful and deceptive messages, addressing real-world applications in negotiation, politics, and online platforms.

## Authors

- Muthuraj Vairamuthu – IIIT Delhi – [GitHub](https://github.com/muthuraj-vairamuthu) – muthuraj22307@iiitd.ac.in  
- Syam Sai Santosh Bandi – IIIT Delhi – syam22528@iiitd.ac.in  
- Pratham Sibal – IIIT Delhi – pratham22374@iiitd.ac.in  

## Project Structure


deception-detection/
├── nlp-project-endsem.ipynb # RoBERTa + Metadata model
├── harbinger-transformer.ipynb # Transformer with harbinger tokens
├── data/
│ ├── train.tsv
│ ├── valid.tsv
│ └── test.tsv
├── utils/
│ ├── tokenizer.py
│ └── dataset.py
├── models/
│ ├── roberta_model.py
│ └── transformer_model.py
└── README.md




## Models

### 1. RoBERTa with Metadata Integration

This model enhances the `roberta-base` transformer with contextual metadata using a cross-attention mechanism. It incorporates features like game phase, sender power, message sentiment, hedging, and more.

- Fusion via cross-attention (12 heads)
- Metadata processed through an MLP
- Class imbalance handled using asymmetric focal loss and weighted sampling

**Results:**
- Macro F1 Score: 0.5900
- Lie Class F1 Score: 0.2631
- Accuracy: 85.08%

Notebook: `nlp-project-endsem.ipynb`

---

### 2. Transformer with Harbinger Tokens

A lightweight custom transformer trained from scratch. It emphasizes a curated set of "harbinger" tokens indicative of deception (e.g., "maybe", "perhaps").

- 2 Transformer layers, 4 attention heads
- Embedding dimension: 127 + binary harbinger mask
- Contextual modeling with previous 2 messages
- Class imbalance addressed using weighted loss

**Results:**
- Macro F1 Score: 0.5600
- Lie Class F1 Score: 0.2121
- Accuracy: 82.38%
- AUC: 0.6112

Notebook: `harbinger-transformer.ipynb`

---

## Dataset

Diplomacy dataset from Peskov et al. (2020), annotated for truthful and deceptive messages.

- Train: 13,132 messages
- Validation: 1,416 messages
- Test: 2,741 messages
- Approx. 5% of the data labeled as deceptive

Preprocessing:
- Unicode normalization
- Feature extraction: sentiment, hedging, power dynamics
- Tokenization with custom vocabulary for the Transformer model

---

## Running on Kaggle

To run the models on Kaggle:

1. Create a new Kaggle Notebook and upload this repository or individual `.ipynb` files.
2. Upload your dataset files (`train.tsv`, `valid.tsv`, `test.tsv`) to the Kaggle session.
3. Enable GPU support:
   - Navigate to "Settings" → "Accelerator" → Select "GPU"
4. Update any file paths in the notebooks to match the Kaggle environment (usually under `/kaggle/input/`).
5. Run all cells in the notebook.

---

## Dependencies

Install required libraries using:

```bash
pip install torch transformers scikit-learn pandas matplotlib seaborn textblob tqdm

```

Evaluation Metrics:
| Model                 | Accuracy | Macro F1 | Lie F1 | AUC    |
| --------------------- | -------- | -------- | ------ | ------ |
| RoBERTa + Metadata    | 85.08%   | 0.5900   | 0.2631 | —      |
| Harbinger Transformer | 82.38%   | 0.5600   | 0.2121 | 0.6112 |


Key Contributions

    Metadata-aware RoBERTa with cross-attention

    Custom transformer with harbinger-token guided attention

    Addressing extreme class imbalance using sampling and specialized loss functions

    Comprehensive EDA and correlation analysis

    Performance surpasses previous state-of-the-art benchmark (Context LSTM+Power)

References

    Denis Peskov et al., It Takes Two to Lie: One to Lie, and One to Listen, EMNLP 2020

    Yinhan Liu et al., RoBERTa: A Robustly Optimized BERT Pretraining Approach, 2019

    Lu et al., ViLBERT: Pretraining Vision-and-Language Representations, 2019

    DePaulo et al., Cues to Deception, Psychological Bulletin, 2003

License

This project is intended for academic and research purposes. Please cite the original Diplomacy dataset authors when using this work.
Contact

For questions, reach out to:

    muthuraj22307@iiitd.ac.in

    syam22528@iiitd.ac.in

    pratham22374@iiitd.ac.in



