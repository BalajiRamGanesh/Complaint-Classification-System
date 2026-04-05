# Complaint Classification System
Automatically classifies customer complaints into the correct department using NLP and deep learning.

## Problem Statement
Manually reviewing and sorting customer complaints is inefficient. This project automates that process using a LSTM based text classifier.

## Data Source
Source: [Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/) Or [Kaggle](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)

## Dataset
- 162,421 consumer financial complaints (CFPB)
- 5 categories: credit card, credit reporting, debt collection, mortgages and loans, retail banking
- Class imbalance handled via undersampling

## Tech Stack
- Python
- PyTorch
- NLTK
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Model Architecture
- Embedding layer (vocab size: 5000, dim: 128)
- Bidirectional LSTM (hidden dim: 128)
- Max sequence length: 170

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 83% |
| Macro F1 | 0.81 |
| Weighted F1 | 0.84 |

| Class | F1 |
|---|---|
| credit_reporting | 0.88 |
| retail_banking | 0.85 |
| mortgages_and_loans | 0.81 |
| debt_collection | 0.75 |
| credit_card | 0.75 |

## Key Observations
- credit reporting scored highest due to distinct complaint language
- credit card and debt collection show overlap, leading to misclassification


## How to Run
```bash
pip install torch nltk scikit-learn pandas numpy matplotlib seaborn
```
Run `complaint_classifier.ipynb` top to bottom.

