# Stack Overflow Question Tag Prediction (Multiclass–Multilabel)

A deep learning model that predicts the most relevant technology tags (e.g., `python`, `java`, `javascript`) for a Stack Overflow question, using both its title and body text.

## Problem Statement

Every question posted on Stack Overflow needs to be tagged so it reaches the right audience and is easy to search. Manually tagging is error-prone and inconsistent. This project automates tag suggestion by learning from historical question–tag pairs, framing it as a multi-label text classification problem (a question can have more than one correct tag).

## Demo

*(Add a screenshot/GIF here: paste a sample question title + body, show the top predicted tags with confidence scores.)*

```
Title: "How to reverse a string in Python?"
Body: "I want to reverse a string without using..."
Predicted tags: python (0.9X), string (0.7X)
```

## Dataset

- Source: Stack Overflow Questions + Tags dataset — *link the exact Kaggle/source dataset you used*
- Filtered down to the **top 10 most common tags** to keep the label space manageable and the dataset balanced
- Each question has: title, body, and one or more associated tags (multi-label target)

## Approach

1. **Data prep:** Loaded and merged questions with their tags, kept only the top 10 tags, and binarized the tag lists into multi-label target vectors.
2. **Text processing:** Tokenized title and body text separately and padded each to a fixed sequence length.
3. **Modeling:** Built a **dual-input GRU network** — separate embedding + GRU branches for title and body — concatenated before a sigmoid output layer (one unit per tag, since a question can have multiple tags).
4. **Training:** Compiled with binary cross-entropy (appropriate for multi-label problems) and trained on the tokenized sequences against the multi-label targets.
5. **Evaluation:** Assessed with F1-score and a per-tag classification report.

## Results

| Metric | Value |
|---|---|
| F1-score (samples avg) | *add your result* |
| Per-tag precision/recall | See classification report cell in the notebook |

*(Tip: call out which tags the model predicts well vs. poorly — e.g., "python" and "java" have distinctive vocabulary while overlapping tags like "c" and "c++" are harder to separate.)*

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## How to Run

```bash
git clone https://github.com/<your-username>/stackoverflow-tag-prediction.git
cd stackoverflow-tag-prediction
pip install -r requirements.txt
```

1. Download the dataset (link above) and place `Questions.csv` / `Tags.csv` in a `data/` folder.
2. Open `DL_P2_Multiclass_Multilabel_prediction_For_stack_overflow_Questions.ipynb` in Jupyter or Colab.
3. Run all cells in order.

## Project Structure

```
├── DL_P2_Multiclass_Multilabel_prediction_For_stack_overflow_Questions.ipynb
├── requirements.txt
├── README.md
└── data/                  # not included — see Dataset section
```

## Future Improvements

- Expand beyond the top 10 tags using hierarchical or embedding-based tag grouping
- Try a pretrained transformer (e.g., DistilBERT) instead of GRUs for better contextual understanding
- Add a threshold-tuning step per tag instead of a single global threshold, since tag frequencies vary widely
- Deploy a small web demo where a user pastes a question and gets live tag suggestions

## License

This project is licensed under the MIT License.
