# Arabic Sentiment Analysis with Classic ML and AraBERT

This notebook demonstrates sentiment analysis for Arabic text using both traditional Machine Learning models (SVM and Linear Regression with TF-IDF features) and a fine-tuned Transformer-based model (AraBERT).

## Dataset

The notebook utilizes the following dataset:
https://huggingface.co/datasets/ImranzamanML/Arabic-Sentiments

The dataset is expected to be available as test.tsv and train.tsv files, containing Arabic text and corresponding sentiment labels (e.g., 'pos' for positive, 'neg' for negative).

## Project Structure

The notebook is divided into several parts:

*   **Part 0: Demonstration**: This section showcases how to load and use the trained models for inference on new text samples.
*   **Part 1: Dependencies and Imports**: Installs necessary libraries (`datasets`, `transformers`, `evaluate`, `arabert`, `scikit-learn`, `torch`) and imports them. Defines high-level constants for model training.
*   **Part 2: Load and Process Dataset**: Loads sentiment datasets (`test.tsv`, `train.tsv`), combines them, and normalizes Arabic text using `ArabertPreprocessor`.
*   **Part 3: Feature Extraction (TF-IDF)**: Generates TF-IDF features from the normalized text data for use with classic ML models.
*   **Part 4: Classic ML Models**: Trains and evaluates Support Vector Machine (LinearSVC) and Linear Regression models using the TF-IDF features. Saves the trained models and vectorizers.
*   **Part 5: AraBERT Transformer Based LLM / NLP Model**: Fine-tunes an AraBERT model for sentiment classification. This section includes preparing the data for the transformer, defining the model, optimizer, loss function, and training loop, and evaluating the model.

## Setup and Installation

To run this notebook, you will need to install the following libraries:

```bash
pip install datasets transformers evaluate arabert
