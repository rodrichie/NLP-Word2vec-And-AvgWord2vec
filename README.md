# NLP-Word2vec-And-AvgWord2vec

## Overview
This notebook explores various natural language processing (NLP) techniques, focusing on text preprocessing and implementing Word2Vec and AvgWord2Vec models. The notebook demonstrates feature extraction from textual data for machine learning tasks.

## Key Features
- **Data Preprocessing:**
  - Text cleaning
  - Tokenization
  - Stopword removal
  - Lemmatization
  - Bag of Words and TF-IDF models
- **Word Embedding:**
  - Training Word2Vec models from scratch
  - Applying pre-trained Word2Vec embeddings
  - Computing average Word2Vec features for sentences
- **Model Evaluation:**
  - Training Naive Bayes models on Bag of Words and TF-IDF features
  - Accuracy, precision, and recall calculations

## Requirements
- Python 3.x
- Libraries:
  - `gensim`
  - `nltk`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `tqdm`

## Workflow
1. **Data Import and Preprocessing:**
   - The dataset includes a "message" column with text data and a "label" column for classification.
   - Text data is cleaned and prepared for feature extraction.
2. **Feature Extraction:**
   - Bag of Words (BoW) and TF-IDF models are created for text vectorization.
   - Word2Vec embeddings are trained and utilized for word representations.
3. **Classification:**
   - Naive Bayes models are trained using BoW and TF-IDF features.
   - Average Word2Vec features are used to improve classification performance.
4. **Model Evaluation:**
   - Metrics such as accuracy, precision, and recall are computed for each model.

## Key Code Highlights
- **Word2Vec Implementation:**
  - Training a Word2Vec model from scratch:
    ```python
    from gensim.models import Word2Vec
    model = Word2Vec(corpus, vector_size=100, window=5, min_count=2, workers=4)
    ```
  - Pre-trained embeddings:
    ```python
    import gensim.downloader as api
    model = api.load('word2vec-google-news-300')
    ```
  - Computing average Word2Vec features for a sentence:
    ```python
    def avg_word2vec(doc):
        return np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
    ```

## Results
- The Naive Bayes classifier achieved competitive results using BoW and TF-IDF features.
- Word2Vec embeddings demonstrated the ability to capture semantic information, potentially improving model performance.

## How to Use
1. Clone or download the repository containing this notebook.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the notebook step by step to preprocess data, train models, and evaluate performance.

## License
This project is open-source and available for educational purposes.

## Credits
- Dataset: Provided within the project.
- Libraries: `gensim`, `nltk`, `scikit-learn`, `pandas`, `numpy`, `tqdm`.

