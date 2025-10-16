

# **Vectorizing Text, Transformations, and n-grams**

This module explains how textual data can be transformed into numerical representations that can be used by machine learning models.
It covers why we need vectors, how to create them, how to transform text, and how to enrich text features using n-grams and other preprocessing techniques.

---

## **3.1. Vectors and Why We Need Them**

### **Why Vectors Are Needed**

Machine learning algorithms and statistical models operate on numerical data.
Textual data, however, is made up of words, phrases and sentences — things that computers cannot directly interpret.
To use text in models, we must convert it into numerical form while preserving as much semantic and structural information as possible.
This process is known as text vectorization.

Vectorization enables us to:

* Represent text as structured numerical features
* Measure similarity or distance between documents
* Feed textual data into algorithms such as logistic regression, SVM, or neural networks

---

### **Bag-of-Words (BoW)**

The Bag-of-Words model represents each document as a vector of word counts or frequencies.
It treats every document as a “bag” containing words — ignoring grammar and word order but keeping track of how often each word appears.

**Example:**

| Document       | I | love | hate | pizza |
| -------------- | - | ---- | ---- | ----- |
| “I love pizza” | 1 | 1    | 0    | 1     |
| “I hate pizza” | 1 | 0    | 1    | 1     |

The result is a term-document matrix, where each column corresponds to a word in the vocabulary and each row corresponds to a document.

Advantages:

* Simple and effective for many tasks
* Works well with traditional ML models

Limitations:

* Ignores word order and context
* Produces large, sparse vectors
* Treats all words as equally important

---

### **TF-IDF (Term Frequency – Inverse Document Frequency)**

TF-IDF improves upon Bag-of-Words by giving more weight to informative words and less weight to frequent but uninformative ones.

[
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
]

Where:

* TF (Term Frequency): How often a term appears in a document.
* IDF (Inverse Document Frequency): How rare the term is across all documents.

[
\text{IDF}(t) = \log\frac{N}{n_t}
]

* N = total number of documents
* nₜ = number of documents containing term t

Interpretation:

* Words like “the” or “is” appear in almost all documents → low IDF
* Words like “anchovies” or “blockchain” appear in few documents → high IDF

Advantages:

* Emphasizes distinctive words
* Reduces the effect of common, uninformative terms

Limitations:

* Still ignores word order
* Cannot capture semantics (for example: “good” vs. “great”)

---

### **Other Common Text Representations**

| Representation                | Description                                                                          | Captures                            |
| ----------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------- |
| Word2Vec                      | Neural model that learns vector representations based on context.                    | Semantic similarity between words   |
| GloVe                         | Global Vectors trained using word co-occurrence statistics.                          | Global semantic relationships       |
| Doc2Vec                       | Extends Word2Vec to represent entire documents.                                      | Document-level meaning              |
| FastText                      | Builds word vectors using subword information (handles morphology and misspellings). | Subword and morphological structure |
| BERT / Transformers           | Contextual embeddings where meaning depends on sentence context.                     | Deep contextual semantics           |
| Sentence Transformers (SBERT) | Embeddings for full sentences and paragraphs.                                        | Sentence-level semantic similarity  |

---

## **3.2. Vector Transformation**

After defining a representation (for example, BoW or TF-IDF), we apply a vector transformation to convert text into numerical matrices.

### **General Steps**

1. Tokenization: Split text into words or tokens.
2. Vocabulary Construction: Create a list of all unique tokens across the corpus.
3. Vectorization: Count or weight each token according to BoW or TF-IDF rules.
4. Transformation: Convert the collection of documents into a feature matrix (shape: n_documents × n_features).

### **Example Workflow**

```
Text → Tokens → Vocabulary → Numerical Vectors → Model Input
```

Example:

```
“I love pizza” → [I, love, pizza]
Vocabulary: [I, love, hate, pizza]
→ Vector: [1, 1, 0, 1]
```

In practice, libraries such as scikit-learn provide tools like:

* CountVectorizer → Bag-of-Words
* TfidfVectorizer → TF-IDF

These automatically handle tokenization, vocabulary building, and transformation.

---

## **3.3. n-grams and Other Preprocessing**

### **n-grams**

An n-gram is a contiguous sequence of n items (typically words) from a piece of text.
They capture local context and limited word order information that single words (unigrams) miss.

| n           | Example (for “New York pizza is great”)       |
| ----------- | --------------------------------------------- |
| 1 (unigram) | New, York, pizza, is, great                   |
| 2 (bigram)  | New York, York pizza, pizza is, is great      |
| 3 (trigram) | New York pizza, York pizza is, pizza is great |

Benefits:

* Captures short phrases and context (for example: “New York” vs. “York”)
* Improves classification accuracy for many tasks

Trade-off:

* Increases dimensionality and sparsity
* Higher n = more features, more computational cost

---

### **Other Text Preprocessing Techniques**

| Step                              | Description                                            | Example / Tool            |
| --------------------------------- | ------------------------------------------------------ | ------------------------- |
| Lowercasing                       | Converts all text to lowercase for consistency.        | “Pizza” → “pizza”         |
| Tokenization                      | Splits sentences into individual words or subwords.    | nltk.word_tokenize, spaCy |
| Stopword Removal                  | Removes very common words with little meaning.         | “the”, “and”, “is”        |
| Stemming                          | Reduces words to their root form (rule-based).         | “running” → “run”         |
| Lemmatization                     | Reduces words to base form using linguistic knowledge. | “better” → “good”         |
| Removing Punctuation / Numbers    | Cleans symbols that add no value.                      | regex (re.sub())          |
| Handling Emojis or Special Tokens | Keep or remove depending on task.                      | custom preprocessing      |

---

## **3.4. Full Workflow Summary**

| Step                   | Task                                                                   | Output                  |
| ---------------------- | ---------------------------------------------------------------------- | ----------------------- |
| 1️⃣ Preprocessing      | Clean, normalize, tokenize text                                        | List of tokens          |
| 2️⃣ Vectorization      | Convert tokens to numerical representation (BoW / TF-IDF / Embeddings) | Feature matrix          |
| 3️⃣ Transformation     | Apply weighting and scaling                                            | Model-ready data        |
| 4️⃣ Feature Enrichment | Add n-grams or domain features                                         | Improved context        |
| 5️⃣ Modeling           | Train ML model (for example, Logistic Regression, SVM, Neural Net)     | Predictions or insights |

---

## **Key Takeaways**

* Text must be vectorized before machine learning models can process it.
* Bag-of-Words and TF-IDF are the most common classical approaches.
* Vector transformation creates structured numerical data for modeling.
* n-grams capture short-range context and improve model expressiveness.
* Preprocessing (cleaning, tokenization, stopword removal, etc.) is crucial for high-quality results.
* Modern NLP often uses embeddings (for example, Word2Vec, BERT) for deeper semantic understanding.

