What is text analysis?

Text analysis is the process of turning unstructured text into structured,
usable information by detecting entities, topics, intent, sentiment, keywords, and relations,
so it can be queried, summarized or used in downstream decisions.


Why should you do text analysis?
You do text analysis mainly for four reasons:
· scale, so you can automatically process thousands to millions of texts
· speed, because algorithms deliver insights in near real time instead of slow manual reviews
· consistency, since the same criteria are applied uniformly without human variability
· actionability, because you turn qualitative text into measurable signals and features that plug directly into dashboards, models or automations

Name two situations where text analysis is useful.
· Customer service: A phone company gets thousands of emails daily. Text analysis spots “billing issue” vs. “network down” and sends each to the right team automatically, cutting wait times. 
· Reviews & social: A café chain scans Google reviews and tweets. The system flags many mentions of “slow drive-thru,” so they add a second barista at peak hours and ratings go up.

Explain why companies use text analysis in sentiment analysis.
· Measure customer experience: quickly see how happy or unhappy customers are and why. 
· Prioritize fixes: cluster feedback by themes (price, delivery, support) to guide action. 
· Protect brand & marketing: track shifts in brand perception and campaign impact early. 
· Reduce risk & churn: flag spikes in negative sentiment to intervene before customers leave. 
· Turn text into KPIs: convert reviews/posts into metrics for dashboards and automation. 
· Scale & speed: process millions of comments in near real time instead of manual reading.



________________________________________________________________________________

Explain what this does:

import spacy

nlp = spacy.load("en_core_web_lg")

doc = nlp("I’m passionate about exploring how Natural Language Processing can contribute to technological advancement and benefit humanity.")

for token in doc:
    print(f"{token.text} -> {token.pos_}")

The code loads spaCy’s large English model (en_core_web_lg),
runs the given sentence through the pipeline and iterates over the 
resulting tokens to print each token together with its coarse part-of-speech (POS) tag.

Under the hood, spaCy first tokenizes the text, then applies its POS tagger. 
The large model also includes pretrained word vectors, a dependency parser, morphology 
and fine-grained tags and a named-entity recognizer, although these components are not used by the print loop.

The output is a structured, token-by-token view of the sentence’s grammatical categories, useful for downstream analysis, 
quality checks, or feature extraction.


____________________________________________________________________________________

**Why do we need vector representations of text in NLP?**
We need vector representations because models work with numbers, not raw text.
By mapping words or documents to vectors, we can measure similarity, feed texts to machine learning algorithms,
and capture meaningful linguistic patterns in a form that supports computation, comparison, and learning.


**What is Bag of Words (BoW)?**
Bag of Words (BoW) is a simple representation that turns a document into a vector of token counts.
It ignores word order and grammar and only records how often each vocabulary term appears in the document.

**What is TF-IDF?**
TF-IDF (Term Frequency–Inverse Document Frequency) weights each term by
how frequent it is in the document (TF) and how rare it is across the corpus (IDF).
The result down-weights very common terms and up-weights informative, distinctive terms.


**What is the difference between BoW and TF-IDF?**
Both BoW and TF-IDF ignore word order, but they differ in weighting.
BoW uses raw counts (or binary presence), treating all terms equally, which can overemphasize common words.
TF-IDF rescales counts using corpus-level rarity, reducing the impact of ubiquitous terms and highlighting terms that better characterize a document.



___________________________________________________________

What is text clustering?
Text clustering is an unsupervised technique that groups documents based on similarity of their content.
The algorithm doesn’t know labels in advance; it discovers structure by placing texts with similar vocabulary or topics into the same cluster.


Give one real-life application of text clustering.
Grouping incoming customer feedback into themes (e.g., delivery issues, pricing, app bugs)
so teams can spot emerging problems without manually reading every message.


What is text classification?
Text classification is a supervised task where we train a model on labeled examples to assign predefined categories to new texts.
Given input text, the model predicts its label, such as topic, intent or sentiment


Give one real-life application of text classification.
Automatically routing support emails by topic to the right department (billing, technical support, cancellations) to speed up response and resolution.


________________________________________

What is cosine similarity and why is it used?
Cosine similarity measures how similar two text vectors are by computing the cosine of the angle between them.
<img width="219" height="60" alt="image" src="https://github.com/user-attachments/assets/a251d838-a1ff-4225-95a4-8dc342b75e39" />


It is used because it focuses on direction rather than magnitude, making it robust when documents differ in length or scale (e.g., TF-IDF or embeddings).
In high-dimensional sparse spaces, it provides a stable, length-invariant similarity score.

Name and explain two types of text summarization.
Two types of text summarization
· Extractive summarization:
selects and stitches together the most important sentences or phrases from the original text without rewriting them.
It’s fast, faithful to the source wording, and works well for formal documents but can be choppy.


· Abstractive summarization: generates new sentences that paraphrase the key ideas, like how a human would summarize.
It can be more concise and coherent across sections, but it’s harder to control and may introduce inaccuracies if not well-guided or grounded



___________________________________

What is a word embedding?
Word embedding is a dense vector representation of a word learned from data so that words with similar meanings have nearby vectors.
Unlike one-hot or Bag-of-Words, embeddings capture semantic and syntactic relationships (e.g., analogy structure) and enable models to generalize across similar words.
Common approaches include Word2Vec, GloVe and fastText (static embeddings), and modern contextual embeddings from transformers.


What is Doc2Vec?
Doc2Vec extends Word2Vec to learn a fixed-length vector for an entire document (sentence, paragraph or article).
It trains a document ID vector alongside word vectors to predict words in context, producing a representation that captures the document’s topics and style.
It has two main variants: PV-DM (Distributed Memory), which uses the document vector plus surrounding words to predict a target word and PV-DBOW (Distributed Bag of Words),
which uses the document vector to predict words sampled from the document.
New documents get vectors via inference using the trained model

___________________________________
