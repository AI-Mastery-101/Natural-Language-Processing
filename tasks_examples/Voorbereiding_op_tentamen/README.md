What is text analysis?

Is the process of turning unstructured text into structured text, used for detecting topics, intent, sentiment....
so that it can be summarized or used for insightful decisions...


Why should you do text analysis?
scale, so you can automatically process thousands to millions of texts
actionability, because you turn qualitative text into measurable signals and features that plug directly into dashboards, models or automations


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

Why do we need vector representations of text in NLP?

What is BoW?
What is TF-IDF?

Difference??

___________________________________________________________

Text clustering = No label
is an unsupervised technique that groups documents based on similarity of their content.
The algorithm doesn’t know labels in advance; it discovers structure by placing texts
with similar vocabulary or topics into the same cluster.

Real-life application (clustering):
Grouping incoming customer feedback into themes (e.g., delivery issues, pricing, app bugs)
so teams can spot emerging problems without manually reading every message.



text classification = Has label
Text classification is a supervised task where we train a model on labeled examples
to assign predefined categories to new texts. Given input text, the model predicts
its label, such as topic, intent or sentim

Real-life application (classification):
Automatically routing support emails by topic to the right department
(billing, technical support, cancellations) to speed up response and resolution


________________________________________

cos similarity??
why is it used??

2 types of text summarization:
TF-IDF text summarization...
eg. Extractive summarization:
selects and stitches together the most important sentences or phrases from the original text without rewriting them.
It’s fast, faithful to the source wording, and works well for formal documents but can be choppy.



Hugging-face summarization model:
Abstractive summarization:
generates new sentences that paraphrase the key ideas, like how a human would summarize.
It can be more concise and coherent across sections, but it’s harder to control and may introduce inaccuracies if not well-guided or grounded.


___________________________________

  Word embedding??
  is a dense vector representation of a word learned from data so that words with similar meanings have nearby vectors..

  

  
  Doc2Vec = Where you convert sentences into vectors


___________________________________
