# Uitgebreide Theorie, Voorbeelden en Best Practices

Word2Vec is een van de meest invloedrijke algoritmes binnen Natural Language Processing (NLP) voor het leren van betekenisvolle vectorrepresentaties van woorden.
Dit document geeft een diepgaande uitleg over:

* de theoretische basis van Word2Vec
* de twee architecturen (CBOW, Skip-Gram)
* hoe het model leert
* belangrijke technieken (Negative Sampling, Subsampling)
* voorbeelden in Python (Gensim)
* best practices voor training
* veelgemaakte fouten en hoe je ze voorkomt

---

# 1. Wat is Word2Vec?

Word2Vec is een methode om woorden om te zetten in compacte, dense vectoren (word embeddings) waarin semantische relaties tussen woorden zijn gecodeerd.

In plaats van woorden als unieke symbolen te behandelen, leert Word2Vec:

* welke woorden vaak samen voorkomen
* welke woorden in vergelijkbare context staan
* hoe woorden betekenis delen op basis van gebruik

De vectoren die Word2Vec produceert hebben krachtige eigenschappen:

* woorden met vergelijkbare betekenis liggen dicht bij elkaar
* analoge relaties kunnen lineair worden gerepresenteerd

  * koning - man + vrouw ≈ koningin
  * Parijs - Frankrijk + Italië ≈ Rome
* syntactische en semantische structuur zit impliciet in de ruimte

---

# 2. Waarom Word2Vec?

Before Word2Vec, NLP gebruikte:

* One-hot vectors
* Bag-of-Words
* TF-IDF

Deze technieken hebben grote beperkingen:

* vectors bevatten geen betekenis
* geen relatie tussen woorden
* dimensionaliteit gelijk aan vocabulairegrootte
* gevoelig voor synoniemen en variaties

Word2Vec lost deze problemen op met compacte vectoren (typisch 50–300 dimensies) die betekenis modelleren.

---

# 3. De Architectuur van Word2Vec

Word2Vec bestaat uit twee modellen:

1. CBOW (Continuous Bag-of-Words)
2. Skip-Gram

Beide zijn eenvoudige neurale netwerken met:

* één verborgen laag
* een embedding-matrix als enige relevante parameter
* softmax of negative sampling voor optimalisatie

---

## 3.1 CBOW (Continuous Bag-of-Words)

CBOW probeert het doelwoord te voorspellen op basis van zijn context.

Voorbeeldzin:
"de hond zit op de mat"

Context (window size = 2):
["de", "zit", "op", "de"]

Doelwoord:
"hond"

Eigenschappen van CBOW:

* snel
* stabiel
* goed voor grote corpora
* werkt beter voor frequente woorden

Intern werkt CBOW als volgt:

1. contextwoorden worden omgezet in vectoren
2. deze vectors worden gemiddeld
3. het model voorspelt het doelwoord

---

## 3.2 Skip-Gram

Skip-Gram doet het omgekeerde van CBOW:
het probeert de context te voorspellen op basis van een doelwoord.

Voorbeeld:

Doelwoord: "hond"
Contextwoorden: ["de", "zit", "op", "de"]

Eigenschappen:

* werkt beter voor zeldzame woorden
* langzamer dan CBOW
* gevoelig voor semantische nuances
* vaak de beste keuze voor kleinere datasets

Skip-Gram produceert vaak hoge kwaliteit embeddings, zeker met **negative sampling**.

---

# 4. Hoe Word2Vec Leert

Hoewel Word2Vec wordt beschreven als een neuraal netwerk, draait alles om het aanleren van de **embedding vectoren**, niet om voorspellen.

Het model probeert:

* woorden die samen voorkomen dichtbij elkaar te plaatsen
* woorden die niet samen voorkomen uit elkaar te duwen

Er zijn twee belangrijke optimalisatiecomponenten:

---

## 4.1 Negative Sampling

Het volledige softmax-proces is te traag voor grote vocabulaires.
Daarom gebruikt Word2Vec negative sampling:

1. Pak het juiste contextwoord (positief voorbeeld)
2. Kies enkele willekeurige woorden uit de vocabulaire (negatieve voorbeelden)
3. Optimaliseer zodat:

   * het juiste woord naar de context wordt getrokken
   * de negatieve woorden worden afgestoten

Negative Sampling zorgt voor:

* hoge snelheid
* zeer goede prestaties
* minder geheugenverbruik

---

## 4.2 Subsampling van Veelvoorkomende Woorden

Woorden zoals:

* de
* een
* het
* en
* van

komen extreem vaak voor.
Ze hebben weinig informatie en vertragen training.

Word2Vec verwijdert deze woorden soms willekeurig.
Dit heet subsampling en verbetert:

* trainingssnelheid
* kwaliteit van embeddings
* minder bias naar functie-woorden

---

# 5. De Vectorruimte

De vectorruimte van Word2Vec is bijzonder omdat:

* afstand ≈ semantische gelijkenis
* richtingen ≈ betekenisrelaties
* clusters ≈ woordcategorieën

Voorbeeldclusters:

* dieren: hond, kat, paard, koe
* landen: Frankrijk, Italië, Duitsland
* werkwoorden: lopen, liep, lopen, gelopen

De ruimte blijkt *lineair* te zijn voor veel taalkundige fenomenen.

---

# 6. Word2Vec in de Praktijk (Python Voorbeelden)

Voorbeelddataset:

```python
sentences = [
    ["ik", "hou", "van", "machine", "learning"],
    ["word2vec", "maakt", "betekenisvolle", "embeddings"],
    ["dit", "is", "een", "voorbeeld", "zin"]
]
```

### Training:

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1     # Skip-Gram
)
```

### Vector van een woord:

```python
model.wv["machine"]
```

### Meest vergelijkbare woorden:

```python
model.wv.most_similar("learning")
```

### Cosine similarity:

```python
model.wv.similarity("machine", "learning")
```

### Analogie:

```python
model.wv.most_similar(positive=["koning", "vrouw"], negative=["man"])
```

---

# 7. Best Practices voor Word2Vec

Hier volgen de belangrijkste richtlijnen voor het trainen van kwalitatief sterke embeddings.

---

## 7.1 Kies Skip-Gram voor kleine of complexe datasets

Aanbevolen in situaties waar:

* data beperkt is
* semantische nuance belangrijk is
* zeldzame woorden cruciaal zijn

CBOW is geschikt voor zeer grote corpora.

---

## 7.2 Gebruik grote window size voor semantiek

Window size bepaalt hoeveel woorden links en rechts deel uitmaken van de context.

* window = 2: syntaxis
* window = 5: algemeen taalgebruik
* window = 10+: diepe semantiek

Voor betekenisvolle embeddings is window = 5 of 10 meestal goed.

---

## 7.3 Min_count verstandig kiezen

Min_count bepaalt welke woorden worden genegeerd.

* min_count = 1: alle woorden, maar veel ruis
* min_count = 5: vaak een goed minimum
* min_count = 10+: voor hele grote datasets

---

## 7.4 Gebruik voldoende dimensies

* 50 dimensies = kleine experimenten
* 100–200 = typische NLP-taken
* 300+ = diepe semantiek

Meer dimensies → meer nuance, maar ook meer kans op overfitting bij kleine datasets.

---

## 7.5 Train lang genoeg

Word2Vec convergeert langzaam.
Gebruik:

```python
epochs = 10–50
```

Meer epochs → betere embeddings.

---

## 7.6 Gebruik subsampling

Het subsampling-mechanisme voorkomt dat veelvoorkomende woorden de training domineren.

Aanbevolen:

```python
sample = 1e-5
```

---

## 7.7 Gebruik negative sampling

Aanbevolen waardes:

* negative = 5–20
* typ. negative = 5 voor grote datasets
* negative = 10–20 voor kleine datasets

---

## 7.8 Visualiseer embeddings

Gebruik PCA of t-SNE om:

* clusters te zien
* foutieve vectoren te detecteren
* kwaliteit te beoordelen

---

# 8. Veelgemaakte Fouten

1. Te kleine dataset
2. Te lage vectorgrootte
3. Geen subsampling gebruiken
4. Min_count te laag zetten → ruis
5. Onvoldoende epochs
6. Word2Vec gebruiken voor documentvergelijking (→ gebruik Doc2Vec)

---

# 9. Wanneer Word2Vec Niet Geschikt Is

Word2Vec faalt in situaties waar:

* woorden meerdere betekenissen hebben (bank: geld vs stoel)
* context afhankelijk is van woordvolgorde
* documenten vergeleken moeten worden (Doc2Vec is beter)
* context van meerdere zinnen belangrijk is

Moderne alternatieven zoals BERT lossen dit op.

---

# 10. Conclusie

Word2Vec is een krachtig en efficiënt algoritme dat woordbetekenissen vastlegt op basis van context. Het heeft de basis gelegd voor moderne NLP en blijft waardevol in toepassingen zoals:

* semantische analyse
* clustering
* analogieën
* zoekmachines
* recommender systems
* data preprocessing voor machine learning modellen

Met de juiste instellingen, voldoende data en goede best practices levert Word2Vec robuuste en interpreteerbare embeddings.

