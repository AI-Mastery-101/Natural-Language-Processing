# Theorie & Toepassing van Word2Vec, Doc2Vec en Andere Word Embeddings

Dit document behandelt de volledige theorie en praktische toepassing van Word2Vec, Doc2Vec en verschillende andere embedding-methoden die belangrijk zijn binnen Natural Language Processing (NLP).
Het doel is een helder en uitgebreid overzicht te geven van wat embeddings zijn, waarom ze bestaan, hoe ze werken en hoe je ze toepast.

---

## 1. Wat Zijn Word Embeddings?

Word embeddings zijn representaties van woorden als numerieke vectoren.
In plaats van one-hot representaties (lange lijsten met nullen) gebruiken embeddings dense, compacte vectoren waarin betekenis is gecodeerd.

Voorbeeld van semantische relaties:

* king - man + woman ≈ queen
* paris - france + italy ≈ rome
* walk ≈ walking ≈ walked

Embeddings hebben drie belangrijke eigenschappen:

1. Contextueel – woorden krijgen vergelijkbare vectoren als ze in vergelijkbare context voorkomen.
2. Semantisch – betekenis komt tot uiting in afstanden tussen vectoren.
3. Algebraïsch – vectoren kunnen analoge relaties vastleggen.

Embeddings vormen de basis van klassieke NLP-modellen en hebben uiteindelijk geleid tot moderne transformer-gebaseerde modellen.

---

## 6.1 Word2Vec

### Theorie Word2Vec

Word2Vec is een methode om woordbetekenissen te leren door context te analyseren. Het gebruikt een neuraal netwerk met één verborgen laag, maar het doel is niet om voorspellingen te doen; het doel is om de gewichten van de verborgen laag te gebruiken als embeddings.

Word2Vec bevat twee hoofdmodellen:

#### CBOW (Continuous Bag-of-Words)

CBOW voorspelt het doelwoord op basis van de contextwoorden.

Voorbeeld:
Context: "de kat op de"
Doelwoord: "mat"

Kenmerken:

* snel
* effectief op grote datasets
* werkt goed voor frequente woorden

#### Skip-Gram

Skip-Gram voorspelt de contextwoorden op basis van het doelwoord.

Voorbeeld:
Doelwoord: "kat"
Contextwoorden: ["de", "zit", "op"]

Kenmerken:

* effectief bij zeldzame woorden
* krachtig op kleinere datasets
* wordt veel gebruikt in onderzoek

#### Hoe Word2Vec leert

Word2Vec leert door:

* woorden die samen voorkomen dichter bij elkaar te plaatsen
* woorden die niet samen voorkomen verder uit elkaar te plaatsen

Belangrijke technieken:

* Negative Sampling: het netwerk leert ook van niet-bijbehorende woorden
* Subsampling: veel voorkomende woorden worden soms verwijderd
* Lookup Tables: woordvectoren worden direct opgehaald uit een tabel

### Apply Word2Vec

Voorbeeld van het trainen van een Word2Vec-model:

```python
from gensim.models import Word2Vec

sentences = [
    ["ik", "hou", "van", "machine", "learning"],
    ["word2vec", "maakt", "betekenisvolle", "embeddings"]
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1   # Skip-Gram, sg=0 voor CBOW
)

model.wv.most_similar("learning")
```

---

## 6.2 Doc2Vec

### Theorie Doc2Vec

Doc2Vec breidt Word2Vec uit naar volledige documenten, paragrafen en zinnen.
In plaats van alleen woordvectoren te leren, worden ook documentvectoren gemaakt die de inhoud van complete teksten representeren.

Doc2Vec kent twee hoofdvarianten:

#### PV-DM (Distributed Memory)

Bij PV-DM wordt de document-ID gecombineerd met contextwoorden om het doelwoord te voorspellen.
Eigenschappen:

* bewaart contextinformatie
* geschikt voor langere documenten
* lijkt op CBOW maar met document-ID

#### PV-DBOW (Distributed Bag-of-Words)

Bij PV-DBOW wordt de documentvector gebruikt om woorden uit het document te voorspellen.
Eigenschappen:

* lijkt op Skip-Gram
* sneller
* weinig geheugen nodig

#### Waarom Doc2Vec gebruiken?

* documentvergelijking
* informatieophaling
* clustering
* semantisch zoeken
* embeddings voor volledige teksten

### Apply Doc2Vec

```python
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

documents = [
    TaggedDocument(["machine", "learning", "is", "krachtig"], [0]),
    TaggedDocument(["embeddings", "koppelen", "betekenis", "aan", "woorden"], [1])
]

model = Doc2Vec(
    documents,
    vector_size=100,
    window=5,
    min_count=1,
    dm=1  # 1 = PV-DM, 0 = PV-DBOW
)

vector = model.infer_vector(["dit", "is", "een", "nieuwe", "zin"])
```

---

## 6.3 Other Word Embeddings

In dit deel bespreken we verschillende embeddingmethoden buiten Word2Vec en Doc2Vec.
Elke methode heeft een eigen theoretische basis en toepassingsgebied.

---

### GloVe (Global Vectors)

#### Theorie GloVe

GloVe gebruikt wereldwijde co-occurrence statistieken.
Het model bouwt een grote matrix die weergeeft hoe vaak woorden samen voorkomen, en factoriseert deze naar vectoren.

Voordelen:

* sterke semantische relaties
* goed in analoge vergelijkingen
* gebruikt globale statistiek in plaats van alleen lokale context

### Toepassing GloVe

```python
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format(
    "glove.6B.100d.txt",
    no_header=True,
    binary=False
)
```

---

### FastText

#### Theorie FastText

FastText breekt woorden op in subwoorden, zogenaamde n-grams.
Hierdoor kan het model ook vectoren genereren voor onbekende woorden.

Voorbeeld:
"lopen" → "lo", "lop", "ope", "pen", "en"

Voordelen:

* werkt goed met onbekende woorden
* gevoelig voor morfologie
* robuust tegen typefouten

### Toepassing FastText

```python
from gensim.models import FastText

model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1
)
```

---

### WordRank

#### Theorie WordRank

WordRank gebruikt een ranking-loss in plaats van een voorspellingsdoel.
Het leert hoe woorden geordend moeten worden op basis van hun gelijkenis.

Voordelen:

* sterk bij kleine of ruisende datasets
* geschikt voor informatie-ophaal systemen
* legt nadruk op relatieve orde in plaats van absolute positie

Implementatie is doorgaans custom en niet inbegrepen in standaard libraries.

---

### VarEmbed

#### Theorie VarEmbed

VarEmbed is gebaseerd op metric learning.
Het richt zich op afstanden, variatie en structuur in woordruimtes.

Voordelen:

* behoudt linguïstische structuren
* werkt goed voor kleine en gevoelige datasets
* nuttig in onderzoek en gespecialiseerde NLP-taken

---

### Poincaré Embeddings

#### Theorie Poincaré

Poincaré embeddings representeren woorden in hyperbolische ruimte.
Deze ruimte is ideaal voor hiërarchische structuren zoals:

* dier → zoogdier → hond → bulldog
* organisatie → afdeling → team

Afstand in hyperbolische ruimte weerspiegelt diepte in een hiërarchie veel beter dan in Euclidische ruimte.

### Toepassing Poincaré

```python
from gensim.models.poincare import PoincareModel

model = PoincareModel(train_data="relations.txt", size=50)
model.train(epochs=50)
```

---

## Conclusie

Embeddings zijn een essentiële bouwsteen van moderne NLP.
De belangrijkste inzichten zijn:

* Word2Vec modelleert betekenis via contextvoorspelling, efficiënt en effectief.
* Doc2Vec embedt volledige documenten en is nuttig voor documentvergelijking en informatieophaling.
* GloVe gebruikt wereldwijde statistiek voor sterke semantische relaties.
* FastText gebruikt subwoorden en is robuust voor onbekende woorden.
* WordRank en VarEmbed bieden alternatieve optimalisatieprincipes.
* Poincaré embeddings modelleren hiërarchieën op een natuurlijke manier.

Samen vormen deze technieken een solide basis voor moderne NLP-architecturen en methoden.

