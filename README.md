# Assignment 1 – Perceptron för OCR

En implementation av neurala nätverk från grunden, byggd steg för steg från en enkel neuron till ett fullt tränat CNN med transfer learning.

Allt är implementerat i **Google Colab** och versionshanterat med **GitHub**.

---

## Innehåll

```
assignment1/
  del1.ipynb   → Bygg en neuron (A1, A2, B, C, D)
  del2.ipynb   → Förbättra nätverket (CNN, augmentation, regularization m.m.)
  del3.ipynb   → Eget dataset + Transfer learning (hund vs katt)
```

---

## Del 1 – Bygg en neuron

Grundläggande implementation av en artificiell neuron, byggd utan färdiga AI-bibliotek.

### A1 – Neuron utan NumPy
En enkel neuron implementerad med vanlig Python och en for-loop.
```
data × vikter + bias = svar
```

### A2 – Neuron med NumPy
Samma neuron men med NumPy:s `np.dot()` för matrismultiplikation – snabbare och mer effektiv.

### B – Helt lager med NumPy
Istället för en enda neuron beräknas ett helt lager av neuroner i en enda matrisoperation.

### C – PyTorch + träning på MNIST
Nätverket byggs om med PyTorch och tränas på MNIST-datasetet – 60 000 handskrivna siffror.

**Resultat:**
```
Noggrannhet: 97.05%
```

### D – GPU (CUDA)
Samma träning som C men kördes på GPU via Google Colabs T4-grafikkort för snabbare beräkning.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Del 2 – Förbättra nätverket

Bygger vidare på del 1 med moderna AI-tekniker för att förbättra noggrannhet och robusthet.

### MLOps
- Modellen sparas efter varje epoch på Google Drive
- Organiserade mappar för versionshantering
- Träningsinformation sparas i en textfil

### Performance Metrics
- Tid per epoch och total träningstid mäts
- Noggrannhet mäts på valideringsdata efter varje epoch

### Data Augmentation
Bilderna varieras konstgjort under träning – rotation och skalning.
```
Resultat: 95.01% (sämre än utan augmentation för MNIST)
```
*Augmentation hjälper inte alltid – beror på datasetet.*

### CNN – Convolutional Neural Network
Bytte från ANN till CNN som är bättre på att förstå bilder.
```
CNN ser kanter, former och mönster
ANN ser bara lösa pixlar
```

### Olika arkitekturer
Testade 1, 2 och 3 convolutional lager och jämförde resultaten:
```
CNN 1 lager  →  98.53%
CNN 2 lager  →  98.84%
CNN 3 lager  →  99.11%  ← bäst
```

### Regularization
Lade till dropout, batch normalization och weight decay för att förhindra overfitting:
```
Utan regularization  →  99.11%
Med regularization   →  98.69%  (lägre men mer robust!)
```

### Hyperparameter Tuning
Testade 5 kombinationer av inställningar automatiskt:
```
Bästa kombinationen:
  Learning rate: 0.001
  Dropout:       0.25
  Filter:        32
  Kernel size:   3
  Noggrannhet:   98.24%
```

---

## Del 3 – Eget dataset + Transfer learning

### Dataset
Hund vs katt-dataset hämtat från Kaggle:
- 20 000 träningsbilder
- 2 500 testbilder
- 2 500 valideringsbilder

### Eget CNN från scratch
Byggde och tränade ett CNN från grunden på hund vs katt:
```
Noggrannhet: 78.77%
Träningstid: ~20 minuter (10 epochs)
```

### Transfer learning med ResNet50
Använde ResNet50 – ett nätverk förtränat på miljoner bilder. Frös alla lager utom det sista och tränade bara det sista lagret på hund vs katt:
```
Noggrannhet: 98.76%
```

**Skillnaden:**
```
Eget CNN   →  78.77%
ResNet50   →  98.76%
```
Transfer learning ger enormt mycket bättre resultat eftersom ResNet50 redan lärt sig känna igen former och mönster från miljoner bilder.

---

## Resultatöversikt

| Modell | Dataset | Noggrannhet |
|--------|---------|-------------|
| ANN | MNIST | 97.05% |
| ANN + Augmentation | MNIST | 95.01% |
| CNN 1 lager | MNIST | 98.53% |
| CNN 2 lager | MNIST | 98.84% |
| CNN 3 lager | MNIST | 99.11% |
| CNN + Regularization | MNIST | 98.69% |
| Hyperparameter tuning | MNIST | 98.24% |
| Eget CNN | Hund vs katt | 78.77% |
| ResNet50 (Transfer learning) | Hund vs katt | 98.76% |

---

## Tekniker och verktyg

```
Python        →  programmeringsspråk
PyTorch       →  AI-bibliotek för att bygga och träna nätverk
NumPy         →  matematikbibliotek för matriser
Google Colab  →  körmiljö med gratis GPU
Kaggle        →  plattform för att hämta dataset
GitHub        →  versionshantering av kod
```

---

## Köra koden

1. Öppna önskad `.ipynb`-fil i Google Colab
2. Aktivera GPU: `Runtime → Change runtime type → T4 GPU`
3. Kör alla celler i ordning: `Runtime → Run all`

> **OBS:** Del 3 kräver ett Kaggle-konto och API-nyckel för att ladda ner datasetet.
