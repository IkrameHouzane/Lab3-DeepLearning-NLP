# Lab3 Deep Learning - NLP avec PyTorch
---

## Objectif

Ce TP a pour objectif de se familiariser avec PyTorch et de construire des architectures de réseaux de neurones pour le traitement du langage naturel (NLP) en utilisant des modèles séquentiels.

Le TP est divisé en deux parties :
1. **Classification de textes arabes** avec RNN, Bi-RNN, GRU et LSTM
2. **Génération de texte** avec GPT-2 (Transformer)

---

## Partie 1 : Classification de textes arabes (Sport)

### 1. Collecte des données
- **Topic unique** : Sport 
- **Sources multiples** :
  - Al Jazeera Sport (4 pages scrapées) → 36 textes
  - Textes manuels variés (sources sportives diverses) → 25 textes
- **Total** : 61 textes uniques en arabe
- **Scores** : Attribués entre 0 et 10 (basés sur la longueur + bruit aléatoire)

### 2. Prétraitement NLP
Pipeline appliqué :
1. Suppression des diacritiques (Tashkeel)
2. Normalisation des lettres arabes (إأآ→ا, ى→ي, ة→ه)
3. Suppression de la ponctuation arabe (،؛؟!)
4. Tokenization
5. Suppression des stop words arabes
6. Conversion en IDs via un vocabulaire de 701 mots
7. Padding/troncature à 100 tokens

**Stemming (expérimentation)** : Le stemmer ISRI de NLTK a été testé. Il réduit le vocabulaire de 2230 à 1337 mots (-40%), mais **dégrade les performances** (MSE moyen 3.84 vs 2.94). La version sans stemming a été retenue pour les résultats finaux.

### 3. Modèles entraînés
Quatre architectures PyTorch comparées :
- **RNN Simple** : réseau récurrent basique
- **Bi-RNN** : RNN bidirectionnel (lit dans les 2 sens)
- **GRU** : Gated Recurrent Unit
- **LSTM** : Long Short-Term Memory

**Hyperparamètres** : embed_dim=64, hidden_dim=64, epochs=50, lr=0.0005, dropout=0.3, batch_size=8, gradient clipping=1.0

### 4. Résultats finaux

| Modèle | MSE ↓ | MAE ↓ | BLEU |
|--------|-------|-------|------|
| RNN Simple | 6.05 | 2.13 | 0.159 |
| Bi-RNN | 7.47 | 2.33 | 0.159 |
| GRU | 5.14 | 1.79 | 0.159 |
| **LSTM** 🏆 | **4.31** | **1.71** | 0.159 |

### 5. Analyse
- **LSTM obtient les meilleures performances** (MSE=4.31, MAE=1.71) : sa capacité à mémoriser les dépendances longues est un avantage.
- **GRU est très proche** : architecture simplifiée mais efficace.
- **Bi-RNN souffre d'overfitting** : avec seulement 61 textes, sa capacité (2x plus de paramètres) le pousse à mémoriser.
- **RNN Simple est limité** par le problème de disparition du gradient.
- **BLEU score bas (0.159)** : attendu car le BLEU n'est pas adapté à la régression.

---

## Partie 2 : Génération de texte avec GPT-2 arabe

### 1. Installation et chargement
- **Bibliothèques** : `transformers`, `datasets`, `accelerate`
- **Modèle** : `aubmindlab/aragpt2-base` (GPT-2 pré-entraîné en arabe)
- **Taille** : 135M paramètres, vocabulaire de 64 000 tokens

### 2. Préparation du dataset
- **Source** : 61 textes arabes sur le sport (Partie 1)
- **Nettoyage** : suppression des caractères non-arabes
- **Tokenization** : max_length=128, padding
- **Split** : 54 train / 7 test

### 3. Fine-tuning
- **Époques** : 5
- **Batch size** : 2
- **Durée** : ~2 min 19s sur GPU T4

**Résultats du fine-tuning :**

| Époque | Training Loss | Validation Loss |
|--------|---------------|-----------------|
| 1 | 6.70 | 4.89 |
| 2 | 5.71 | 4.66 |
| 3 | 4.38 | 4.55 |
| 4 | 4.04 | 4.53 |
| 5 | **3.72** | **4.49** |

### 4. Génération de texte

**Exemples de générations :**

| Prompt | Texte généré |
|--------|--------------|
| فاز فريق ريال مدريد | فاز فريق ريال مدريد على مضيفه أتلتيكو مدريد بنتيجة ثلاثة أهداف مقابل هدف واحد.. |
| أعلن الاتحاد الدولي لكرة القدم | أعلن الاتحاد الدولي لكرة القدم عن تمديد عقوبة بحق المهاجم المصري محمد صلاح... |
| تأهل المنتخب المغربي | تأهل المنتخب المغربي إلى نهائي كأس أمم أفريقيا بعد فوزه على الجزائر... |
| حقق اللاعب محمد صلاح | حقق اللاعب محمد صلاح جائزة أفضل لاعب في الدوري الإنجليزي للمرة الحادية عشرة... |

### 5. Comparaison avant/après fine-tuning

**Avant (modèle de base)** : génère un rapport de match complet (style journalistique long)
**Après (modèle fine-tuné)** : génère un résumé court et direct (style de nos textes scrapés)

Le fine-tuning a **adapté le modèle au style de notre dataset** (textes courts de 15-20 tokens).

---

## Ce que j'ai appris

- Le pipeline NLP complet : scraping → nettoyage → tokenization → vectorisation → modèle
- La différence entre RNN, GRU et LSTM (complexité vs performance)
- L'importance du learning rate et du dropout pour éviter le "modèle qui prédit la moyenne"
- La gestion de l'overfitting avec peu de données
- Le fine-tuning d'un Transformer pré-entraîné (GPT-2) sur un dataset personnalisé
- La génération de texte avec Hugging Face
- **La démarche scientifique** : tester (stemming), comparer, analyser, et choisir la meilleure version

---

## Technologies utilisées

- **Python 3**
- **PyTorch** (modèles RNN/GRU/LSTM)
- **Hugging Face Transformers** (GPT-2)
- **BeautifulSoup** (web scraping)
- **NLTK** (NLP arabe)
- **Google Colab** (GPU T4)
---
