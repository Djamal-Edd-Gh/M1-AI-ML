"""
Le dataset Iris est l’un des jeux de données les plus célèbres en apprentissage automatique et en statistiques. 
Il contient des informations sur 150 fleurs d’Iris, réparties en 3 espèces différentes :

- Iris setosa
- Iris versicolor
- Iris virginica

Pour chaque fleur, 4 mesures ont été enregistrées :

- Longueur du sépale (sepal length)
- Largeur du sépale (sepal width)
- Longueur du pétale (petal length)
- Largeur du pétale (petal width)

L’objectif classique avec ce dataset est de prédire l’espèce de la fleur à partir de ces mesures, 
ce qui en fait un excellent exemple pour l’apprentissage supervisé.
"""

# -----------------------------
# Import des bibliothèques
# -----------------------------
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Charger le dataset Iris
# -----------------------------
iris = load_iris()

# Créer un DataFrame avec les données
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Ajouter la colonne "species" pour l'espèce
df["species"] = iris.target

# Afficher les 5 premières lignes
print(df.head())

# -----------------------------
# Pairplot : visualiser les relations entre les variables
# -----------------------------
sns.pairplot(df, hue="species")
plt.suptitle("Pairplot : Comparaison des mesures par espèce", y=1.02)
plt.show()

"""
Explication :
    
 - Chaque point représente une fleur.
 - Les couleurs correspondent aux espèces.
 - Les diagonales montrent la distribution de chaque variable.
 - Si les espèces sont bien séparées dans un graphique, cela montre que la variable peut aider à les distinguer.
"""

# -----------------------------
# Boxplots : comparer la distribution des mesures par espèce
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Boxplot pour la longueur des sépales
sns.boxplot(data=df, x="species", y="sepal length (cm)", ax=axes[0,0])
axes[0,0].set_title("Sepal Length")
axes[0,0].set_xlabel("Espèce (0=setosa, 1=versicolor, 2=virginica)")
axes[0,0].set_ylabel("Longueur sépale (cm)")

# Boxplot pour la largeur des sépales
sns.boxplot(data=df, x="species", y="sepal width (cm)", ax=axes[0,1])
axes[0,1].set_title("Sepal Width")

# Boxplot pour la longueur des pétales
sns.boxplot(data=df, x="species", y="petal length (cm)", ax=axes[1,0])
axes[1,0].set_title("Petal Length")

# Boxplot pour la largeur des pétales
sns.boxplot(data=df, x="species", y="petal width (cm)", ax=axes[1,1])
axes[1,1].set_title("Petal Width")

plt.tight_layout()
plt.show()

"""
Explication :
    
 - La boîte représente l'intervalle interquartile (50% des données centrales).
 - La ligne à l'intérieur de la boîte est la médiane.
 - Les "whiskers" (traits) montrent les valeurs extrêmes.
 - Les points hors de la boîte sont des valeurs atypiques (outliers).
 - Si les boîtes d'espèces différentes ne se chevauchent pas, cela signifie que la variable distingue bien les espèces.
"""