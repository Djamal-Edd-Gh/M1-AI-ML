# -----------------------------
# Import des librairies nécessaires
# -----------------------------
from sklearn.datasets import load_iris               # Pour charger le dataset Iris
from sklearn.model_selection import train_test_split # Pour séparer les données en train/test
from sklearn.neighbors import KNeighborsClassifier   # Pour créer un modèle KNN
from sklearn.metrics import accuracy_score           # Pour évaluer la précision du modèle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay  # Pour afficher la matrice de confusion

# -----------------------------
# 1. Charger les données Iris
# -----------------------------
iris = load_iris()    # Charge le dataset Iris
X = iris.data         # Variables explicatives (features) : longueur/largeur sépale/pétale
y = iris.target       # Cible (labels) : 0=setosa, 1=versicolor, 2=virginica

# -----------------------------
# 2. Séparer les données en train et test
# -----------------------------
"""
On garde 80% des données pour entraîner le modèle et 20% pour le tester
random_state=42 permet de reproduire exactement la même séparation à chaque exécution
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Créer le modèle KNN
# -----------------------------
"""
KNN (K-Nearest Neighbors) classe une nouvelle observation
selon les k voisins les plus proches dans l'espace des features.
Paramètres importants :
n_neighbors=5 → nombre de voisins pris en compte pour la prédiction
weights="distance" → les voisins proches comptent plus que les plus éloignés
p → type de distance : 1=Manhattan, 2=Euclidienne
"""

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    p=2
)

# -----------------------------
# 4. Entraîner le modèle
# -----------------------------
knn.fit(X_train, y_train)  # Apprentissage à partir des données d'entraînement

# -----------------------------
# 5. Faire des prédictions
# -----------------------------
y_pred = knn.predict(X_test)  # Prédire les labels pour les données de test

# -----------------------------
# 6. Évaluer la précision
# -----------------------------
# accuracy_score compare les prédictions aux vraies valeurs
print("Accuracy :", accuracy_score(y_test, y_pred))

# -----------------------------
# 7. Matrice de confusion
# -----------------------------
"""La matrice de confusion permet de voir quelles classes ont été bien ou mal prédites"""

ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.title("Matrice de confusion KNN")
plt.show()

"""Explications pour les étudiants :
 - La précision (accuracy) correspond au pourcentage de bonnes prédictions.
 - La matrice de confusion montre combien d'exemples ont été correctement
   classés pour chaque espèce et où le modèle s'est trompé.
 - KNN est simple mais puissant : la qualité dépend du choix de k, du type de distance et du poids des voisins.
"""