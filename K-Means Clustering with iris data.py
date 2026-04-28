"""
Objectif : Réaliser un partitionnement avec l'algorithme K-Means.

Contrairement au clustering hiérarchique qui construit un arbre, l'algorithme K-Means 
nécessite de définir à l'avance le nombre de groupes (K) que l'on souhaite trouver.
Il cherche ensuite à regrouper les données autour de K "centroïdes" (centres de gravité)
en minimisant la distance entre les points et le centroïde de leur groupe.
"""

# -----------------------------
# Import des bibliothèques
# -----------------------------
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# -----------------------------
# 1. Charger les données
# -----------------------------
iris = load_iris()
X = iris.data

# -----------------------------
# 2. Normalisation des données
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

"""
Explication :
 
 - Tout comme le clustering hiérarchique, K-Means utilise des calculs de distance 
   (généralement la distance Euclidienne) pour assigner chaque point à un centroïde.
 - La normalisation (ici entre 0 et 1) reste indispensable pour éviter qu'une 
   variable avec de grandes valeurs ne dicte à elle seule la formation des clusters.
"""

# -----------------------------
# 3. Trouver le bon K : La méthode du coude (Elbow Method)
# -----------------------------

"""
Explication :
 
 Comment choisir la valeur de K si on ne la connaît pas à l'avance ?
 On entraîne le modèle avec K=1, puis K=2, etc., jusqu'à K=10.
 Pour chaque essai, on récupère "l'inertie" (la somme des distances au carré entre 
 chaque point et son centroïde). Plus l'inertie est basse, plus les clusters sont denses.
 On cherche le "coude" sur le graphique : le point où rajouter un cluster supplémentaire 
 ne fait plus chuter l'inertie de manière significative.
"""

inerties = []
K_range = range(1, 11)

for k in K_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_test.fit(X_scaled)
    inerties.append(kmeans_test.inertia_)

# Affichage du graphique du coude
plt.figure(figsize=(8, 5))
plt.plot(K_range, inerties, marker='o', linestyle='--')
plt.title("Méthode du Coude (Elbow Method)")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertie")
plt.xticks(K_range)
plt.show()

# -----------------------------
# 4. Modèle K-Means final
# -----------------------------

"""
Explication :
 
 D'après le graphique du coude (et parce qu'on connaît le dataset Iris), 
 la cassure se situe autour de K=3. Nous allons donc entraîner notre modèle 
 final avec 3 clusters.
"""

# Entraînement du modèle avec K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X_scaled)

# Récupération des étiquettes (le numéro du cluster attribué à chaque fleur)
labels = kmeans.labels_

# Récupération des coordonnées des centroïdes (les "centres" des 3 groupes)
centroides = kmeans.cluster_centers_

# -----------------------------
# 5. Visualisation des clusters
# -----------------------------

"""
Explication :
 
 Nos données ont 4 dimensions (4 variables). Il est impossible de les dessiner en 4D.
 Pour la visualisation, nous allons choisir de ne tracer que 2 variables :
 par exemple, la longueur des sépales (colonne 0) et la largeur des sépales (colonne 1).
 
 Nous colorerons les points selon le cluster trouvé par K-Means, et nous 
 placerons une croix rouge à l'emplacement exact de chaque centroïde.
"""

plt.figure(figsize=(10, 6))

# Nuage de points (Scatter plot) des deux premières variables
# c=labels permet de colorer les points selon les groupes trouvés par le K-Means
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

# Ajout des centroïdes sur le graphique (croix rouges)
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='X', s=200, label='Centroïdes')

plt.title("Clustering K-Means (Visualisation sur les Sépales)")
plt.xlabel("Sepal Length (normalisé)")
plt.ylabel("Sepal Width (normalisé)")
plt.legend()
plt.show()
