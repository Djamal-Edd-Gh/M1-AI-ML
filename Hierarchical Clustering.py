"""
Objectif : Réaliser un clustering hiérarchique (CAH - Classification Ascendante Hiérarchique).

Contrairement à d'autres algorithmes (comme K-Means), le clustering hiérarchique 
ne nécessite pas de spécifier le nombre de clusters à l'avance. 
Il construit un arbre de classification (le dendrogramme) qui permet de visualiser 
la distance et les rapprochements entre chaque point et chaque groupe de points.
"""

# -----------------------------
# Import des bibliothèques
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram

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
 
 - Le clustering hiérarchique repose sur le calcul de distances (ex: distance Euclidienne).
 - Si les variables n'ont pas la même échelle (ex: centimètres vs millimètres), 
   les variables avec de grandes valeurs domineront le calcul de la distance globale.
 - Le MinMaxScaler est utilisé ici pour ramener proportionnellement toutes les 
   valeurs entre 0 et 1, afin que chaque variable ait le même poids.
"""

# -----------------------------
# 3. Construction du modèle de clustering
# -----------------------------

"""
Méthodes possibles pour le paramètre "method" dans linkage :
 
 - 'ward'     -> minimise la variance intra-cluster (recommandé) ⚠️ utilise OBLIGATOIREMENT la distance EUCLIDIENNE
 - 'single'   -> distance minimale (effet de chaînage)
 - 'complete' -> distance maximale (clusters compacts)
 - 'average'  -> distance moyenne (bon compromis)
 - 'weighted' -> moyenne pondérée simple
 - 'centroid' -> distance entre centroïdes (peut créer des inversions)
 - 'median'   -> variante centroid (instable aussi)
"""

# Calcul des distances et création des liaisons (fusions) avec la méthode de Ward
Z = linkage(X_scaled, method='ward')

# -----------------------------
# 4. Affichage du Dendrogramme
# -----------------------------
plt.figure(figsize=(10, 6))

# Génération de l'arbre
# On utilise color_threshold=2.5 pour forcer la coloration en 3 groupes.
# Cela trace une ligne imaginaire à une distance de 2.5 pour couper les branches.
dendrogram(Z, color_threshold=2.5)

plt.title("Dendrogram (Hierarchical Clustering - Iris + MinMaxScaler)")
plt.xlabel("Index des échantillons (Samples)")
plt.ylabel("Distance de fusion")
plt.show()

"""
Explication :
 
 - L'axe horizontal (X) représente chaque échantillon individuel.
 - L'axe vertical (Y) représente la "distance" à laquelle deux points ou deux groupes ont été fusionnés.
 - Une ligne horizontale indique une fusion entre deux groupes. Plus cette ligne est haute, plus les groupes fusionnés sont différents.
 - Le `color_threshold` permet de couper visuellement l'arbre pour révéler nos 3 clusters distincts.
"""

# -----------------------------
# 5. Heatmap avec clustering (Clustermap)
# -----------------------------
sns.clustermap(
    X_scaled,
    method='ward',
    metric='euclidean',
    cmap='viridis',
    figsize=(8, 8)
)

plt.show()

"""
Explication :
 
 - Le Clustermap est la combinaison d'une carte de chaleur (heatmap) et de dendrogrammes.
 - Les lignes (échantillons) et les colonnes (variables) sont réorganisées automatiquement pour rapprocher ce qui se ressemble.
 - Les couleurs de la heatmap correspondent aux valeurs des variables (après normalisation entre 0 et 1).
 - Cela permet de voir d'un coup d'œil quelles mesures caractérisent chaque cluster (par exemple, un groupe de fleurs avec de très petits pétales apparaîtra avec une couleur sombre dans ces colonnes).
"""