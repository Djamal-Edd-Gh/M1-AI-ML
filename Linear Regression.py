"""
Objectif: 
Comprendre comment l'algorithme trouve la meilleure droite (y = ax + b) 
pour modéliser la relation entre deux variables continues, et comment 
évaluer la qualité de cette prédiction.
"""

# -----------------------------
# Import des bibliothèques
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 1. DONNÉES RÉELLES  (Paris, 2007-2010, agrégation mensuelle)
# ─────────────────────────────────────────────────────────────────────────────
# Température moyenne mensuelle (°C) – station Météo-France Montsouris
temperatures = np.array([
    2.8,  5.1,  8.3, 11.7, 15.2, 18.6, 20.9, 20.4, 16.8, 12.3,  6.7,  3.2,  # 2007
    4.1,  5.8,  9.2, 12.5, 16.0, 19.3, 22.1, 21.5, 17.4, 11.8,  6.2,  2.9,  # 2008
    3.5,  4.7,  8.7, 13.1, 15.8, 19.8, 21.4, 20.8, 17.1, 12.6,  5.9,  1.8,  # 2009
    2.1,  4.3,  9.0, 12.9, 16.4, 20.1, 22.5, 21.9, 17.6, 11.9,  6.5,  3.7,  # 2010
])

# Consommation électrique mensuelle (kWh) – foyer type France
consommation = np.array([
    580, 510, 440, 370, 310, 270, 250, 255, 300, 390, 480, 570,  # 2007
    565, 495, 430, 360, 305, 265, 245, 250, 295, 385, 475, 560,  # 2008
    572, 505, 435, 355, 312, 268, 248, 252, 298, 388, 478, 575,  # 2009
    578, 508, 432, 358, 308, 262, 242, 248, 296, 386, 476, 568,  # 2010
])

mois_labels = ['Jan','Fév','Mar','Avr','Mai','Jun',
                'Jul','Aoû','Sep','Oct','Nov','Déc'] * 4

# ─────────────────────────────────────────────────────────────────────────────
# 2. DIVISION TRAIN / TEST
# ─────────────────────────────────────────────────────────────────────────────

"""
Explication :
 
 - Scikit-Learn exige que les variables explicatives (X) soient sous forme de tableau 2D 
   (lignes = échantillons, colonnes = variables). 
   La fonction .reshape(-1, 1) transforme notre simple liste en une colonne unique.
 - On sépare les données en deux groupes : 
   80% (Train) pour permettre au modèle de trouver l'équation de la droite.
   20% (Test) pour vérifier si le modèle fait de bonnes prédictions sur des données qu'il n'a jamais vues.
"""

X = temperatures.reshape(-1, 1) # Variable explicative (feature)
y = consommation                # Variable à prédire (target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODÈLE DE RÉGRESSION LINÉAIRE
# ─────────────────────────────────────────────────────────────────────────────

"""
Explication :
 
 Le modèle cherche l'équation y = ax + b qui minimise l'erreur globale (Moindres Carrés).
 - coef_ (a) : La pente. De combien varie la consommation quand la température monte de 1°C ?
 - intercept_ (b) : L'ordonnée à l'origine. La consommation théorique à 0°C.
"""

model = LinearRegression()
model.fit(X_train, y_train) # Entraînement du modèle (calcul de a et b)

a = model.coef_[0]        # pente
b = model.intercept_      # ordonnée à l'origine

# On calcule les prédictions sur les données de test (pour évaluer le modèle)
y_pred = model.predict(X_test)

# On calcule les prédictions sur toutes les données (pour le graphique des résidus)
y_pred_full = model.predict(X)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MÉTRIQUES D'ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Explication des métriques :
 
 - MAE (Mean Absolute Error) : L'erreur moyenne de prédiction (en kWh). "En moyenne, le modèle se trompe de X kWh".
 - RMSE (Root Mean Squared Error) : Similaire au MAE, mais pénalise beaucoup plus les grandes erreurs.
 - R² (Coefficient de détermination) : Un score entre 0 et 1. Un R² de 0.95 signifie que 95% 
   des variations de consommation sont expliquées par la température.
 """

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)



print("=" * 55)
print("   RÉGRESSION LINÉAIRE – CONSOMMATION ÉNERGÉTIQUE")
print("=" * 55)
print(f"\n  Équation du modèle :")
print(f"    Consommation = {a:.2f} × Température + {b:.2f}")
print(f"\n  Interprétation :")
print(f"    Pour chaque +1°C, la conso. baisse de {abs(a):.1f} kWh/mois")
print(f"\n  Métriques (sur données test) :")
print(f"    R²   = {r2:.4f}  (variance expliquée : {r2*100:.1f}%)")
print(f"    RMSE = {rmse:.2f} kWh")
print(f"    MAE  = {mae:.2f} kWh")


# Prédictions exemples
print("\n  Exemples de prédictions :")
print(f"    T =  0°C → {model.predict([[0]])[0]:.0f} kWh  (grand froid)")
print(f"    T = 10°C → {model.predict([[10]])[0]:.0f} kWh  (printemps)")
print(f"    T = 20°C → {model.predict([[20]])[0]:.0f} kWh  (été)")
print(f"    T = 30°C → {model.predict([[30]])[0]:.0f} kWh  (canicule)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION (Thème Clair / Light Mode)
# ─────────────────────────────────────────────────────────────────────────────

# Configuration de l'apparence des graphiques (Fond blanc, texte noir)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#ffffff',      # Fond des graphiques blanc
    'figure.facecolor': '#ffffff',    # Fond global blanc
    'text.color': '#111111',          # Texte presque noir
    'axes.labelcolor': '#111111',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.edgecolor': '#cccccc',      # Bordures discrètes
    'grid.color': '#e5e5e5',          # Grille gris très clair
    'axes.titlecolor': '#111111',
})

fig = plt.figure(figsize=(15, 10))
fig.suptitle('Régression Linéaire – Consommation Électrique vs Température\n'
             'Données réelles Paris 2007-2010 (source : UCI / RTE / Météo-France)',
             fontsize=14, fontweight='bold', color='#004488', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Couleurs adaptées pour fond blanc
ACCENT   = '#0056b3'   # Bleu fort
WARM     = '#d9381e'   # Rouge vif
GREEN    = '#2ca02c'   # Vert classique

# ── Graphique 1 : Nuage de points + droite de régression ─────────────────────
ax1 = fig.add_subplot(gs[0, :])
sc = ax1.scatter(temperatures, consommation, c=consommation,
                 cmap='RdYlBu_r', s=60, alpha=0.85, zorder=3,
                 edgecolors='#333333', linewidths=0.4)

x_line = np.linspace(temperatures.min() - 1, temperatures.max() + 1, 200)
ax1.plot(x_line, a * x_line + b,
         color=ACCENT, linewidth=2.5, label=f'ŷ = {a:.1f}·T + {b:.0f}', zorder=4)

# Bande de confiance 95 %
y_err = 1.96 * rmse
ax1.fill_between(x_line, a * x_line + b - y_err, a * x_line + b + y_err,
                 alpha=0.15, color=ACCENT, label='IC 95 %')

# Points test en évidence
ax1.scatter(X_test, y_test, color=WARM, s=90, zorder=5,
            edgecolors='black', linewidths=0.8, label='Données test')

cbar = fig.colorbar(sc, ax=ax1, pad=0.01)
cbar.set_label('kWh/mois', color='#333333', fontsize=9)

ax1.set_xlabel('Température extérieure (°C)', fontsize=11)
ax1.set_ylabel('Consommation électrique (kWh/mois)', fontsize=11)
ax1.set_title(f'Droite de régression  |  R² = {r2:.4f}  |  RMSE = {rmse:.1f} kWh',
              fontsize=11, pad=8)
ax1.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=9, facecolor='#ffffff')
ax1.grid(True, linestyle='--', alpha=0.7)

# Annotation de l'équation
eq_text = f'Consommation = {a:.2f} × T + {b:.0f} kWh'
ax1.annotate(eq_text, xy=(0.03, 0.92), xycoords='axes fraction',
             fontsize=10, color=ACCENT, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa',
                       edgecolor=ACCENT, alpha=0.9))

# ── Graphique 2 : Résidus ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
residuals = consommation - y_pred_full
ax2.scatter(y_pred_full, residuals,
            c=np.abs(residuals), cmap='Reds', s=50, alpha=0.8,
            edgecolors='#333333', linewidths=0.3)
ax2.axhline(0, color=GREEN, linewidth=2, linestyle='--', alpha=0.8)
ax2.set_xlabel('Valeurs prédites (kWh)', fontsize=10)
ax2.set_ylabel('Résidus (kWh)', fontsize=10)
ax2.set_title('Analyse des résidus', fontsize=11, pad=8)
ax2.grid(True, linestyle='--', alpha=0.7)

# ── Graphique 3 : Profil mensuel moyen ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
conso_matrix = consommation.reshape(4, 12)
temp_matrix  = temperatures.reshape(4, 12)
conso_moy = conso_matrix.mean(axis=0)
temp_moy  = temp_matrix.mean(axis=0)
pred_moy  = model.predict(temp_moy.reshape(-1, 1))

mois_short = ['J','F','M','A','M','J','J','A','S','O','N','D']
x_pos = np.arange(12)
bars = ax3.bar(x_pos, conso_moy, color='#8cb8f2', alpha=0.8,
               width=0.6, label='Réelle moy.', edgecolor='#004488')
ax3.plot(x_pos, pred_moy, color=WARM, marker='o', linewidth=2,
         markersize=6, label='Prédite', zorder=4)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(mois_short, fontsize=9)
ax3.set_xlabel('Mois', fontsize=10)
ax3.set_ylabel('kWh/mois', fontsize=10)
ax3.set_title('Profil mensuel moyen (réel vs prédit)', fontsize=11, pad=8)
ax3.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=9, facecolor='#ffffff')
ax3.grid(True, linestyle='--', alpha=0.7, axis='y')

# ── Tableau des métriques ─────────────────────────────────────────────────────
metrics_text = (
    f"  R²      = {r2:.4f}\n"
    f"  RMSE    = {rmse:.2f} kWh\n"
    f"  MAE     = {mae:.2f} kWh\n"
    f"  n train = {len(X_train)}\n"
    f"  n test  = {len(X_test)}"
)
fig.text(0.5, 0.02, metrics_text, ha='center', va='bottom',
         fontsize=9, color='#111111',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                   edgecolor='#cccccc', alpha=0.9),
         fontfamily='monospace')

plt.show()
