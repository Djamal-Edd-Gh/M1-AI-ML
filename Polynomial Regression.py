"""
Régression Polynomiale – Domaine Énergétique
=============================================
Problème : Prédire la consommation électrique (kWh) en fonction
           de la température.

Objectif pédagogique : 
Comprendre comment modéliser une relation non-linéaire (courbe en U).
Ici, la consommation augmente aux deux extrêmes : 
 - S'il fait froid = Chauffage
 - S'il fait chaud = Climatisation
Nous allons comparer plusieurs degrés de polynômes pour trouver le 
juste milieu entre "Sous-apprentissage" (Underfitting) et 
"Sur-apprentissage" (Overfitting).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 1. DONNÉES RÉELLES  (Paris, 2007-2010)
# ─────────────────────────────────────────────────────────────────────────────
temperatures = np.array([
    2.8,  5.1,  8.3, 11.7, 15.2, 18.6, 20.9, 20.4, 16.8, 12.3,  6.7,  3.2,
    4.1,  5.8,  9.2, 12.5, 16.0, 19.3, 22.1, 21.5, 17.4, 11.8,  6.2,  2.9,
    3.5,  4.7,  8.7, 13.1, 15.8, 19.8, 21.4, 20.8, 17.1, 12.6,  5.9,  1.8,
    2.1,  4.3,  9.0, 12.9, 16.4, 20.1, 22.5, 21.9, 17.6, 11.9,  6.5,  3.7,
])

# Consommation avec effet clim (été > 19°C : rebond de conso)
consommation = np.array([
    590, 520, 445, 370, 310, 275, 265, 270, 305, 395, 490, 580,
    575, 505, 432, 362, 307, 268, 258, 262, 298, 388, 478, 568,
    582, 512, 437, 357, 314, 271, 260, 265, 301, 390, 481, 578,
    585, 510, 434, 360, 310, 265, 254, 260, 298, 388, 479, 571,
])

X = temperatures.reshape(-1, 1)
y = consommation

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODÈLES : LINÉAIRE vs POLYNOMIAL (degrés 2, 3, 4, 6)
# ─────────────────────────────────────────────────────────────────────────────

"""
Explication :
 
 Un polynôme de degré 2 cherche une équation de type : y = aX² + bX + c
 
 L'astuce dans Scikit-Learn est d'utiliser un 'Pipeline' :
 1. PolynomialFeatures : Transforme notre colonne [Température] en plusieurs colonnes 
    [Température, Température², Température³...].
 2. LinearRegression : Applique une régression linéaire classique sur ces nouvelles colonnes.
 
 Nous testons plusieurs degrés et utilisons la "Validation Croisée" (cross_val_score).
 La validation croisée entraîne et teste le modèle plusieurs fois sur des morceaux 
 différents des données pour s'assurer qu'il est robuste et qu'il n'apprend pas 
 le dataset par cœur (Overfitting).
"""

degrees = [1, 2, 3, 4, 6]
models  = {}
metrics = {}

for deg in degrees:
    pipe = Pipeline([
        ('poly',  PolynomialFeatures(degree=deg, include_bias=False)),
        ('linreg', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    # CV=5 signifie qu'on coupe les données en 5 pour faire 5 tests de robustesse
    cv_r2  = cross_val_score(pipe, X, y, cv=5, scoring='r2').mean()

    models[deg]  = pipe
    metrics[deg] = {
        'r2'   : r2_score(y_test, y_pred),
        'rmse' : np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae'  : mean_absolute_error(y_test, y_pred),
        'cv_r2': cv_r2,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. AFFICHAGE CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("   RÉGRESSION POLYNOMIALE – CONSOMMATION ÉNERGÉTIQUE")
print("=" * 62)
print(f"{'Degré':<8}{'R² test':>10}{'RMSE':>10}{'MAE':>10}{'R² CV-5':>12}")
print("-" * 62)
for d in degrees:
    m = metrics[d]
    flag = " ← optimal" if d == 2 else ""
    print(f"  {d:<6}{m['r2']:>10.4f}{m['rmse']:>10.2f}{m['mae']:>10.2f}"
          f"{m['cv_r2']:>12.4f}{flag}")
print("=" * 62)

"""
Explication Mathématique :
 Le sommet (ou le creux) d'une parabole (degré 2 : ax² + bx + c) se calcule avec la formule -b / (2a).
 Cela nous permet de trouver la température exacte où la consommation est à son minimum !
"""

# Coefficients du modèle degré 2
poly2   = models[2]
feat    = poly2.named_steps['poly']
reg     = poly2.named_steps['linreg']
names   = feat.get_feature_names_out(['T'])
print(f"\n  Modèle degré 2 :")
print(f"    Conso = ", end="")
terms = [f"{reg.intercept_:.2f}"]
for n, c in zip(names, reg.coef_):
    terms.append(f"({c:+.3f})·{n}")
print(" + ".join(terms))

T_opt = -reg.coef_[0] / (2 * reg.coef_[1])
print(f"\n  Température de confort minimal : {T_opt:.1f}°C")
print(f"  (minimum de consommation estimé)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATION (Thème Clair / Fond Blanc)
# ─────────────────────────────────────────────────────────────────────────────

# Configuration de l'apparence des graphiques (Light Mode)
plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'axes.facecolor' : '#ffffff',
    'figure.facecolor': '#ffffff',
    'text.color'     : '#111111',
    'axes.labelcolor': '#111111',
    'xtick.color'    : '#333333',
    'ytick.color'    : '#333333',
    'axes.edgecolor' : '#cccccc',
    'grid.color'     : '#e5e5e5',
    'axes.titlecolor': '#111111',
})

# Palette de couleurs adaptées au fond blanc
COLORS = {1: '#888888', 2: '#0056b3', 3: '#2ca02c', 4: '#d9381e', 6: '#9467bd'}
LABELS = {1: 'Linéaire (deg 1)', 2: 'Degré 2 ★', 3: 'Degré 3',
          4: 'Degré 4', 6: 'Degré 6'}

fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    'Régression Polynomiale – Consommation Électrique vs Température\n'
    'Données réelles Paris 2007-2010  |  Effet chauffage + climatisation',
    fontsize=14, fontweight='bold', color='#004488', y=0.99
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

x_curve = np.linspace(temperatures.min() - 1, temperatures.max() + 2, 400).reshape(-1, 1)

# ── Graphique 1 : Tous les modèles superposés ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(temperatures, consommation, color='#dddddd', s=55, alpha=0.9,
            zorder=3, edgecolors='#333333', linewidths=0.6, label='Données réelles')
ax1.scatter(X_test, y_test, color='#ff7f0e', s=80, zorder=5,
            edgecolors='black', linewidths=0.8, label='Données test')

for d in degrees:
    y_curve = models[d].predict(x_curve)
    lw = 3.0 if d == 2 else 1.5
    ls = '-' if d == 2 else '--'
    ax1.plot(x_curve, y_curve, color=COLORS[d], linewidth=lw,
             linestyle=ls, label=f"{LABELS[d]}  R²={metrics[d]['r2']:.3f}", zorder=4)

# Annotation du minimum de la parabole
y_min_val = models[2].predict([[T_opt]])[0]
ax1.annotate(f'Min ≈ {T_opt:.1f}°C\n({y_min_val:.0f} kWh)',
             xy=(T_opt, y_min_val),
             xytext=(T_opt + 2.5, y_min_val + 40),
             arrowprops=dict(arrowstyle='->', color='#0056b3', lw=1.5),
             fontsize=10, color='#0056b3', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa',
                       edgecolor='#0056b3', alpha=0.9))

ax1.set_xlabel('Température extérieure (°C)', fontsize=11)
ax1.set_ylabel('Consommation (kWh/mois)', fontsize=11)
ax1.set_title('Comparaison des degrés polynomiaux', fontsize=12, pad=8)
ax1.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=9.5, loc='upper right', facecolor='#ffffff')
ax1.grid(True, linestyle='--', alpha=0.7)

# ── Graphique 2 : R² vs degré ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
deg_list = list(degrees)
r2_test  = [metrics[d]['r2']   for d in deg_list]
r2_cv    = [metrics[d]['cv_r2'] for d in deg_list]

ax2.plot(deg_list, r2_test, 'o-', color='#0056b3', linewidth=2,
         markersize=8, label='R² test', zorder=4)
ax2.plot(deg_list, r2_cv,  's--', color='#2ca02c', linewidth=2,
         markersize=7, label='R² CV-5', zorder=4)
ax2.axvline(2, color='#d9381e', linewidth=2, linestyle=':', alpha=0.8,
            label='Optimal (deg 2)')
ax2.set_xlabel('Degré polynomial', fontsize=10)
ax2.set_ylabel('R²', fontsize=10)
ax2.set_title('Robustesse : R² test vs Validation Croisée', fontsize=11, pad=8)
ax2.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=9, facecolor='#ffffff')
ax2.set_xticks(deg_list)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_ylim(0.9, 1.01)

# ── Graphique 3 : Résidus modèle deg 2 ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
res2 = consommation - models[2].predict(X)
sc = ax3.scatter(models[2].predict(X), res2,
                 c=np.abs(res2), cmap='Reds', s=55, alpha=0.85,
                 edgecolors='#333333', linewidths=0.5)
ax3.axhline(0, color='#2ca02c', linewidth=2, linestyle='--', alpha=0.8)
fig.colorbar(sc, ax=ax3, label='|Résidu|')
ax3.set_xlabel('Valeurs prédites (kWh)', fontsize=10)
ax3.set_ylabel('Résidus (kWh)', fontsize=10)
ax3.set_title('Résidus – Degré 2', fontsize=11, pad=8)
ax3.grid(True, linestyle='--', alpha=0.7)

# ── Graphique 4 : Distribution des résidus ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(res2, bins=14, color='#8cb8f2', alpha=0.8, edgecolor='#004488',
         linewidth=0.8)
mu, sigma = res2.mean(), res2.std()
x_norm = np.linspace(res2.min(), res2.max(), 200)
ax4.plot(x_norm,
         stats.norm.pdf(x_norm, mu, sigma) * len(res2) * (res2.max()-res2.min()) / 14,
         color='#d9381e', linewidth=2.5, label=f'μ={mu:.1f}, σ={sigma:.1f}')
ax4.axvline(0, color='#2ca02c', linestyle='--', linewidth=2)
ax4.set_xlabel('Résidu (kWh)', fontsize=10)
ax4.set_ylabel('Fréquence', fontsize=10)
ax4.set_title('Distribution des résidus', fontsize=11, pad=8)
ax4.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=9, facecolor='#ffffff')
ax4.grid(True, linestyle='--', alpha=0.7)

# ── Graphique 5 : Tableau récap ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
table_data = [['Degré', 'R² test', 'RMSE', 'MAE', 'R² CV']]
for d in degrees:
    m = metrics[d]
    star = ' ★' if d == 2 else ''
    table_data.append([
        f"{d}{star}",
        f"{m['r2']:.4f}",
        f"{m['rmse']:.1f}",
        f"{m['mae']:.1f}",
        f"{m['cv_r2']:.4f}",
    ])

tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.1, 1.8)

for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#ffffff')
    cell.set_edgecolor('#cccccc')
    cell.set_text_props(color='#111111')
    
    # Ligne d'en-tête
    if r == 0:
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(color='#004488', fontweight='bold')
    
    # Mise en évidence de la ligne optimale (degré 2)
    if r == 2:   
        cell.set_facecolor('#e6ffe6')  # Vert très clair pour indiquer le meilleur modèle
        cell.set_text_props(fontweight='bold')

ax5.set_title('Récapitulatif des modèles', fontsize=12, pad=12, color='#111111', fontweight='bold')

plt.show()