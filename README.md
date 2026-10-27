# ⚽💰 Football Finance Intelligence

### Analyser les joueurs de football comme des actifs financiers — Python · SQL · Machine Learning · Agents IA

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://football-finance-intelligence-6acara8yfxvutijsfweqls.streamlit.app/)

👉 **[Ouvrir le dashboard interactif](https://football-finance-intelligence-6acara8yfxvutijsfweqls.streamlit.app/)**

Ce projet applique les outils de l'**ingénierie financière** (croissance, analyse des écarts, rendement, risque, amortissement IAS 38, valorisation) au **marché des transferts du football**, puis utilise un **comité d'agents IA** pour produire des notes de recrutement chiffrées.

---

## 🔑 Résultats clés

| | Résultat |
|---|---|
| 🌍 **Marché** | Dépenses observées record de **12,1 Md€** en 2025/26. La Premier League capte **52 %** des dépenses du Big 5 et reste la **seule ligue importatrice nette** (−3,2 Md€ sur 3 saisons). |
| 📈 **Joueur-actif** | La valeur d'un joueur cesse de croître vers **23-24 ans** et se déprécie fortement après 29 ans. Un portefeuille de joueurs à plus de 10 M€ acheté fin 2020 perd **46 %** en 5 ans ; seuls les 21 ans et moins sont rentables (**+41 %**). |
| 🏟️ **Clubs** | L'argent explique **57 %** des points (Moneyball). Lens, Union Berlin et Brest font bien mieux que leur budget ; Brentford obtient un ROI de **+407 %** sur ses reventes. |
| 🤖 **Modèle** | Un modèle **XGBoost** explique **88 %** des écarts de valeur (R² = 0,88), avec **SHAP** pour expliquer chaque estimation. |
| 🇲🇦 **Maroc** | Le Maroc est devenu la **sélection africaine la plus valorisée** (+12,7 % par an depuis 2018). **77 %** de l'effectif type est né à l'étranger. Le Mondial 2022 a revalorisé ses joueurs de **+22 %** en 6 mois. |
| 🧠 **Agents IA** | Un comité **multi-agents** (scout, analyste financier, rédacteur) produit une note de recrutement complète, **sans jamais inventer de chiffres** : tous les calculs sont réalisés par des outils Python. |

---

## 🗂️ Les modules

| Module | Notebook | Contenu |
|---|---|---|
| 1. Marché des transferts | [`01_marche_des_transferts.ipynb`](01_marche_des_transferts.ipynb) | TCAM, décomposition volume / prix, parts de marché, indice HHI, flux entre ligues (Sankey) |
| 2. Le joueur comme actif | [`02_joueur_actif_financier.ipynb`](02_joueur_actif_financier.ipynb) | Cycle de vie de la valeur, rendement / risque, ratio de Sharpe, backtest, amortissement IAS 38 |
| 3. Rentabilité des transferts | [`03_rentabilite_transferts.ipynb`](03_rentabilite_transferts.ipynb) | ROI de revente, plus-values comptables, clubs « traders », Moneyball, matrice des clubs |
| 4. Modèle de valorisation | [`04_modele_valorisation.ipynb`](04_modele_valorisation.ipynb) | Régression linéaire, Random Forest, XGBoost, SHAP, validation croisée, joueurs sous-évalués |
| 5. Focus Maroc | [`05_focus_maroc.ipynb`](05_focus_maroc.ipynb) | Vivier marocain, diaspora, comparaison africaine, étude d'événement du Mondial 2022 |
| 6. Agents IA | [`06_agents_ia.ipynb`](06_agents_ia.ipynb) | Outils Python, agent analyste (Gemini), comité de recrutement multi-agents |
| 7. Dashboard | [`app.py`](app.py) | Application Streamlit en 7 pages |

---

## 🔬 Démarche : vérifier avant de conclure

Chaque résultat a été remis en question. Les principaux biais identifiés et corrigés :

- **Biais de survie** : la base suit surtout les joueurs encore actifs, ce qui gonfle la croissance apparente du marché. → Usage d'indicateurs relatifs (parts de marché, médianes).
- **Biais de sélection** : le classement par ratio de Sharpe ne retenait que les joueurs ayant réussi. → **Backtest** sans information future.
- **Championnats non couverts** : 18 championnats sans données de matchs faussaient le modèle. → Exclusion : le R² passe de 0,78 à **0,88**.
- **Joueurs prêtés** : un joueur prêté recevait le niveau de son club propriétaire, ce qui a conduit le comité IA à une **recommandation fausse**. → Calcul du **club réel** de chaque joueur et exclusion des prêts probables.
- **Micro-valeurs** : un ratio est trompeur sur les petites valeurs. → Seuil minimal de 1 M€.

> 💡 Leçon principale : **les agents IA amplifient les erreurs des données**. La solution consiste à corriger les données à la source, pas à reformuler la consigne.

---

## 🛠️ Stack technique

| Domaine | Outils |
|---|---|
| Données et SQL | DuckDB, SQL (CTE, fonctions de fenêtre, ASOF JOIN) |
| Analyse | Python, pandas, NumPy |
| Visualisation | Plotly (graphiques interactifs, Sankey, treemap) |
| Machine learning | scikit-learn, XGBoost, SHAP |
| IA générative | Google Gemini (appel de fonctions, multi-agents) |
| Application | Streamlit, Streamlit Community Cloud |
| Environnement | Google Colab, GitHub |

---

## ▶️ Lancer le projet

**Le dashboard** est accessible en ligne : [ouvrir l'application](https://football-finance-intelligence-6acara8yfxvutijsfweqls.streamlit.app/).

**En local :**
```bash
pip install -r requirements.txt
streamlit run app.py
```
La base Transfermarkt (environ 200 Mo) est téléchargée automatiquement au premier lancement.
Pour activer le comité IA, ajouter une clé Gemini dans `.streamlit/secrets.toml` :
```toml
GEMINI_API_KEY = "votre-cle"
```

**Les notebooks** s'ouvrent dans Google Colab et téléchargent eux-mêmes les données.

---

## ⚠️ Limites

- Les valeurs marchandes sont des **estimations Transfermarkt**, pas des prix de transaction.
- Seuls **10 %** des transferts ont un montant connu et positif.
- L'amortissement suppose un **contrat de 5 ans** (durée réelle non disponible).
- Le modèle de valorisation ne mesure ni le **potentiel**, ni les **blessures**, ni le **contexte** (géopolitique, politique de recrutement d'un club).
- Le championnat marocain n'est pas couvert par la base.

Le modèle et les agents sont des **outils de présélection** : la décision finale relève de l'analyse humaine.

---

## 📚 Source des données

Données issues de [Transfermarkt](https://www.transfermarkt.com) via le projet open source [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (David Cereijo).

---

## 👤 Auteur

**Ahmed Badi** — Ingénieur financier (EMSI Rabat), orienté data et BI
[GitHub](https://github.com/Ahmed-Badi3)
