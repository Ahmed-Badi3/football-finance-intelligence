"""
⚽💰 Football Finance Intelligence — Dashboard Streamlit
Analyser les joueurs de football comme des actifs financiers.

Données : Transfermarkt (projet open source transfermarkt-datasets).
Auteur : Ahmed — Ingénieur financier (EMSI Rabat).
"""

import os
import urllib.request
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Football Finance Intelligence", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
URL_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"
FICHIER_BASE = "transfermarkt-datasets.duckdb"
FICHIER_VALO = "valorisation_joueurs.csv"

BIG5 = ("GB1", "ES1", "IT1", "L1", "FR1")
NOMS_LIGUES = {"GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A",
               "L1": "Bundesliga", "FR1": "Ligue 1"}
COULEURS_LIGUES = {"Premier League": "#636EFA", "LaLiga": "#EF553B", "Ligue 1": "#00CC96",
                   "Serie A": "#AB63FA", "Bundesliga": "#FFA15A"}
ROUGE_MAROC, VERT_MAROC = "#C1272D", "#006233"
URL_GITHUB = "https://github.com/Ahmed-Badi3/football-finance-intelligence"


def annee_sql(colonne):
    """Transforme une saison '25/26' en année de début (2025)."""
    return (f"CASE WHEN CAST(LEFT({colonne}, 2) AS INT) > 50 "
            f"THEN 1900 + CAST(LEFT({colonne}, 2) AS INT) "
            f"ELSE 2000 + CAST(LEFT({colonne}, 2) AS INT) END")


# ---------------------------------------------------------------------------
# Connexion à la base (téléchargée une seule fois)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Téléchargement de la base Transfermarkt (environ 200 Mo, une seule fois)…")
def connexion():
    if not os.path.exists(FICHIER_BASE):
        urllib.request.urlretrieve(URL_BASE, FICHIER_BASE)
    con = duckdb.connect(FICHIER_BASE, read_only=True)
    con.execute("SET memory_limit = '700MB'")
    return con


def q(sql):
    return connexion().cursor().execute(sql).df()


# ---------------------------------------------------------------------------
# Module 1 — Marché des transferts
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calcul du marché des transferts…")
def donnees_marche():
    saisons = q(f"""
        SELECT transfer_season AS saison, {annee_sql('transfer_season')} AS annee,
               COUNT(*) AS nb_transferts,
               CAST(SUM(transfer_fee) / 1e6 AS DOUBLE) AS depenses_M
        FROM transfers
        WHERE transfer_fee > 0 AND transfer_date <= CURRENT_DATE
        GROUP BY 1, 2
        ORDER BY annee
    """)
    saisons = saisons[(saisons["annee"] >= 2012) & (saisons["annee"] <= 2025)].copy()
    saisons["prix_moyen_M"] = saisons["depenses_M"] / saisons["nb_transferts"]
    n0, p0 = saisons["nb_transferts"].shift(1), saisons["prix_moyen_M"].shift(1)
    saisons["effet_volume"] = (saisons["nb_transferts"] - n0) * p0
    saisons["effet_prix"] = (saisons["prix_moyen_M"] - p0) * saisons["nb_transferts"]
    saisons["variation_pct"] = saisons["depenses_M"].pct_change() * 100

    big5 = q(f"""
        SELECT t.transfer_season AS saison, {annee_sql('t.transfer_season')} AS annee,
               c.domestic_competition_id AS ligue,
               CAST(SUM(t.transfer_fee) / 1e6 AS DOUBLE) AS depenses_M
        FROM transfers t
        JOIN clubs c ON t.to_club_id = c.club_id
        WHERE t.transfer_fee > 0 AND t.transfer_date <= CURRENT_DATE
          AND c.domestic_competition_id IN {BIG5}
        GROUP BY 1, 2, 3
        ORDER BY annee
    """)
    big5 = big5[(big5["annee"] >= 2012) & (big5["annee"] <= 2025)].copy()
    big5["ligue"] = big5["ligue"].map(NOMS_LIGUES)
    parts = big5.pivot_table(index="saison", columns="ligue", values="depenses_M",
                             aggfunc="sum", sort=False)
    parts = parts.div(parts.sum(axis=1), axis=0) * 100
    hhi = (parts ** 2).sum(axis=1)
    return saisons, big5, parts, hhi


# ---------------------------------------------------------------------------
# Module 2 — Le joueur comme actif financier
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calcul des rendements par âge…")
def donnees_actif():
    return q("""
        WITH annuel AS (
            SELECT player_id, YEAR(date) AS annee,
                   arg_max(market_value_in_eur, date) AS valeur
            FROM player_valuations
            WHERE market_value_in_eur > 0
            GROUP BY 1, 2
        ),
        avec_prec AS (
            SELECT *,
                   LAG(valeur) OVER (PARTITION BY player_id ORDER BY annee) AS valeur_prec,
                   LAG(annee)  OVER (PARTITION BY player_id ORDER BY annee) AS annee_prec
            FROM annuel
        ),
        rend AS (
            SELECT a.annee - YEAR(TRY_CAST(p.date_of_birth AS DATE)) AS age,
                   a.valeur * 1.0 / a.valeur_prec - 1 AS rendement
            FROM avec_prec a
            JOIN players p ON a.player_id = p.player_id
            WHERE a.valeur_prec IS NOT NULL
              AND a.annee = a.annee_prec + 1
              AND a.annee BETWEEN 2016 AND 2025
              AND p.date_of_birth IS NOT NULL
        )
        SELECT age,
               COUNT(*) AS nb,
               MEDIAN(rendement) * 100 AS rendement_median,
               AVG(CASE WHEN rendement > 0 THEN 1 ELSE 0 END) * 100 AS pct_hausse,
               AVG(LN(1 + rendement)) * 100 AS rendement_moyen_log,
               STDDEV_SAMP(LN(1 + rendement)) * 100 AS volatilite
        FROM rend
        WHERE age BETWEEN 17 AND 36
        GROUP BY age
        ORDER BY age
    """)


# ---------------------------------------------------------------------------
# Module 3 — Rentabilité des transferts et Moneyball
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reconstitution des cycles achat → revente…")
def donnees_reventes():
    cycles = q("""
        WITH achats AS (
            SELECT t.player_id, t.player_name,
                   t.to_club_id AS club_id, t.to_club_name AS club,
                   t.transfer_date AS date_achat,
                   CAST(t.transfer_fee AS DOUBLE) AS prix_achat,
                   YEAR(t.transfer_date) - YEAR(TRY_CAST(p.date_of_birth AS DATE)) AS age_achat
            FROM transfers t
            JOIN players p ON t.player_id = p.player_id
            WHERE t.transfer_fee > 0
              AND t.transfer_date BETWEEN DATE '2012-07-01' AND CURRENT_DATE
        ),
        ventes AS (
            SELECT player_id, from_club_id AS club_id, transfer_date AS date_vente,
                   CAST(transfer_fee AS DOUBLE) AS prix_vente, to_club_name AS acheteur
            FROM transfers
            WHERE transfer_fee > 0 AND transfer_date <= CURRENT_DATE
        ),
        cycles AS (
            SELECT a.*, v.date_vente, v.prix_vente, v.acheteur,
                   ROW_NUMBER() OVER (PARTITION BY a.player_id, a.club_id, a.date_achat
                                      ORDER BY v.date_vente) AS rn
            FROM achats a
            JOIN ventes v ON a.player_id = v.player_id
                         AND a.club_id = v.club_id
                         AND v.date_vente > a.date_achat
        )
        SELECT * EXCLUDE (rn) FROM cycles WHERE rn = 1
    """)
    cycles["duree_ans"] = (pd.to_datetime(cycles["date_vente"])
                           - pd.to_datetime(cycles["date_achat"])).dt.days / 365.25
    cycles = cycles[cycles["duree_ans"] >= 0.5].copy()
    cycles["age_achat"] = cycles["age_achat"].astype(float)
    cycles["prix_achat_M"] = cycles["prix_achat"] / 1e6
    cycles["prix_vente_M"] = cycles["prix_vente"] / 1e6
    cycles["roi"] = cycles["prix_vente"] / cycles["prix_achat"] - 1
    cycles["plus_value_M"] = cycles["prix_vente_M"] - cycles["prix_achat_M"]
    vnc = cycles["prix_achat_M"] * (1 - cycles["duree_ans"].clip(upper=5) / 5)
    cycles["pv_comptable_M"] = cycles["prix_vente_M"] - vnc
    cycles["tranche"] = pd.cut(cycles["age_achat"], bins=[0, 21, 24, 27, 45],
                               labels=["21 ans et -", "22-24", "25-27", "28 et +"])

    par_age = (cycles.groupby("tranche", observed=True)
               .agg(nb=("roi", "count"),
                    roi_median=("roi", lambda s: s.median() * 100),
                    pct_revendu_plus_cher=("roi", lambda s: (s > 0).mean() * 100),
                    pct_pv_comptable=("pv_comptable_M", lambda s: (s > 0).mean() * 100))
               .reset_index())
    par_age["tranche"] = par_age["tranche"].astype(str)

    clubs = (cycles.groupby("club_id")
             .agg(club=("club", "first"),
                  nb_cycles=("roi", "count"),
                  investi_M=("prix_achat_M", "sum"),
                  encaisse_M=("prix_vente_M", "sum"),
                  plus_value_M=("plus_value_M", "sum"),
                  pct_gagnants=("roi", lambda s: (s > 0).mean() * 100))
             .reset_index())
    clubs["roi_global"] = (clubs["encaisse_M"] / clubs["investi_M"] - 1) * 100

    top_flop = (cycles[["player_name", "club", "acheteur", "age_achat", "prix_achat_M",
                        "prix_vente_M", "duree_ans", "plus_value_M"]])
    synthese = {"nb_cycles": len(cycles),
                "pct_plus_cher": (cycles["roi"] > 0).mean() * 100,
                "pct_pv_comptable": (cycles["pv_comptable_M"] > 0).mean() * 100,
                "roi_median": cycles["roi"].median() * 100}
    return par_age, clubs, top_flop, synthese


@st.cache_data(show_spinner="Calcul du Moneyball (valeur des effectifs vs points)…")
def donnees_moneyball():
    mb = q(f"""
        WITH pts AS (
            SELECT cg.club_id, g.competition_id AS ligue,
                   TRY_CAST(g.season AS INT) AS saison,
                   COUNT(*) AS matchs,
                   SUM(CASE WHEN cg.own_goals > cg.opponent_goals THEN 3
                            WHEN cg.own_goals = cg.opponent_goals THEN 1
                            ELSE 0 END) AS points
            FROM club_games cg
            JOIN games g ON cg.game_id = g.game_id
            WHERE g.competition_id IN {BIG5}
            GROUP BY 1, 2, 3
        ),
        effectif AS (
            SELECT a.player_club_id AS club_id,
                   TRY_CAST(g.season AS INT) AS saison,
                   a.player_id,
                   SUM(a.minutes_played) AS minutes,
                   ANY_VALUE(MAKE_DATE(TRY_CAST(g.season AS INT), 8, 1)) AS date_ref
            FROM appearances a
            JOIN games g ON a.game_id = g.game_id
            WHERE g.competition_id IN {BIG5}
            GROUP BY 1, 2, 3
            HAVING SUM(a.minutes_played) >= 450
        ),
        valo AS (
            SELECT e.club_id, e.saison, e.player_id, v.market_value_in_eur AS valeur
            FROM effectif e
            ASOF LEFT JOIN player_valuations v
              ON e.player_id = v.player_id AND e.date_ref >= v.date
        ),
        val_club AS (
            SELECT club_id, saison, SUM(valeur) / 1e6 AS valeur_effectif_M
            FROM valo
            GROUP BY 1, 2
        )
        SELECT p.*, c.name AS club, vc.valeur_effectif_M
        FROM pts p
        JOIN val_club vc ON p.club_id = vc.club_id AND p.saison = vc.saison
        LEFT JOIN clubs c ON p.club_id = c.club_id
        WHERE p.saison BETWEEN 2015 AND 2025
    """)
    mb = mb[(mb["matchs"] >= 30) & (mb["valeur_effectif_M"] > 0)].copy()
    mb["ligue"] = mb["ligue"].map(NOMS_LIGUES)
    mb["ppm"] = mb["points"] / mb["matchs"]
    mb["valeur_rel"] = mb["valeur_effectif_M"] / mb.groupby(["ligue", "saison"])["valeur_effectif_M"].transform("mean")
    mb["log_val"] = np.log(mb["valeur_rel"])
    b, a = np.polyfit(mb["log_val"], mb["ppm"], 1)
    r2 = np.corrcoef(mb["log_val"], mb["ppm"])[0, 1] ** 2
    mb["ppm_predit"] = a + b * mb["log_val"]
    mb["surperf_pts"] = (mb["ppm"] - mb["ppm_predit"]) * 38
    mb["saison_txt"] = mb["saison"].astype(int).astype(str)
    return mb, a, b, r2


# ---------------------------------------------------------------------------
# Module 4 — Valorisation (fichier exporté du notebook) + détection des prêts
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement du modèle de valorisation…")
def donnees_valo():
    valo = pd.read_csv(FICHIER_VALO)
    clubs_reels = q("""
        WITH minutes_par_club AS (
            SELECT a.player_id, a.player_club_id AS club_reel_id, c.name AS club_reel,
                   ROW_NUMBER() OVER (PARTITION BY a.player_id
                                      ORDER BY SUM(a.minutes_played) DESC) AS rang
            FROM appearances a
            JOIN games g ON a.game_id = g.game_id
            JOIN clubs c ON a.player_club_id = c.club_id
            WHERE TRY_CAST(g.season AS INT) = 2025
            GROUP BY a.player_id, a.player_club_id, c.name
        )
        SELECT m.player_id, m.club_reel,
               TRY_CAST(p.current_club_id AS INT) <> m.club_reel_id AS pret_probable
        FROM minutes_par_club m
        JOIN players p ON m.player_id = p.player_id
        WHERE m.rang = 1
    """)
    valo = valo.merge(clubs_reels, on="player_id", how="left")
    valo["pret_probable"] = valo["pret_probable"].astype("boolean").fillna(False).astype(bool)
    valo["club_reel"] = valo["club_reel"].fillna(valo["club"])
    return valo


@st.cache_data
def historique_valeur(player_id):
    return q(f"""
        SELECT YEAR(date) AS annee,
               arg_max(market_value_in_eur, date) / 1e6 AS valeur_M
        FROM player_valuations
        WHERE player_id = {int(player_id)}
        GROUP BY 1
        ORDER BY 1
    """)


# ---------------------------------------------------------------------------
# Module 5 — Focus Maroc
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calcul de la valeur des sélections africaines…")
def donnees_maroc():
    afrique = q("""
        WITH joueurs AS (
            SELECT player_id,
                   CASE WHEN country_of_citizenship = 'Morocco' THEN 'Maroc'
                        WHEN country_of_citizenship = 'Senegal' THEN 'Sénégal'
                        WHEN country_of_citizenship = 'Egypt'   THEN 'Égypte'
                        WHEN country_of_citizenship ILIKE '%ivoire%' THEN 'Côte d''Ivoire'
                        WHEN country_of_citizenship = 'Nigeria' THEN 'Nigeria'
                        WHEN country_of_citizenship = 'Algeria' THEN 'Algérie'
                   END AS pays
            FROM players
            WHERE COALESCE(international_caps, 0) >= 1
        ),
        annuel AS (
            SELECT player_id, YEAR(date) AS annee,
                   arg_max(market_value_in_eur, date) AS valeur
            FROM player_valuations
            WHERE market_value_in_eur > 0
            GROUP BY 1, 2
        ),
        classe AS (
            SELECT j.pays, a.annee, a.valeur,
                   ROW_NUMBER() OVER (PARTITION BY j.pays, a.annee ORDER BY a.valeur DESC) AS rang
            FROM joueurs j
            JOIN annuel a ON j.player_id = a.player_id
            WHERE j.pays IS NOT NULL AND a.annee BETWEEN 2012 AND 2026
        )
        SELECT pays, annee, SUM(valeur) / 1e6 AS valeur_top26_M
        FROM classe
        WHERE rang <= 26
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)
    vivier = q("""
        SELECT p.name, p.position, p.country_of_birth AS pays_naissance,
               p.current_club_name AS club,
               COALESCE(c.name, 'hors base') AS championnat,
               p.market_value_in_eur / 1e6 AS valeur_M,
               DATE_DIFF('day', CAST(p.date_of_birth AS DATE), CURRENT_DATE) / 365.25 AS age,
               COALESCE(p.international_caps, 0) AS selections
        FROM players p
        LEFT JOIN competitions c ON p.current_club_domestic_competition_id = c.competition_id
        WHERE p.country_of_citizenship = 'Morocco'
          AND TRY_CAST(p.last_season AS INT) >= 2025
    """)
    vivier["championnat"] = vivier["championnat"].str.replace("-", " ").str.title()
    vivier["club"] = vivier["club"].fillna("Club inconnu")
    vivier["valeur_M"] = vivier["valeur_M"].fillna(0)
    vivier["naissance"] = np.where(vivier["pays_naissance"] == "Morocco",
                                   "Né au Maroc", "Né à l'étranger")
    return afrique, vivier


# ---------------------------------------------------------------------------
# Module 6 — Outils des agents IA
# ---------------------------------------------------------------------------
def executer_sql(requete: str) -> str:
    """Exécute une requête SQL en lecture seule (SELECT) sur la base Transfermarkt
    et renvoie au maximum 50 lignes de résultat sous forme de texte."""
    texte = requete.strip().rstrip(";")
    if not texte.lower().startswith(("select", "with")):
        return "Refusé : seules les requêtes SELECT sont autorisées."
    try:
        resultat = q(f"SELECT * FROM ({texte}) LIMIT 50")
        return resultat.to_string(index=False) if len(resultat) else "Aucun résultat."
    except Exception as e:
        return f"Erreur SQL : {e}"


def chercher_sous_evalues(poste: str = "", age_max: float = 40,
                          valeur_min_M: float = 1.0, valeur_max_M: float = 1000,
                          ligue: str = "", nationalite: str = "",
                          nb_resultats: int = 10, inclure_prets: bool = False) -> str:
    """Cherche les joueurs les plus sous-évalués selon le modèle de valorisation.
    poste : 'Defender', 'Midfield', 'Attack' ou 'Goalkeeper' (vide = tous).
    valeur_min_M : 1 M€ par défaut. En dessous, le modèle est peu fiable (erreur > 40 %).
    ligue : code du championnat, par exemple 'GB1', 'ES1', 'IT1', 'L1', 'FR1',
            'PO1', 'NL1', 'BE1' (vide = tous).
    nationalite : en anglais, par exemple 'Morocco' (vide = toutes).
    inclure_prets : False par défaut, pour exclure les prêts probables (estimation biaisée).
    Seuls les joueurs ayant joué au moins 900 minutes cette saison sont retenus.
    Ratio = valeur estimée / valeur réelle."""
    valo = donnees_valo()
    df = valo[(valo["minutes_n"] >= 900) & (valo["age"] <= age_max)
              & (valo["valeur_M"] >= valeur_min_M) & (valo["valeur_M"] <= valeur_max_M)]
    if not inclure_prets:
        df = df[~df["pret_probable"]]
    if poste:
        df = df[df["position"] == poste]
    if ligue:
        df = df[df["ligue"] == ligue]
    if nationalite:
        df = df[df["nationalite"] == nationalite]
    if df.empty:
        return "Aucun joueur ne correspond à ces critères."
    colonnes = ["name", "club", "club_reel", "ligue", "nationalite", "position", "age",
                "minutes_n", "valeur_M", "valeur_estimee_M", "ratio"]
    return df.nlargest(int(nb_resultats), "ratio")[colonnes].round(2).to_string(index=False)


def fiche_joueur(nom: str) -> str:
    """Renvoie le profil d'un joueur : club enregistré, club réel de la saison,
    championnat, âge, poste, contrat, statistiques, valeur réelle, valeur estimée
    par le modèle et historique de sa valeur. Signale les prêts probables."""
    valo = donnees_valo()
    trouves = valo[valo["name"].str.contains(nom, case=False, na=False)]
    if trouves.empty:
        return f"Aucun joueur trouvé pour '{nom}' dans le fichier de valorisation."
    j = trouves.iloc[0]
    histo = historique_valeur(j["player_id"]).tail(6).round(1)
    alerte = ""
    if j["pret_probable"]:
        alerte = (f"\n⚠️ ALERTE : le joueur est enregistré à {j['club']} mais a joué la saison "
                  f"à {j['club_reel']} (prêt probable). Son estimation est BIAISÉE à la hausse "
                  f"et ne doit pas servir de base à une recommandation.")
    return (f"{j['name']} | Club enregistré : {j['club']} ({j['ligue']}) | "
            f"Club réel cette saison : {j['club_reel']} | {j['nationalite']} | "
            f"{j['position']} ({j['sub_position']}) | {j['age']:.1f} ans\n"
            f"Contrat restant : {j['annees_contrat']:.1f} ans | Sélections : {j['selections']}\n"
            f"Saison : {j['minutes_n']:.0f} minutes, {j['buts_n']:.0f} buts, "
            f"{j['passes_n']:.0f} passes décisives\n"
            f"Valeur réelle : {j['valeur_M']:.1f} M€ | Valeur estimée : {j['valeur_estimee_M']:.1f} M€ "
            f"| Ratio : {j['ratio']:.2f}{alerte}\n"
            f"Historique de valeur (M€) :\n{histo.to_string(index=False)}")


def simuler_amortissement(prix_achat_M: float, duree_contrat_ans: int,
                          valeur_estimee_M: float) -> str:
    """Simule l'amortissement linéaire d'un transfert (norme IAS 38).
    Pour chaque année du contrat : charge d'amortissement, valeur nette comptable (VNC)
    et plus-value latente si le joueur était revendu à sa valeur estimée par le modèle."""
    duree = max(int(duree_contrat_ans), 1)
    charge = prix_achat_M / duree
    lignes = [f"Prix d'achat : {prix_achat_M:.1f} M€ | Contrat : {duree} ans | "
              f"Charge annuelle : {charge:.2f} M€",
              "Année | VNC fin d'année (M€) | Plus-value latente si revente à la valeur estimée (M€)"]
    for annee in range(1, duree + 1):
        vnc = prix_achat_M - charge * annee
        lignes.append(f"{annee} | {vnc:.1f} | {valeur_estimee_M - vnc:+.1f}")
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# Module 6 — Agents IA (Gemini)
# ---------------------------------------------------------------------------
MODELE = "gemini-3.5-flash"
MODELE_SECOURS = "gemini-flash-lite-latest"
MAX_COMITES_PAR_SESSION = 3

SCHEMA = """
BASE DE DONNÉES (DuckDB) — montants en euros
- players(player_id, name, position, sub_position, date_of_birth, country_of_citizenship,
  current_club_name, current_club_domestic_competition_id, market_value_in_eur,
  contract_expiration_date, international_caps)
- transfers(player_id, player_name, transfer_date, transfer_season, from_club_name,
  to_club_name, transfer_fee, market_value_in_eur) — transfer_season au format '25/26'
- clubs(club_id, name, domestic_competition_id, total_market_value)
- player_valuations(player_id, date, market_value_in_eur)
- appearances(player_id, game_id, date, competition_id, goals, assists, minutes_played)
- games(game_id, competition_id, season, date, home_club_id, away_club_id,
  home_club_goals, away_club_goals)
Codes des championnats : GB1 Angleterre, ES1 Espagne, IT1 Italie, L1 Allemagne,
FR1 France, PO1 Portugal, NL1 Pays-Bas, BE1 Belgique.
"""


def instructions_base():
    return f"""Date du jour : {date.today():%d/%m/%Y}. Toute date antérieure est passée.
Tu es un analyste financier spécialisé dans le football, au service d'un directeur sportif.

RÈGLES ABSOLUES
1. N'invente JAMAIS de chiffre. Chaque chiffre doit venir d'un outil.
2. Utilise toujours les outils pour répondre. Si aucun outil ne permet de répondre, dis-le clairement.
3. Réponds en français, de façon structurée : réponse directe, chiffres clés, puis limites.
4. Signale les limites : joueurs prêtés, championnats non couverts, estimations du modèle
   qui ne mesurent ni le potentiel ni les blessures.
5. N'utilise jamais de LaTeX : écris les montants en texte simple (par exemple « 15 M€ »).

OUTILS
- chercher_sous_evalues : recherche de joueurs sous-évalués (modèle de machine learning).
- fiche_joueur : profil complet d'un joueur.
- executer_sql : toute autre question sur la base (transferts, clubs, valeurs, matchs).
- simuler_amortissement : amortissement comptable d'un transfert (IAS 38).
{SCHEMA}"""


ROLE_SCOUT = """
TON RÔLE : SCOUT.
Propose 3 joueurs maximum correspondant au besoin du club.
Pour chacun : club, championnat, âge, poste, minutes jouées, valeur réelle, valeur estimée,
ratio et contrat restant, uniquement à partir des outils.
Utilise fiche_joueur pour chaque candidat et écarte tout joueur signalé comme prêt probable.
"""

ROLE_ANALYSTE = """
TON RÔLE : DIRECTEUR FINANCIER.
Pour chaque joueur proposé par le scout :
1. Hypothèse de prix d'achat = valeur réelle actuelle. Contrat proposé : 5 ans.
2. Utilise simuler_amortissement (prix, 5 ans, valeur estimée).
3. Évalue les risques : âge (la valeur progresse jusqu'à environ 23 ans et se déprécie
   fortement après 29 ans), contrat restant (moins de 2 ans = levier de négociation pour
   l'acheteur), ratio du modèle, temps de jeu.
4. Classe les joueurs du plus intéressant au moins intéressant, en justifiant.
"""

ROLE_REDACTEUR = """
Tu es le rédacteur du comité de recrutement d'un club de football.
Rédige en français une NOTE DE RECRUTEMENT professionnelle, en Markdown, avec :
1. Un titre et la date du jour
2. Un résumé exécutif (3 lignes maximum) avec la recommandation finale
3. Un tableau comparatif des joueurs
4. Une analyse par joueur (atouts, risques, impact financier)
5. Les risques et limites de l'analyse
6. Les prochaines étapes (analyse vidéo, visite médicale, négociation)
RÈGLES : n'ajoute AUCUN chiffre absent des documents fournis. N'utilise jamais de LaTeX.
"""


@st.cache_resource
def client_gemini():
    try:
        cle = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None
    from google import genai
    from google.genai import types
    return genai.Client(api_key=cle, http_options=types.HttpOptions(timeout=60000))


def agent(nom, role, consigne, outils, journal, modele=MODELE):
    """Un agent générique : un rôle, une consigne, une liste d'outils.
    Les appels d'outils sont ajoutés au journal pour être affichés."""
    from google.genai import types
    reglages = dict(system_instruction=instructions_base() + role, temperature=0.2)
    if outils:
        reglages["tools"] = outils
    try:
        r = client_gemini().models.generate_content(
            model=modele, contents=consigne, config=types.GenerateContentConfig(**reglages))
    except Exception as e:
        if modele != MODELE_SECOURS:
            journal.append(f"⚠️ {modele} indisponible, passage au modèle de secours")
            return agent(nom, role, consigne, outils, journal, MODELE_SECOURS)
        return f"Erreur : {e}"
    for etape in (r.automatic_function_calling_history or []):
        for part in (etape.parts or []):
            if part.function_call:
                journal.append(f"🔧 {nom} appelle : {part.function_call.name}")
    return r.text or ""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_accueil():
    st.title("⚽💰 Football Finance Intelligence")
    st.subheader("Analyser les joueurs de football comme des actifs financiers")
    st.markdown(
        "Ce projet applique les outils de l'**analyse financière** (croissance, décomposition "
        "des écarts, rendement, risque, amortissement, valorisation) au marché des transferts, "
        "puis utilise des **agents IA** pour produire des notes de recrutement.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Record des dépenses observées", "12,1 Md€", "saison 2025/26")
    c2.metric("Part de la Premier League (Big 5)", "52 %", "2025/26")
    c3.metric("Précision du modèle (R²)", "0,88", "erreur médiane 33 %")
    c4.metric("Effectif type du Maroc", "609 M€", "1ʳᵉ sélection africaine")

    st.markdown("### 🔑 Les résultats clés")
    st.markdown("""
1. **Un marché tiré par les prix et de plus en plus concentré** : la Premier League est la seule
   ligue importatrice nette ; le marché fonctionne comme s'il ne comptait plus que 3 ligues.
2. **Un joueur est un actif qui se déprécie** : forte appréciation jusqu'à 23 ans, puis dépréciation
   accélérée après 29 ans. Un portefeuille de joueurs à plus de 10 M€ acheté fin 2020 perd 46 % en 5 ans.
3. **L'argent explique 57 % des points** : Lens, Union Berlin ou Brest font bien mieux que leur budget.
4. **Un modèle XGBoost explique 88 % des écarts de valeur**, avec SHAP pour comprendre ses décisions.
5. **Le Maroc est devenu la sélection africaine la plus valorisée**, portée par sa diaspora,
   et le Mondial 2022 a révélé des talents sous-évalués.
6. **Un comité de recrutement multi-agents** (scout, analyste financier, rédacteur) produit des notes
   de recrutement chiffrées, sans jamais inventer de chiffres.
""")
    st.markdown("### 🧭 Navigation")
    st.markdown("Utilise le menu à gauche pour explorer chaque module.")
    st.info("**Démarche :** chaque résultat a été vérifié (biais de survie, biais de sélection, "
            "joueurs prêtés, championnats non couverts). Les limites sont indiquées dans chaque page.")
    st.markdown(f"📂 Code source et notebooks : [{URL_GITHUB}]({URL_GITHUB})")
    st.caption("Données : Transfermarkt, via le projet open source transfermarkt-datasets.")


def page_marche():
    st.title("🌍 Module 1 — Le marché des transferts")
    saisons, big5, parts, hhi = donnees_marche()

    derniere = saisons.iloc[-1]
    covid = saisons[saisons["saison"] == "20/21"]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Dépenses observées {derniere['saison']}", f"{derniere['depenses_M']:,.0f} M€".replace(",", " "))
    if not covid.empty:
        c2.metric("Choc du Covid (2020/21)", f"{covid['variation_pct'].iloc[0]:.0f} %")
    pl = parts.iloc[-1].get("Premier League", np.nan)
    c3.metric("Part de la Premier League (Big 5)", f"{pl:.0f} %")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=saisons["saison"], y=saisons["depenses_M"], name="Dépenses (M€)"))
    fig.add_trace(go.Scatter(x=saisons["saison"], y=saisons["nb_transferts"], name="Nombre de transferts",
                             mode="lines+markers", yaxis="y2"))
    fig.update_layout(title="Évolution du marché des transferts", hovermode="x unified",
                      yaxis=dict(title="Dépenses (M€)"),
                      yaxis2=dict(title="Nombre de transferts", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")
    st.warning("**Biais de survie :** la base enregistre surtout les transferts des joueurs encore suivis "
               "aujourd'hui. Les saisons anciennes sont sous-représentées : la croissance réelle est "
               "inférieure à la croissance observée.")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=saisons["saison"], y=saisons["effet_volume"], name="Effet volume"))
    fig.add_trace(go.Bar(x=saisons["saison"], y=saisons["effet_prix"], name="Effet prix"))
    fig.update_layout(barmode="relative", title="D'où vient la variation des dépenses ? (M€)",
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Dépenses = nombre de transferts × prix moyen. L'effet volume mesure l'impact du nombre "
               "d'achats, l'effet prix celui du prix moyen : c'est l'analyse des écarts du contrôle de gestion.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.area(parts, title="Parts de marché au sein du Big 5 (%)",
                      color_discrete_map=COULEURS_LIGUES)
        fig.update_layout(hovermode="x unified", yaxis_title="%", legend_title="")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.line(x=hhi.index, y=hhi.values, markers=True, title="Concentration du marché (indice HHI)",
                      labels={"x": "Saison", "y": "HHI"})
        fig.add_hline(y=2000, line_dash="dash", annotation_text="Répartition égale (2 000)")
        st.plotly_chart(fig, width="stretch")
    st.caption(f"Nombre de ligues « équivalentes » (10 000 ÷ HHI) : {10000 / hhi.iloc[-1]:.1f} "
               f"en {hhi.index[-1]}, contre {10000 / hhi.max():.1f} au plus fort de la concentration.")


def page_actif():
    st.title("📈 Module 2 — Le joueur vu comme un actif financier")
    ra = donnees_actif()

    ra["sens"] = np.where(ra["rendement_median"] >= 0, "Appréciation", "Dépréciation")
    fig = px.bar(ra, x="age", y="rendement_median", color="sens",
                 color_discrete_map={"Appréciation": "#00CC96", "Dépréciation": "#EF553B"},
                 hover_data={"pct_hausse": ":.0f", "nb": True, "sens": False},
                 title="Rendement annuel médian de la valeur d'un joueur selon son âge (2016-2025)",
                 labels={"age": "Âge", "rendement_median": "Rendement médian (%)",
                         "pct_hausse": "% de joueurs en hausse", "sens": ""})
    st.plotly_chart(fig, width="stretch")
    st.caption("Point de bascule vers 23-24 ans : à partir de 24 ans, moins d'un joueur sur deux "
               "prend de la valeur. La dépréciation s'accélère après 29 ans.")

    ra["phase"] = pd.cut(ra["age"], bins=[16, 23, 28, 36],
                         labels=["Jeunes (17-23)", "Plateau (24-28)", "Seniors (29-36)"]).astype(str)
    fig = px.scatter(ra, x="volatilite", y="rendement_moyen_log", text="age", color="phase", size="nb",
                     title="Rendement / risque selon l'âge (rendements logarithmiques)",
                     labels={"volatilite": "Risque : volatilité annuelle (%)",
                             "rendement_moyen_log": "Rendement annuel moyen (%)", "phase": "Phase"})
    fig.update_traces(textposition="top center")
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, width="stretch")
    st.info("**Lecture financière :** relation rendement / risque quasi linéaire. Les jeunes joueurs offrent "
            "le rendement le plus élevé, avec le risque le plus élevé. Les joueurs de plus de 29 ans sont des "
            "**actifs dominés** au sens de Markowitz : leur achat se justifie par l'apport sportif.")
    st.markdown("**Backtest (notebook du Module 2) :** un portefeuille de joueurs valant au moins 10 M€ "
                "acheté fin 2020 perd **46 %** en 5 ans. Seuls les joueurs de 21 ans et moins sont rentables (+41 %).")


def page_clubs():
    st.title("🏟️ Module 3 — Rentabilité des transferts et Moneyball")
    par_age, clubs, top_flop, synthese = donnees_reventes()

    c1, c2, c3 = st.columns(3)
    c1.metric("Cycles achat → revente", f"{synthese['nb_cycles']:,}".replace(",", " "))
    c2.metric("Revendus plus cher qu'achetés", f"{synthese['pct_plus_cher']:.0f} %")
    c3.metric("En plus-value comptable", f"{synthese['pct_pv_comptable']:.0f} %")
    st.caption("Grâce à l'amortissement, un joueur revendu à perte peut générer une plus-value comptable.")

    fig = go.Figure([
        go.Bar(x=par_age["tranche"], y=par_age["pct_revendu_plus_cher"], name="% revendus plus cher"),
        go.Bar(x=par_age["tranche"], y=par_age["pct_pv_comptable"], name="% en plus-value comptable"),
    ])
    fig.update_layout(barmode="group", title="Rentabilité de la revente selon l'âge à l'achat",
                      xaxis_title="Âge à l'achat", yaxis_title="%")
    st.plotly_chart(fig, width="stretch")

    ct = clubs[clubs["nb_cycles"] >= 15]
    classement = pd.concat([ct.nlargest(15, "plus_value_M"), ct.nsmallest(10, "plus_value_M")]) \
        .drop_duplicates("club_id").sort_values("plus_value_M")
    fig = px.bar(classement, x="plus_value_M", y="club", orientation="h", color="roi_global",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                 hover_data={"nb_cycles": True, "investi_M": ":.0f", "encaisse_M": ":.0f"},
                 title="Clubs « traders » : plus-value totale à la revente (top 15 et flop 10, M€)",
                 labels={"plus_value_M": "Plus-value totale (M€)", "club": "", "roi_global": "ROI (%)"})
    fig.update_layout(height=700)
    st.plotly_chart(fig, width="stretch")

    mb, a, b, r2 = donnees_moneyball()
    fig = px.scatter(mb, x="valeur_rel", y="ppm", color="ligue", log_x=True, opacity=0.6,
                     hover_name="club", color_discrete_map=COULEURS_LIGUES,
                     hover_data={"saison_txt": True, "points": True, "surperf_pts": ":.1f", "ligue": False},
                     title=f"Moneyball : valeur relative de l'effectif vs points par match (R² = {r2:.2f})",
                     labels={"valeur_rel": "Valeur relative de l'effectif (1 = moyenne de la ligue)",
                             "ppm": "Points par match", "saison_txt": "Saison", "ligue": "Ligue"})
    x = np.linspace(mb["valeur_rel"].min(), mb["valeur_rel"].max(), 100)
    fig.add_trace(go.Scatter(x=x, y=a + b * np.log(x), mode="lines",
                             line=dict(color="black", dash="dash"), name="Points prédits par le budget"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"L'argent explique {r2:.0%} des écarts de points. Doubler sa valeur relative rapporte "
               f"environ {b * np.log(2) * 38:.0f} points par saison.")

    eff = (mb.groupby("club_id").agg(club=("club", "first"), nb_saisons=("saison", "count"),
                                     surperf_moy=("surperf_pts", "mean")).reset_index())
    matrice = eff.merge(clubs[["club_id", "nb_cycles", "roi_global"]], on="club_id")
    matrice = matrice[(matrice["nb_saisons"] >= 5) & (matrice["nb_cycles"] >= 10)].copy()
    conditions = [(matrice["surperf_moy"] >= 0) & (matrice["roi_global"] >= 0),
                  (matrice["surperf_moy"] >= 0),
                  (matrice["roi_global"] >= 0)]
    matrice["profil"] = np.select(conditions, ["⭐ Modèle complet", "🏆 Champions qui dépensent",
                                               "🏪 Vendeurs"], default="💸 Gaspilleurs")
    fig = px.scatter(matrice, x="roi_global", y="surperf_moy", color="profil", size="nb_cycles",
                     hover_name="club",
                     title="Matrice des clubs : rentabilité des reventes × efficacité sportive",
                     labels={"roi_global": "ROI global des reventes (%)",
                             "surperf_moy": "Surperformance Moneyball (points/saison)", "profil": "Profil"})
    fig.add_vline(x=0, line_dash="dash")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")
    st.caption("Limite : seules les reventes avec indemnité sont comptées (les départs libres sont exclus).")


def page_valorisation():
    st.title("🤖 Module 4 — Le modèle de valorisation : trouver les joueurs sous-évalués")
    st.markdown("Un modèle **XGBoost** estime la valeur de chaque joueur (R² = 0,88 sur les championnats "
                "couverts). **Ratio = valeur estimée ÷ valeur réelle** : au-dessus de 1, le joueur est "
                "potentiellement sous-évalué.")
    valo = donnees_valo()

    c1, c2, c3, c4 = st.columns(4)
    postes = c1.multiselect("Poste", ["Attack", "Midfield", "Defender", "Goalkeeper"],
                            default=["Attack", "Midfield", "Defender", "Goalkeeper"])
    age_max = c2.slider("Âge maximum", 17, 38, 24)
    valeur = c3.slider("Valeur réelle (M€)", 1, 100, (1, 20))
    minutes_min = c4.slider("Minutes jouées minimum (2025/26)", 0, 3000, 1500, step=100)
    c5, c6, c7 = st.columns(3)
    ligues = sorted(valo["ligue"].dropna().unique())
    ligue = c5.selectbox("Championnat", ["Tous"] + ligues)
    nations = sorted(valo["nationalite"].dropna().unique())
    nation = c6.selectbox("Nationalité", ["Toutes"] + nations)
    inclure_prets = c7.checkbox("Inclure les prêts probables (estimation biaisée)", value=False)

    df = valo[valo["position"].isin(postes) & (valo["age"] <= age_max)
              & valo["valeur_M"].between(*valeur) & (valo["minutes_n"] >= minutes_min)]
    if ligue != "Tous":
        df = df[df["ligue"] == ligue]
    if nation != "Toutes":
        df = df[df["nationalite"] == nation]
    if not inclure_prets:
        df = df[~df["pret_probable"]]

    st.markdown(f"**{len(df)} joueurs** correspondent aux critères.")
    colonnes = ["name", "club", "club_reel", "ligue", "nationalite", "position", "age",
                "minutes_n", "valeur_M", "valeur_estimee_M", "ratio"]
    st.dataframe(df.nlargest(25, "ratio")[colonnes].round(2), width="stretch", hide_index=True)

    if not df.empty:
        df = df.assign(statut=np.select([df["ratio"] >= 1.5, df["ratio"] <= 0.67],
                                        ["Sous-évalué", "Surévalué"], default="Juste prix"))
        fig = px.scatter(df, x="valeur_M", y="valeur_estimee_M", color="statut", log_x=True, log_y=True,
                         hover_name="name", hover_data={"club": True, "age": ":.0f", "ratio": ":.2f"},
                         color_discrete_map={"Sous-évalué": "#00CC96", "Juste prix": "#AAAAAA",
                                             "Surévalué": "#EF553B"},
                         title="Valeur réelle vs valeur estimée par le modèle",
                         labels={"valeur_M": "Valeur réelle (M€)", "valeur_estimee_M": "Valeur estimée (M€)",
                                 "statut": ""})
        bornes = [df["valeur_M"].min(), df["valeur_M"].max()]
        fig.add_trace(go.Scatter(x=bornes, y=bornes, mode="lines", line=dict(dash="dash", color="black"),
                                 name="Juste prix"))
        st.plotly_chart(fig, width="stretch")

    st.markdown("### 🔎 Fiche joueur")
    noms = valo.sort_values("valeur_M", ascending=False)["name"].tolist()
    defaut = noms.index("Achraf Hakimi") if "Achraf Hakimi" in noms else 0
    nom = st.selectbox("Choisir un joueur", noms, index=defaut)
    j = valo[valo["name"] == nom].iloc[0]
    if j["pret_probable"]:
        st.error(f"⚠️ Prêt probable : enregistré à {j['club']}, mais a joué la saison à {j['club_reel']}. "
                 "Son estimation est biaisée à la hausse.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valeur réelle", f"{j['valeur_M']:.1f} M€")
    m2.metric("Valeur estimée", f"{j['valeur_estimee_M']:.1f} M€", f"ratio {j['ratio']:.2f}")
    m3.metric("Âge", f"{j['age']:.1f} ans")
    m4.metric("Contrat restant", f"{j['annees_contrat']:.1f} ans")
    st.markdown(f"**{j['club']}** ({j['ligue']}) · {j['position']} ({j['sub_position']}) · "
                f"{j['nationalite']} · {j['selections']:.0f} sélections · "
                f"{j['minutes_n']:.0f} minutes, {j['buts_n']:.0f} buts, {j['passes_n']:.0f} passes décisives")
    histo = historique_valeur(j["player_id"])
    fig = px.line(histo, x="annee", y="valeur_M", markers=True, title=f"Historique de valeur — {nom}",
                  labels={"annee": "Année", "valeur_M": "Valeur (M€)"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Le modèle ne mesure ni le potentiel, ni les blessures, ni le contexte : c'est un outil de "
               "présélection, pas de décision.")


def page_maroc():
    st.title("🇲🇦 Module 5 — Focus Maroc : la valeur économique des Lions de l'Atlas")
    afrique, vivier = donnees_maroc()

    internationaux = vivier[vivier["selections"] >= 1]
    top26 = internationaux.nlargest(26, "valeur_M")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Joueurs marocains actifs", len(vivier))
    c2.metric("Valeur du vivier", f"{vivier['valeur_M'].sum():,.0f} M€".replace(",", " "))
    c3.metric("Effectif type (26 internationaux)", f"{top26['valeur_M'].sum():,.0f} M€".replace(",", " "))
    c4.metric("Effectif type né à l'étranger",
              f"{(top26['naissance'] == 'Né à l' + chr(39) + 'étranger').sum()} / 26")

    couleurs = {"Maroc": ROUGE_MAROC, "Sénégal": "#00853F", "Égypte": "#000000",
                "Côte d'Ivoire": "#F77F00", "Nigeria": "#1E90FF", "Algérie": "#9B59B6"}
    fig = px.line(afrique, x="annee", y="valeur_top26_M", color="pays", markers=True,
                  color_discrete_map=couleurs,
                  title="Valeur de l'effectif type (26 internationaux les plus chers), 2012-2026",
                  labels={"annee": "Année", "valeur_top26_M": "Valeur (M€)", "pays": "Pays"})
    fig.update_traces(line=dict(width=1.5), opacity=0.6)
    fig.update_traces(selector=dict(name="Maroc"), line=dict(width=4), opacity=1)
    for x, texte in [(2022, "Mondial 2022"), (2024, "CAN 2023"), (2025, "CAN 2025 au Maroc")]:
        fig.add_vline(x=x, line_dash="dot", line_color="grey", annotation_text=texte,
                      annotation_position="top left")
    fig.update_layout(hovermode="x unified", height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption("4ᵉ sélection africaine en 2018, 1ʳᵉ depuis 2024. L'année 2026 est partielle (estimations jusqu'en juin).")

    fig = px.bar(top26.sort_values("valeur_M"), x="valeur_M", y="name", orientation="h", color="naissance",
                 color_discrete_map={"Né au Maroc": VERT_MAROC, "Né à l'étranger": ROUGE_MAROC},
                 hover_data={"club": True, "pays_naissance": True, "age": ":.1f", "selections": True},
                 title="L'effectif type du Maroc : les 26 internationaux les plus chers",
                 labels={"valeur_M": "Valeur marchande (M€)", "name": "", "naissance": ""})
    fig.update_layout(height=750)
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        par_champ = (vivier.groupby("championnat").agg(joueurs=("name", "count"),
                                                        valeur_M=("valeur_M", "sum"))
                     .sort_values("valeur_M", ascending=False).head(10).round(1))
        st.markdown("**Où jouent les Marocains ?**")
        st.dataframe(par_champ, width="stretch")
    with col2:
        pays = internationaux["pays_naissance"].value_counts().head(8)
        fig = px.pie(values=pays.values, names=pays.index, title="Pays de naissance des internationaux",
                     hole=0.4)
        st.plotly_chart(fig, width="stretch")
    st.info("**Lecture financière :** la formation de la majorité des internationaux a été financée par des "
            "clubs européens. Le Maroc bénéficie de leur talent sans en supporter le coût : une stratégie "
            "de sourcing efficace. Le Mondial 2022 (notebook du Module 5) a surtout revalorisé des joueurs "
            "peu exposés (Ounahi, Amrabat), illustrant l'efficience des marchés.")


def page_comite():
    st.title("🧠 Module 6 — Le comité de recrutement IA")
    st.markdown("Trois agents spécialisés travaillent en chaîne : un **scout** 🕵️ trouve les profils, un "
                "**analyste financier** 💰 simule l'amortissement et évalue les risques, un **rédacteur** ✍️ "
                "produit la note de recrutement. **L'IA ne calcule jamais** : tous les chiffres viennent "
                "d'outils Python.")

    if client_gemini() is None:
        st.warning("La clé API Gemini n'est pas configurée sur cette application. "
                   "Le comité est consultable dans le notebook du Module 6 sur GitHub.")
        st.markdown(f"[Voir le notebook]({URL_GITHUB})")
        return

    st.session_state.setdefault("nb_comites", 0)
    restants = MAX_COMITES_PAR_SESSION - st.session_state["nb_comites"]
    besoin = st.text_area(
        "Le besoin du club",
        "Le club cherche un défenseur de 23 ans maximum, titulaire dans un championnat européen, "
        "pour un budget maximum de 15 M€. Priorité au potentiel de revente.", height=100)
    st.caption(f"Analyses restantes pour cette session : {restants} (limite pour préserver le quota gratuit).")

    if st.button("🚀 Lancer le comité", disabled=restants <= 0, type="primary"):
        st.session_state["nb_comites"] += 1
        journal = []
        with st.status("Le comité travaille… (1 à 2 minutes)", expanded=True) as statut:
            st.write("🕵️ Le scout cherche des profils…")
            shortlist = agent("Scout", ROLE_SCOUT, besoin,
                              [chercher_sous_evalues, fiche_joueur, executer_sql], journal)
            st.write("💰 L'analyste financier évalue les profils…")
            analyse = agent("Analyste", ROLE_ANALYSTE,
                            f"BESOIN DU CLUB :\n{besoin}\n\nPROPOSITIONS DU SCOUT :\n{shortlist}",
                            [fiche_joueur, simuler_amortissement, executer_sql], journal)
            st.write("✍️ Le rédacteur rédige la note…")
            note = agent("Rédacteur", ROLE_REDACTEUR,
                         f"BESOIN DU CLUB :\n{besoin}\n\nSHORTLIST DU SCOUT :\n{shortlist}"
                         f"\n\nANALYSE FINANCIÈRE :\n{analyse}", [], journal)
            for ligne in journal:
                st.write(ligne)
            statut.update(label="✅ Note de recrutement prête", state="complete", expanded=False)
        st.session_state["derniere_note"] = note

    if "derniere_note" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["derniere_note"])
        st.download_button("📄 Télécharger la note (Markdown)", st.session_state["derniere_note"],
                           file_name="note_recrutement.md")
        st.caption("Note générée par IA à partir de données publiques. Une relecture humaine reste indispensable.")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Accueil": page_accueil,
    "🌍 Marché des transferts": page_marche,
    "📈 Le joueur comme actif": page_actif,
    "🏟️ Clubs et Moneyball": page_clubs,
    "🤖 Joueurs sous-évalués": page_valorisation,
    "🇲🇦 Focus Maroc": page_maroc,
    "🧠 Comité de recrutement IA": page_comite,
}

st.sidebar.title("⚽💰 Football Finance Intelligence")
choix = st.sidebar.radio("Navigation", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Projet de portfolio — Ingénierie financière × data × IA")
st.sidebar.caption(f"[Code sur GitHub]({URL_GITHUB})")
PAGES[choix]()
