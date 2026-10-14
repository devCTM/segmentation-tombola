# -*- coding: utf-8 -*-
"""
Segmentation marketing des participants à une tombola — K-Means
=================================================================

Application Streamlit permettant de :
  1. Charger un fichier Excel de transactions/participants
  2. Agréger les données au niveau "participant" (fréquence, montant,
     récence, diversité des destinations, profil socio-démographique...)
  3. Préparer les variables (standardisation + encodage one-hot)
  4. Choisir le nombre de segments k (méthode du coude + score de silhouette)
  5. Exécuter K-Means
  6. Visualiser et interpréter les segments d'un point de vue marketing
  7. Exporter les résultats (Excel / CSV)
  8. Mesurer la récurrence d'achat par email payeur (colonne PayerEmail) et la
     rattacher aux participants pour profiler les segments

Deux pages (sélecteur dans la barre latérale) :
  - « Segmentation des participants » (K-Means, récurrence)
  - « Retours clients » (analyse des colonnes « À améliorer » / « Souhait »)

Lancement :
    pip install -r requirements.txt
    streamlit run app.py

Le fichier Excel des participants se charge via le bouton d'upload de la
barre latérale.
"""

import hashlib
import html
import hmac
import io
import unicodedata
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# Configuration générale
# =============================================================================
st.set_page_config(
    page_title="Segmentation Tombola — K-Means",
    page_icon="🎯",
    layout="wide",
)

# =============================================================================
# 🔐 Authentification (porte d'accès avant toute l'application)
# =============================================================================
# Les mots de passe ne sont JAMAIS stockés en clair : seulement un sel + un hash
# PBKDF2-HMAC-SHA256 (200 000 itérations) par utilisateur.
# Pour changer les comptes sans toucher au code, définir dans .streamlit/secrets.toml :
#   [auth_users]
#   admin = ["<sel_hex>", "<hash_hex>"]
AUTH_ITERATIONS = 200_000
AUTH_MAX_ATTEMPTS = 5          # échecs avant verrouillage
AUTH_LOCK_SECONDS = 120        # durée du verrouillage
AUTH_SESSION_MINUTES = 120     # déconnexion automatique après inactivité

_DEFAULT_AUTH_USERS = {
    "admin": ("a4671b4f0aa017f7ee4ab849d3ac9900", "7ed5b2da6708223e2e8b8e2792a7cee5ca3e9542b9c34502d02f294e2a6be9e8"),
    "analyste": ("f70eb9426d59dfa25262d2ead2b80d20", "897f71d8c52096fdbfdb33048554e053c75835626b6c1425c18b329ee8da34e7"),
    "marketing": ("01e48e5b5e53fedf72670382b88200e8", "c52353b2de9cda1ac13e76950c9683eed918e502e9cf9a7f5b398a3d420b8328"),
    # "direction": ("426614cb3c182cb639b813c98f77a65a", "71f6e7dfe476b68b4370187240da3165e072c766622cf30b0d02afe5aff6db92"),
}


def _auth_users() -> dict:
    try:
        if "auth_users" in st.secrets:
            return {u: tuple(v) for u, v in dict(st.secrets["auth_users"]).items()}
    except Exception:
        pass
    return _DEFAULT_AUTH_USERS


@st.cache_resource
def _auth_failures() -> dict:
    """Compteur d'échecs partagé par tous les visiteurs du serveur (résiste au F5)."""
    return {}


def _check_credentials(username: str, password: str) -> bool:
    users = _auth_users()
    salt_hex, hash_hex = users.get(username.strip().lower(), ("00" * 16, "00" * 32))
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                    bytes.fromhex(salt_hex), AUTH_ITERATIONS)
    ok = hmac.compare_digest(candidate.hex(), hash_hex)
    return ok and username.strip().lower() in users


_LOGIN_CSS = """
<style>
:root {color-scheme: light;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"], [data-testid="stToolbar"], footer {display: none !important;}
.stApp {background: #faf9f5 !important; color: #141413;}
.block-container {padding: 11vh 1rem 2rem !important; max-width: 460px !important;}
[data-testid="InputInstructions"] {display: none !important;}
[data-testid="stForm"] [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]),
[data-testid="stForm"] [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) div,
[data-testid="stFormSubmitButton"] > button {width: 100% !important;}

.login-brand {text-align: center; margin-bottom: 2rem; animation: fadeUp .6s ease both;}
.login-brand .mark {
    width: 72px; height: 72px; margin: 0 auto 1.2rem; border-radius: 20px;
    display: flex; align-items: center; justify-content: center; font-size: 36px;
    background: linear-gradient(135deg, #5bbcf0, #2b97dc);
    box-shadow: 0 10px 28px rgba(56,169,232,.35);
}
.login-brand h1 {
    white-space: nowrap; font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-weight: 400; font-size: 2.35rem; letter-spacing: -.015em; line-height: 1.15;
    color: #141413; margin: 0 0 .6rem; padding: 0;
}
.login-brand p {color: #73726c; font-size: 1rem; margin: 0;}

[data-testid="stForm"] {
    background: #ffffff; border: 1px solid #e8e6dc; border-radius: 18px;
    padding: 30px 28px 22px; box-shadow: 0 1px 2px rgba(20,20,19,.04), 0 12px 32px rgba(20,20,19,.06);
    animation: fadeUp .7s .08s ease both;
}
[data-testid="stForm"] label p {color: #3d3d3a !important; font-size: .9rem; font-weight: 500;}
[data-testid="stForm"] [data-testid="stTextInputRootElement"],
[data-testid="stForm"] [data-baseweb="input"], [data-testid="stForm"] [data-baseweb="base-input"] {
    background-color: #ffffff !important; border-radius: 10px !important;
}
[data-testid="stForm"] [data-testid="stTextInputRootElement"] {
    border: 1px solid #d9d6ca !important; transition: border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stForm"] [data-testid="stTextInputRootElement"]:focus-within {
    border-color: #38a9e8 !important; box-shadow: 0 0 0 3px rgba(56,169,232,.22) !important;
}
[data-testid="stForm"] input {
    color: #141413 !important; background: transparent !important;
    -webkit-text-fill-color: #141413 !important; padding: .7rem .85rem !important;
}
[data-testid="stForm"] input::placeholder {color: #a8a69c !important; -webkit-text-fill-color: #a8a69c !important;}
[data-testid="stForm"] [data-baseweb="input"] button svg {fill: #73726c !important;}

[data-testid="stForm"] button[kind="secondaryFormSubmit"], [data-testid="stForm"] button[kind="primaryFormSubmit"] {
    width: 100%; border: 0 !important; border-radius: 10px; padding: .7rem 0; margin-top: .4rem;
    background: #1b94d8 !important; color: #ffffff !important; font-weight: 600; font-size: .98rem;
    transition: background .18s ease, transform .12s ease;
}
[data-testid="stForm"] button[kind$="FormSubmit"] p {color: #ffffff !important;}
[data-testid="stForm"] button[kind$="FormSubmit"]:hover {background: #1480ba !important;}
[data-testid="stForm"] button[kind$="FormSubmit"]:active {transform: scale(.985);}

[data-testid="stAlert"] {border-radius: 12px;}
.login-foot {text-align: center; color: #a8a69c; font-size: .82rem; margin-top: 1.4rem;}
@keyframes fadeUp {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: none;}}
</style>
"""


_SIDEBAR_USER_CSS = """
<style>
.user-card {display: flex; align-items: center; gap: 12px; padding: 12px 14px; margin: .4rem 0 .6rem;
    background: #ffffff; border: 1px solid #e8e6dc; border-radius: 14px;
    box-shadow: 0 1px 2px rgba(20,20,19,.04), 0 6px 16px rgba(20,20,19,.05);}
.user-card .avatar {flex: none; width: 42px; height: 42px; border-radius: 50%; color: #ffffff;
    display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.1rem;
    background: linear-gradient(135deg, #5bbcf0, #2b97dc); box-shadow: 0 4px 12px rgba(56,169,232,.35);}
.user-card .who {min-width: 0; line-height: 1.3;}
.user-card .who small {display: flex; align-items: center; gap: 6px; color: #73726c; font-size: .7rem;
    letter-spacing: .06em; text-transform: uppercase;}
.user-card .who .dot {width: 7px; height: 7px; border-radius: 50%; background: #2f9e6d;
    box-shadow: 0 0 0 3px rgba(47,158,109,.18);}
.user-card .who b {display: block; color: #141413; font-size: 1rem; font-weight: 600;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.st-key-logout_btn button {border-radius: 10px; font-weight: 600; color: #c4472d;
    background: #ffffff; border: 1px solid #f0cbc2; transition: background .15s ease, border-color .15s ease;}
.st-key-logout_btn button:hover {background: #fdf1ee; border-color: #e6a999; color: #a93a22;}
.st-key-logout_btn button:active {transform: scale(.985);}
</style>
"""


def require_login() -> None:
    """Affiche l'écran de connexion et arrête l'exécution tant que l'accès n'est pas validé."""
    now = time.time()
    if st.session_state.get("auth_user"):
        if now - st.session_state.get("auth_last_seen", now) > AUTH_SESSION_MINUTES * 60:
            st.session_state.pop("auth_user", None)
            st.session_state["auth_msg"] = "Session expirée, veuillez vous reconnecter."
        else:
            st.session_state["auth_last_seen"] = now
            user = str(st.session_state["auth_user"])
            st.sidebar.markdown(_SIDEBAR_USER_CSS, unsafe_allow_html=True)
            st.sidebar.markdown(
                f'<div class="user-card"><div class="avatar">{html.escape(user[:1].upper())}</div>'
                f'<div class="who"><small><span class="dot"></span>Connecté</small>'
                f"<b>{html.escape(user)}</b></div></div>",
                unsafe_allow_html=True,
            )
            if st.sidebar.button("Se déconnecter", key="logout_btn", icon=":material/logout:",
                                 width="stretch"):
                for k in ("auth_user", "auth_last_seen"):
                    st.session_state.pop(k, None)
                st.rerun()
            return

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="login-brand"><div class="mark">🎯</div>'
        "<h1>Heureux de vous revoir</h1>"
        "<p>Connectez-vous pour accéder à la segmentation des participants.</p></div>",
        unsafe_allow_html=True,
    )
    if st.session_state.get("auth_msg"):
        st.info(st.session_state.pop("auth_msg"))

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Identifiant", placeholder="ex. analyste")
        password = st.text_input("Mot de passe", type="password", placeholder="Votre mot de passe")
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        key = username.strip().lower()
        fails = _auth_failures()
        count, locked_until = fails.get(key, (0, 0.0))
        if locked_until > now:
            st.error(f"🔒 Trop de tentatives. Réessayez dans {int(locked_until - now)} s.")
        elif username and password and _check_credentials(username, password):
            fails.pop(key, None)
            st.session_state["auth_user"] = key
            st.session_state["auth_last_seen"] = now
            st.rerun()
        else:
            count += 1
            if count >= AUTH_MAX_ATTEMPTS:
                fails[key] = (0, now + AUTH_LOCK_SECONDS)
                st.error(f"🔒 Compte verrouillé {AUTH_LOCK_SECONDS // 60} minutes après "
                         f"{AUTH_MAX_ATTEMPTS} échecs.")
            else:
                fails[key] = (count, 0.0)
                time.sleep(0.8)  # ralentit les attaques par force brute
                st.error(f"Identifiant ou mot de passe incorrect "
                         f"({AUTH_MAX_ATTEMPTS - count} essai(s) restant(s)).")

    st.markdown('<div class="login-foot">Accès réservé aux utilisateurs autorisés</div>',
                unsafe_allow_html=True)
    st.stop()


require_login()

# =============================================================================
# 💬 Page « Retours clients » (colonnes « À améliorer » / « Souhait »)
# =============================================================================
FB_THEMES = {
    "Wifi": r"wi-?fi|ويفي|واي ?فاي|internet",
    "Chargeurs / USB": r"charg|usb|type ?c|prise|شاحن|شواحن",
    "Sièges / confort": r"seat|si[eè]ge|confort|comfort|leg ?room|كراسي|كرسي|مقاعد|راحة",
    "Toilettes": r"toilet|restroom|\bwc\b|مرحاض|حمام",
    "Prix": r"prix|price|cheap|pas cher|tarif|ثمن|سعر|أثمنة|اثمنة",
    "Ponctualité": r"retard|late|delay|horaire|ponctu|تأخر|تاخر|موعد",
    "Propreté": r"propre|clean|hygi|نظاف|نظيف",
    "Climatisation": r"clim|air cond|تكييف|مكيف|chaud|froid",
    "Application / réservation": r"appli|\bapp\b|site|booking|r[eé]serv|ticket|billet|تطبيق|حجز",
    "Remboursement / annulation": r"rembours|annul|cancel|refund|modif|[ée]chang",
    "Bagages": r"bagage|luggage|baggage|حقائب|شنطة",
    "Personnel": r"staff|personnel|chauffeur|driver|accueil|[ée]quipe|agent|hôtesse|hotesse|موظف|سائق|طاقم|استقبال",
    "Service": r"service|خدمة|خدمات",
    "Destinations / fréquence": r"ville|destination|fr[eé]quen|more bus|plus de bus|مدن|وجهات|رحلات",
    "Fidélité / promotions": r"fid[eé]lit|loyalty|r[eé]compense|reward|points|r[eé]duction|offre|promo|discount",
}
FB_RIEN = re.compile(
    r"^\W*(rien|ras|rien à signaler|no|non|none|nothing|n/?a|nada|walo+|"
    r"all (is )?(good|great|perfect)|tout (est )?(bien|parfait)|parfait|"
    r"لا|لا شيء|لا شكرا|كل شيء (رائع|جيد)|كل شيئ جيد)\W*$", re.I)
FB_STOP = set("pour dans avec plus vous nous the and that this with have would like more très tout "
              "bien avoir être sont mais cette aussi comme les des une est pas sur par que qui "
              "for you can all add make ainsi chaque entre tous oui non rien nothing everything good bon "
              "merci better aimerais would could very really just على في من إلى عن هذا هذه كل لا".split())
FB_COL = {"ameliorer": "À améliorer", "souhait": "Souhait"}


def fb_clean(x):
    return "" if pd.isna(x) else re.sub(r"_x000D_|\s+", " ", str(x)).strip()


FB_KEYS_AME = ("amelior", "improve", "enhance", "تحسين", "تحسن")
FB_KEYS_SOU = ("souhait", "wish", "suggest", "request", "رغب", "اقتراح", "امنية")


def fb_norm(x) -> str:
    """Nom de colonne normalisé : sans accents, minuscules, sans espaces ni ponctuation."""
    t = unicodedata.normalize("NFKD", str(x))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return "".join(ch for ch in t.lower() if ch.isalnum())


def fb_detect(columns) -> tuple:
    """Détecte automatiquement les colonnes « À améliorer » et « Souhait » d'après leur nom
    (français, anglais ou arabe). À défaut de nom reconnu, repli sur la position : la colonne
    non encore attribuée la plus à gauche."""
    cols = list(columns)
    normed = [fb_norm(c) for c in cols]

    def find(keys, exclude=None):
        for c, n in zip(cols, normed):
            if c != exclude and any(k in n for k in keys):
                return c
        return None

    ame = find(FB_KEYS_AME)
    sou = find(FB_KEYS_SOU, exclude=ame)
    if ame is None:
        ame = next((c for c in cols if c != sou), None)
    if sou is None:
        sou = next((c for c in cols if c != ame), None)
    return ame, sou


@st.cache_data(show_spinner=False)
def fb_colonnes(file_bytes: bytes) -> list:
    return pd.read_excel(io.BytesIO(file_bytes), nrows=0).columns.tolist()


@st.cache_data(show_spinner="Analyse en cours…")
def fb_charger(file_bytes: bytes, col_ame, col_sou):
    df = pd.read_excel(io.BytesIO(file_bytes))
    df = df[[col_ame, col_sou]].copy()
    df.columns = ["ameliorer", "souhait"]
    df = df.apply(lambda s: s.map(fb_clean))
    df["id"] = range(2, len(df) + 2)  # n° de ligne Excel
    for c in FB_COL:
        df[f"{c}_rien"] = df[c].str.match(FB_RIEN)
        for t, p in FB_THEMES.items():
            df[f"{c}|{t}"] = df[c].str.contains(p, flags=re.I, regex=True) & ~df[f"{c}_rien"]
    df["arabe"] = (df.ameliorer + df.souhait).str.contains(r"[؀-ۿ]")
    return df


def render_feedback_page() -> None:
    """Dashboard des retours clients (« À améliorer » / « Souhait »), sur une seule page
    (sections empilées). Page autonome : fichier Excel et filtres propres, dans la barre latérale."""
    st.title("📊 Retours clients – À améliorer / Souhait")

    with st.sidebar:
        st.divider()
        st.header("Retours clients")
        f = st.file_uploader("Fichier Excel des retours", type=["xlsx"], key="fb_file")
    if f is None:
        st.info("Chargez le fichier .xlsx des retours clients dans la barre latérale.")
        return
    file_bytes = f.getvalue()
    colonnes = fb_colonnes(file_bytes)
    if len(colonnes) < 2:
        st.error("Le fichier doit contenir au moins deux colonnes (« À améliorer » et « Souhait »).")
        return
    det_ame, det_sou = fb_detect(colonnes)
    with st.sidebar.expander("🧩 Colonnes détectées", expanded=False):
        st.caption("Détection automatique d'après les noms de colonnes ; corrigez-la si besoin.")
        col_ame = st.selectbox("Colonne « À améliorer »", colonnes, index=colonnes.index(det_ame),
                               key=f"fb_col_ame_{f.name}")
        col_sou = st.selectbox("Colonne « Souhait »", colonnes, index=colonnes.index(det_sou),
                               key=f"fb_col_sou_{f.name}")
    if col_ame == col_sou:
        st.error("Choisissez deux colonnes différentes pour « À améliorer » et « Souhait ».")
        return
    st.caption(f"Colonnes utilisées : « À améliorer » = **{col_ame}** · « Souhait » = **{col_sou}**")
    df = fb_charger(file_bytes, col_ame, col_sou)

    with st.sidebar:
        langue = st.segmented_control("Langue", ["Toutes", "Arabe", "Latin"], default="Toutes",
                                      key="fb_langue") or "Toutes"
        themes_sel = st.pills("Filtrer par thème", list(FB_THEMES), selection_mode="multi",
                              key="fb_themes") or []
        recherche = st.text_input("Mot-clé (texte libre)", key="fb_recherche")

    d = df
    if langue == "Arabe":
        d = d[d.arabe]
    elif langue == "Latin":
        d = d[~d.arabe]
    if themes_sel:
        m = False
        for c in FB_COL:
            for t in themes_sel:
                m = m | d[f"{c}|{t}"]
        d = d[m]
    if recherche:
        m = (d.ameliorer.str.contains(recherche, case=False, regex=False)
             | d.souhait.str.contains(recherche, case=False, regex=False))
        d = d[m]

    def utiles(c):
        s = d[c]
        return s[(s != "") & ~d[f"{c}_rien"]]

    kp = st.columns(5)
    kp[0].metric("Réponses (filtrées)", len(d))
    kp[1].metric("« À améliorer » utiles", len(utiles("ameliorer")))
    kp[2].metric("« Souhait » utiles", len(utiles("souhait")))
    kp[3].metric("Part arabe", f"{d.arabe.mean():.0%}" if len(d) else "–")
    kp[4].metric("Vides (2 colonnes)", int(((d.ameliorer == "") & (d.souhait == "")).sum()))

    if d.empty:
        st.warning("Aucune donnée avec ces filtres.")
        return

    # Une seule page : les sections sont empilées
    CMAP = {"À améliorer": "#d1603d", "Souhait": "#2f7d6d"}

    # ---------------------------------------------------------------- Thèmes
    st.divider()
    st.header("Thèmes")
    rows = [{"Thème": t, "Colonne": FB_COL[c], "Mentions": int(d[f"{c}|{t}"].sum())}
            for c in FB_COL for t in FB_THEMES]
    t = pd.DataFrame(rows)
    piv = t.pivot(index="Thème", columns="Colonne", values="Mentions")
    piv["Total"] = piv.sum(axis=1)
    piv = piv.sort_values("Total")
    ordre = piv.index.tolist()
    style = st.segmented_control("Style du graphique", ["Barres groupées", "Aires"],
                                 default="Barres groupées", key="fb_style") or "Barres groupées"

    if style == "Barres groupées":
        fig = px.bar(t, y="Thème", x="Mentions", color="Colonne", barmode="group", orientation="h",
                     category_orders={"Thème": ordre}, color_discrete_map=CMAP, height=650)
    else:
        cats = piv.sort_values("Total", ascending=False).index.tolist()
        fig = go.Figure()
        for c_, col_ in CMAP.items():
            fig.add_trace(go.Scatter(x=cats, y=[piv.loc[cat, c_] for cat in cats], name=c_,
                                     mode="lines+markers", fill="tozeroy",
                                     line=dict(color=col_, shape="spline")))
        fig.update_layout(height=550, yaxis_title="Mentions")
        fig.update_xaxes(tickangle=-35)
    st.plotly_chart(fig, width="stretch")
    st.caption("Une réponse peut compter dans plusieurs thèmes. Détection par mots-clés (approximative).")

    # Thèmes en pourcentage : part des réponses utiles (hors vides et « Rien / RAS ») de chaque colonne
    st.subheader("Thèmes en pourcentage")
    st.caption("Pourcentage des réponses avec contenu de chaque colonne qui citent le thème. "
               "Une réponse peut citer plusieurs thèmes : le total dépasse donc 100 %.")
    base_n = {c: max(1, len(utiles(c))) for c in FB_COL}
    pc = pd.DataFrame([{"Thème": t_, "Colonne": FB_COL[c],
                        "Pourcentage": 100 * int(d[f"{c}|{t_}"].sum()) / base_n[c]}
                       for c in FB_COL for t_ in FB_THEMES])
    ordre_pc = pc.groupby("Thème").Pourcentage.sum().sort_values().index.tolist()
    fig_pc = px.bar(pc, y="Thème", x="Pourcentage", color="Colonne", barmode="group", orientation="h",
                    category_orders={"Thème": ordre_pc}, color_discrete_map=CMAP, height=650,
                    text=pc.Pourcentage.map(lambda v: f"{v:.1f} %"))
    fig_pc.update_traces(textposition="outside", cliponaxis=False)
    fig_pc.update_xaxes(title="% des réponses avec contenu", ticksuffix=" %")
    st.plotly_chart(fig_pc, width="stretch")

    # ------------------------------------------------------- Mots fréquents
    st.divider()
    st.header("Mots fréquents")
    n = st.slider("Nombre de mots", 10, 50, 20, key="fb_nmots")
    c1, c2 = st.columns(2)
    for col_ui, c, color in ((c1, "ameliorer", "#d1603d"), (c2, "souhait", "#2f7d6d")):
        mots = Counter(w for x in utiles(c).str.lower()
                       for w in re.findall(r"[\w؀-ۿ]{4,}", x) if w not in FB_STOP).most_common(n)
        g = pd.DataFrame(mots, columns=["Mot", "Occurrences"]).iloc[::-1]
        col_ui.subheader(FB_COL[c])
        col_ui.plotly_chart(px.bar(g, x="Occurrences", y="Mot", orientation="h",
                                   color_discrete_sequence=[color], height=max(350, 22 * n)),
                            width="stretch")

    # ------------------------------------------------- Autres graphiques
    st.divider()
    st.header("Autres graphiques")
    th = pd.DataFrame({
        "Thème": list(FB_THEMES),
        "À améliorer": [int(d[f"ameliorer|{t_}"].sum()) for t_ in FB_THEMES],
        "Souhait": [int(d[f"souhait|{t_}"].sum()) for t_ in FB_THEMES]})
    th["Total"] = th["À améliorer"] + th["Souhait"]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Répartition des thèmes (treemap)")
        if (th.Total > 0).any():
            st.plotly_chart(px.treemap(th[th.Total > 0], path=["Thème"], values="Total",
                                       color="Total", color_continuous_scale="Teal"),
                            width="stretch")
        else:
            st.info("Aucun thème détecté avec ces filtres.")
    with c2:
        st.subheader("Langue des réponses")
        lang = pd.Series(["Arabe" if a else "Latin (FR/EN/darija)" for a in d.arabe]).value_counts()
        st.plotly_chart(px.pie(values=lang.values, names=lang.index, hole=0.55,
                               color_discrete_sequence=["#2f7d6d", "#d1603d"]),
                        width="stretch")

    # ------------------------------------------- Explorer les réponses
    st.divider()
    st.header("Explorer les réponses")
    st.subheader("Lire les réponses d'un thème")
    t_choisi = st.selectbox("Thème", list(FB_THEMES), key="fb_theme_choisi")
    colA, colB = st.columns(2)
    for ui, c in ((colA, "ameliorer"), (colB, "souhait")):
        sel = d.loc[d[f"{c}|{t_choisi}"], c]
        ui.markdown(f"**{FB_COL[c]}** – {len(sel)} réponses")
        for x in (sel.sample(min(8, len(sel)), random_state=1) if len(sel) else []):
            ui.markdown(f"- {x[:200]}")

    # Téléchargement des lignes du thème choisi
    mask = d[f"ameliorer|{t_choisi}"] | d[f"souhait|{t_choisi}"]
    exp = d.loc[mask, ["id", "ameliorer", "souhait"]].rename(columns={"id": "Ligne Excel", **FB_COL})
    exp.insert(1, "Thème", t_choisi)
    exp["Trouvé dans"] = ["Les deux" if a and b else ("À améliorer" if a else "Souhait")
                          for a, b in zip(d.loc[mask, f"ameliorer|{t_choisi}"],
                                          d.loc[mask, f"souhait|{t_choisi}"])]
    st.divider()
    st.write(f"**{len(exp)} lignes** pour le thème « {t_choisi} » (réponses citant ce thème dans au "
             "moins une colonne, filtres de la barre latérale appliqués).")
    nom = re.sub(r"[^\w]+", "_", t_choisi).strip("_").lower()
    b1, b2 = st.columns(2)
    b1.download_button("⬇️ Télécharger en CSV", exp.to_csv(index=False).encode("utf-8-sig"),
                       f"lignes_{nom}.csv", "text/csv", disabled=exp.empty, width="stretch",
                       key="fb_dl_csv")
    buf = io.BytesIO()
    exp.to_excel(buf, index=False, sheet_name=t_choisi[:30].replace("/", "-"))
    b2.download_button("⬇️ Télécharger en Excel", buf.getvalue(), f"lignes_{nom}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       disabled=exp.empty, width="stretch", key="fb_dl_xlsx")


# --- Navigation entre les pages -------------------------------------------------
PAGE_SEG = "🎯 Segmentation"
PAGE_FB = "💬 Retours clients"
page = st.sidebar.segmented_control("🧭 Navigation", [PAGE_SEG, PAGE_FB], default=PAGE_SEG,
                                    key="nav_page", width="stretch") or PAGE_SEG
if page == PAGE_FB:
    render_feedback_page()
    st.stop()

NONE_LABEL = "— Aucune —"
SEED = 42

NUMERIC_LABELS = {
    "n_achats": "Fréquence d'achat (nb de billets)",
    "montant_total": "Montant total dépensé",
    "panier_moyen": "Panier moyen",
    "recence_jours": "Récence (jours depuis le dernier achat)",
    "anciennete_jours": "Ancienneté (jours depuis le 1er achat)",
    "n_destinations": "Nombre de destinations distinctes",
    "n_origines": "Nombre d'origines distinctes",
    "delai_reservation_jours": "Délai moyen de réservation (jours entre achat et voyage)",
    "taux_operation_non_standard_pct": "Taux de transactions modifiées/remboursées (%)",
    "taux_reponse_souhait_pct": "Taux de réponse au champ Souhait (%)",
    "taux_scan_embarquement": "Taux de scan à l'embarquement (%)",
    # --- Récurrence d'achat (issue de l'email payeur, rattachée au participant) ---
    "n_commandes_payeur": "Récurrence : nb de commandes du payeur",
    "intervalle_median_jours_payeur": "Intervalle médian entre commandes du payeur (jours)",
    "billets_par_commande_payeur": "Billets par commande (payeur)",
    "n_participants_payes_payeur": "Nb de participants payés par le même email",
    "n_emails": "Nb d'emails payeurs distincts utilisés",
}

# Variables de récurrence uniquement illustratives (profilage des segments) :
# l'intervalle est vide pour tous les payeurs à 1 commande (l'imputation par la
# médiane les ferait passer pour des clients au rythme médian) et le nb de
# participants payés décrit le payeur, pas le comportement du participant.
ILLUSTRATIVE_ONLY_NUMERIC = {"intervalle_median_jours_payeur", "n_participants_payes_payeur"}

CAT_LABELS = {
    "genre": "Genre",
    "tranche_age": "Tranche d'âge",
    "situation": "Situation professionnelle",
    "raison_principale": "Raison de voyage principale",
    "periode_principale": "Période de voyage principale",
    "accompagnement_principal": "Type d'accompagnement principal",
    "moyen_paiement": "Moyen de paiement préféré",
    "origine_principale": "Origine principale",
    "destination_principale": "Destination principale",
    "type_produit": "Type de produit",
    "type_operation": "Type d'opération dominant",
    "type_agence": "Type d'agence",
    "mode_saisie": "Mode de saisie du billet",
    "tranche_anticipation": "Tranche d'anticipation",
    "classe_recurrence_payeur": "Classe de récurrence du payeur (billets)",
}


# =============================================================================
# Chargement des données
# =============================================================================
@st.cache_data(show_spinner=False)
def list_sheets(file_bytes: bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return xl.sheet_names


@st.cache_data(show_spinner="Lecture du fichier Excel...")
def load_excel(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def guess_col(columns, candidates):
    cols_lower = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.strip().lower() in cols_lower:
            return cols_lower[cand.strip().lower()]
    return None


def to_float_robust(s: pd.Series) -> pd.Series:
    """Nombres lus de façon tolérante : « 14,658 » (virgule décimale) devient 14.658."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.strip().str.replace(",", ".", regex=False), errors="coerce")


def to_datetime_robust(s: pd.Series) -> pd.Series:
    """Dates lues de façon tolérante. Si la colonne contient des numéros de série Excel
    (ex. 46227,375 = 2026-08-02 09:00) au lieu de vraies dates, ils sont convertis ; sinon
    lecture classique. Sans cela, un numéro de série serait lu comme une date de 1970."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    nums = to_float_robust(s)
    n_ok = int(s.notna().sum())
    if n_ok and nums.notna().sum() >= 0.8 * n_ok and 20000 <= nums.median() <= 80000:
        return pd.to_datetime(nums, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def clean_tranche_label(s: pd.Series) -> pd.Series:
    """Étiquettes de tranche nettoyées : l'apostrophe finale (« 1-5' ») sert à Excel à empêcher
    la conversion en date ; elle est retirée pour l'affichage."""
    t = s.astype("string").str.strip().str.strip("'’`´").str.strip()
    # « =1' » (exactement 1) : le « = » est retiré, sinon Excel lirait « =1 » comme une formule à l'export
    t = t.str.replace(r"^[=\s]+", "", regex=True)
    # « - » seul (tranche non calculée) = valeur absente : jamais affichée comme tranche
    return t.mask(t.eq("") | t.str.fullmatch(r"[-–—_.\s]+", na=False))


def select_col(label, columns, default_candidates, key):
    options = [NONE_LABEL] + list(columns)
    guess = guess_col(columns, default_candidates)
    index = options.index(guess) if guess in options else 0
    return st.sidebar.selectbox(label, options, index=index, key=key)


# =============================================================================
# Agrégation au niveau participant (RFM + profil comportemental)
# =============================================================================
def mode_per_id(df: pd.DataFrame, id_col: str, col: str) -> pd.Series:
    """Modalité la plus fréquente de `col` pour chaque identifiant."""
    def _mode(s):
        m = s.mode()
        return m.iat[0] if not m.empty else np.nan
    return df.groupby(id_col)[col].agg(_mode)


def _explode_tags(df: pd.DataFrame, id_col: str, col: str, sep: str, exclude_values, date_col=None):
    """Une ligne par (identifiant, étiquette individuelle), valeurs vides exclues."""
    cols = [id_col, col] + ([date_col] if date_col else [])
    tmp = df[cols].copy()
    tmp[col] = tmp[col].fillna("-").astype(str)
    tmp = tmp.assign(_tag=tmp[col].str.split(sep)).explode("_tag")
    tmp["_tag"] = tmp["_tag"].str.strip()
    excl = {v.lower() for v in exclude_values}
    return tmp[~tmp["_tag"].str.lower().isin(excl)]


def top_tag_per_id(df: pd.DataFrame, id_col: str, col: str, sep: str = "|",
                    exclude_values=("-", "", "nan", "none"), date_col=None) -> pd.Series:
    """
    Pour une colonne multi-valeurs (ex: 'Loisirs | Travail'), renvoie pour
    chaque identifiant l'étiquette individuelle la plus fréquente sur l'ensemble
    de ses lignes. En cas d'égalité, règle fixe et reproductible :
      1. l'étiquette citée dans l'achat le plus récent (si `date_col` est fourni) ;
      2. à défaut, l'étiquette la plus courante dans tout le fichier ;
      3. à défaut, l'ordre alphabétique.
    """
    tmp = _explode_tags(df, id_col, col, sep, exclude_values, date_col)
    if tmp.empty:
        return pd.Series(dtype=object, name=col)
    freq_globale = tmp["_tag"].value_counts()
    if date_col:
        counts = tmp.groupby([id_col, "_tag"]).agg(n=("_tag", "size"), dernier=(date_col, "max")).reset_index()
    else:
        counts = tmp.groupby([id_col, "_tag"]).size().reset_index(name="n")
        counts["dernier"] = pd.Timestamp(0)
    counts["dernier"] = counts["dernier"].fillna(pd.Timestamp(0))
    counts["freq_globale"] = counts["_tag"].map(freq_globale)
    counts = counts.sort_values([id_col, "n", "dernier", "freq_globale", "_tag"],
                                ascending=[True, False, False, False, True], kind="stable")
    return counts.drop_duplicates(subset=id_col, keep="first").set_index(id_col)["_tag"]


@st.cache_data(show_spinner="Analyse des raisons de voyage...")
def build_reasons_multihot(df: pd.DataFrame, colmap: dict, sep: str = "|",
                           exclude_values=("-", "", "nan", "none")) -> pd.DataFrame:
    """Une ligne par participant, une colonne par raison : 1 si le participant a cité cette
    raison dans au moins un de ses achats (aucune information perdue, contrairement à la
    seule raison principale). Les participants sans raison valide sont absents de l'index."""
    id_col, col = colmap["id"], colmap["reason"]
    tmp = df[[id_col, col]].copy()
    tmp[id_col] = tmp[id_col].astype(str)
    tmp = _explode_tags(tmp, id_col, col, sep, exclude_values).reset_index(drop=True)
    if tmp.empty:
        return pd.DataFrame()
    return pd.crosstab(tmp[id_col], tmp["_tag"]).clip(upper=1)


def reasons_mentions_line(multi, ids, top: int = 6, with_global: bool = True) -> str:
    """« Travail 45% (global 30%) · Loisirs / Vacances 40% (global 52%) ... » : part des
    participants du segment qui citent chaque raison, comparée à l'ensemble des participants."""
    if multi is None or multi.empty:
        return ""
    seg = multi.reindex(list(ids)).fillna(0).mean() * 100
    glob = multi.mean() * 100
    seg = seg[seg > 0].sort_values(ascending=False).head(top)
    if not with_global:
        return " · ".join(f"{k} {v:.0f}%" for k, v in seg.items())
    return " · ".join(f"{k} {v:.0f}% (global {glob[k]:.0f}%)" for k, v in seg.items())


@st.cache_data(show_spinner="Agrégation des participants...")
def build_participant_table(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """Transforme une table de transactions en une table 1 ligne = 1 participant."""
    id_col = colmap["id"]
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    g = df.groupby(id_col)

    part = pd.DataFrame(index=g.size().index)
    part["n_achats"] = g.size()

    if colmap.get("price"):
        pcol = colmap["price"]
        df[pcol] = to_float_robust(df[pcol])
        part["montant_total"] = g[pcol].sum()
        part["panier_moyen"] = g[pcol].mean()

    if colmap.get("date"):
        dcol = colmap["date"]
        df[dcol] = to_datetime_robust(df[dcol])
        ref_date = df[dcol].max() + pd.Timedelta(days=1)
        part["premier_achat"] = g[dcol].min()
        part["dernier_achat"] = g[dcol].max()
        part["recence_jours"] = (ref_date - part["dernier_achat"]).dt.days
        part["anciennete_jours"] = (ref_date - part["premier_achat"]).dt.days

        if colmap.get("trip_date"):
            tcol = colmap["trip_date"]
            df[tcol] = to_datetime_robust(df[tcol])
            df["_delai_reservation"] = (df[tcol] - df[dcol]).dt.days
            part["delai_reservation_jours"] = g["_delai_reservation"].mean()

    if colmap.get("dest"):
        part["n_destinations"] = g[colmap["dest"]].nunique()
        part["destination_principale"] = mode_per_id(df, id_col, colmap["dest"])
    if colmap.get("origin"):
        part["n_origines"] = g[colmap["origin"]].nunique()
        part["origine_principale"] = mode_per_id(df, id_col, colmap["origin"])

    simple_map = {
        "genre": "genre", "age": "tranche_age", "situation": "situation",
        "payment": "moyen_paiement", "agency": "type_agence", "saisie": "mode_saisie",
        "type": "type_produit",
    }
    for key, label in simple_map.items():
        col = colmap.get(key)
        if col:
            part[label] = mode_per_id(df, id_col, col)

    if colmap.get("operation"):
        ocol = colmap["operation"]
        part["type_operation"] = mode_per_id(df, id_col, ocol)
        global_mode_op = df[ocol].mode()
        if not global_mode_op.empty:
            ref_op = global_mode_op.iat[0]
            part["taux_operation_non_standard_pct"] = g[ocol].apply(
                lambda s: (s.astype(str) != str(ref_op)).mean() * 100
            )

    if colmap.get("wish"):
        wcol = colmap["wish"]
        part["taux_reponse_souhait_pct"] = g[wcol].apply(
            lambda s: (~s.astype(str).str.strip().isin(["", "-", "nan"])).mean() * 100
        )

    multi_map = {
        "reason": "raison_principale",
        "period": "periode_principale",
        "company": "accompagnement_principal",
    }
    for key, label in multi_map.items():
        col = colmap.get(key)
        if col:
            top = top_tag_per_id(df, id_col, col, date_col=colmap.get("date"))
            part[label] = top.reindex(part.index).fillna("Non renseigné")

    if colmap.get("scan"):
        scol = colmap["scan"]
        part["taux_scan_embarquement"] = g[scol].apply(
            lambda s: s.astype(str).str.strip().str.lower().isin(["oui", "yes", "true", "1"]).mean() * 100
        )

    if colmap.get("anticipation_h"):
        hcol = colmap["anticipation_h"]
        df[hcol] = to_float_robust(df[hcol])
        part["anticipation_moyenne_heures"] = g[hcol].mean()
        df["_tranche_anticipation"] = bucket_anticipation_heures(df[hcol])
        part["tranche_anticipation"] = mode_per_id(df, id_col, "_tranche_anticipation")

    part = part.reset_index().rename(columns={id_col: "participant_id"})
    return part


# =============================================================================
# Préparation des variables pour K-Means
# =============================================================================
def prepare_features(part_df: pd.DataFrame, numeric_cols, cat_cols, cat_weight=1.0, max_categories=None):
    X_parts, feature_names = [], []
    numeric_used = [c for c in numeric_cols if c in part_df.columns]
    if numeric_used:
        num_data = part_df[numeric_used].astype(float)
        num_data = num_data.fillna(num_data.median())
        scaler = StandardScaler()
        X_num = scaler.fit_transform(num_data)
        X_parts.append(X_num)
        feature_names.extend(numeric_used)

    cat_used = [c for c in cat_cols if c in part_df.columns]
    if cat_used:
        cat_data = part_df[cat_used].astype(str).fillna("Non renseigné")
        if max_categories:
            for col in cat_used:
                top_vals = cat_data[col].value_counts().nlargest(max_categories - 1).index
                cat_data[col] = np.where(cat_data[col].isin(top_vals), cat_data[col], "Autres")
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # compatibilité anciennes versions de scikit-learn
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        X_cat = encoder.fit_transform(cat_data) * cat_weight
        X_parts.append(X_cat)
        feature_names.extend(encoder.get_feature_names_out(cat_used).tolist())

    if not X_parts:
        return np.zeros((len(part_df), 0)), [], numeric_used, cat_used
    X = np.hstack(X_parts)
    return X, feature_names, numeric_used, cat_used


@st.cache_data(show_spinner="Calcul de la courbe du coude et du score de silhouette...")
def compute_elbow_silhouette(X: np.ndarray, k_min: int, k_max: int, seed: int = SEED):
    ks = list(range(k_min, k_max + 1))
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if 1 < k < len(X):
            sils.append(silhouette_score(X, labels))
        else:
            sils.append(np.nan)
    return ks, inertias, sils


def run_kmeans(X: np.ndarray, k: int, seed: int = SEED):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


@st.cache_data(show_spinner="Comparaison des segmentations k=2 à k=5...")
def compute_k_comparison(X: np.ndarray, k_min: int = 2, k_max: int = 5, seed: int = SEED):
    """Calcule en une seule passe labels/inertie/silhouette pour k=k_min..k_max,
    pour alimenter l'onglet de comparaison sans relancer K-Means à chaque interaction."""
    results = {}
    for k in range(k_min, k_max + 1):
        labels, km_model = run_kmeans(X, k, seed)
        sil = silhouette_score(X, labels) if 1 < k < len(X) else np.nan
        results[k] = {"labels": labels, "inertia": km_model.inertia_, "silhouette": sil}
    return results


def compute_pca(X: np.ndarray, n_components: int):
    """Projection PCA à n_components dimensions (2 ou 3), avec repli propre
    si le nombre de variables disponibles est insuffisant."""
    n_comp = min(n_components, X.shape[1]) if X.shape[1] > 0 else 0
    if n_comp < n_components:
        coords = np.zeros((X.shape[0], n_components))
        if n_comp > 0:
            pca = PCA(n_components=n_comp, random_state=SEED)
            coords[:, :n_comp] = pca.fit_transform(X)
            var_exp = np.zeros(n_components)
            var_exp[:n_comp] = pca.explained_variance_ratio_
        else:
            var_exp = np.zeros(n_components)
        return coords, var_exp
    pca = PCA(n_components=n_components, random_state=SEED)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def numeric_profile(part_df: pd.DataFrame, numeric_cols, cluster_col="cluster"):
    global_mean = part_df[numeric_cols].mean()
    global_std = part_df[numeric_cols].std().replace(0, np.nan)
    cluster_mean = part_df.groupby(cluster_col)[numeric_cols].mean()
    z = (cluster_mean - global_mean) / global_std
    return cluster_mean, z


def profile_numeric_cols(df: pd.DataFrame) -> list:
    """Toutes les variables numériques disponibles pour PROFILER les segments, qu'elles
    aient servi ou non au clustering (variables actives + variables illustratives).
    Sans cela, avec le réglage par défaut (aucune variable numérique active), les
    z-scores sont vides et tous les segments reçoivent le même persona."""
    return [c for c in NUMERIC_LABELS if c in df.columns]


def segment_vs_global(cluster_mean: pd.DataFrame, df: pd.DataFrame, cols: list, active: list, cluster_id):
    """Tableau « segment vs moyenne globale » avec indication des variables actives."""
    comp = pd.DataFrame({
        "Segment": cluster_mean.loc[cluster_id, cols],
        "Moyenne globale": df[cols].mean(),
    }).round(1)
    comp["Dans le clustering"] = ["Oui" if i in active else "Non" for i in comp.index]
    comp.index = [NUMERIC_LABELS.get(i, i) for i in comp.index]
    return comp


def categorical_profile(part_df: pd.DataFrame, cat_col: str, cluster_col="cluster"):
    tab = pd.crosstab(part_df[cluster_col], part_df[cat_col], normalize="index") * 100
    return tab.round(1)


def top_lift_modality(part_df: pd.DataFrame, cat_col: str, cluster_id, cluster_col="cluster", min_share=0.05):
    """
    Modalité sur-représentée dans le cluster par rapport à la moyenne globale.

    Seules les modalités représentant au moins `min_share` (5% par défaut) du segment
    sont candidates : cela évite qu'une catégorie anecdotique (ex. 1 ou 2 participants
    sur une modalité quasi absente du dataset entier, comme "Non renseigné" ou une
    agence marginale) ne remonte artificiellement avec un ratio de sur-représentation
    énorme mais une présence réelle négligeable. Si aucune modalité n'atteint ce seuil,
    on se replie sur la simple majorité du segment.
    """
    global_dist = part_df[cat_col].value_counts(normalize=True)
    cluster_dist = part_df.loc[part_df[cluster_col] == cluster_id, cat_col].value_counts(normalize=True)
    if cluster_dist.empty:
        return None, None
    candidates = cluster_dist[cluster_dist >= min_share]
    if candidates.empty:
        top_mod = cluster_dist.index[0]
        return top_mod, cluster_dist.iloc[0] * 100
    lift = (candidates / global_dist.reindex(candidates.index)).sort_values(ascending=False)
    top_mod = lift.index[0]
    return top_mod, candidates.get(top_mod, np.nan) * 100


# Ordre de lecture "naturel" pour la phrase-portrait d'un segment, et petit
# gabarit de formulation par variable (ex: "voyage pour Travail", "en Été").
_SENTENCE_ORDER = ["genre", "tranche_age", "situation", "raison_principale",
                    "periode_principale", "accompagnement_principal",
                    "moyen_paiement", "type_agence", "mode_saisie"]
_SENTENCE_TEMPLATES = {
    "raison_principale": "voyage pour {}",
    "periode_principale": "en {}",
    "moyen_paiement": "paiement {}",
    "type_agence": "agence {}",
}


def ordered_cat_cols(cat_cols):
    """Ordonne une liste de variables catégorielles selon l'ordre de lecture
    "naturel" défini par _SENTENCE_ORDER (les variables non listées suivent, dans
    leur ordre d'origine)."""
    return [c for c in _SENTENCE_ORDER if c in cat_cols] + [c for c in cat_cols if c not in _SENTENCE_ORDER]


def majority_modality(part_df: pd.DataFrame, cat_col: str, cluster_id, cluster_col="cluster"):
    """Modalité la plus fréquente au sein du segment (simple majorité, sans comparaison
    à la moyenne globale) — utilisée pour la phrase-portrait "type" d'un segment."""
    cluster_dist = part_df.loc[part_df[cluster_col] == cluster_id, cat_col].value_counts(normalize=True)
    if cluster_dist.empty:
        return None, None
    return cluster_dist.index[0], cluster_dist.iloc[0] * 100


def modality_breakdown(part_df: pd.DataFrame, cat_col: str, cluster_id, cluster_col="cluster", max_modalities=5):
    """Répartition (%) de TOUTES les modalités d'une variable catégorielle au sein
    d'un segment, triée par fréquence décroissante — pas seulement la modalité
    majoritaire. Les modalités au-delà de max_modalities sont regroupées sous un
    total "+N autre(s)" pour rester lisible sur les variables à forte cardinalité."""
    dist = part_df.loc[part_df[cluster_col] == cluster_id, cat_col].value_counts(normalize=True) * 100
    if dist.empty:
        return ""
    top = dist.iloc[:max_modalities]
    bits = [f"{mod} ({pct:.0f}%)" for mod, pct in top.items()]
    rest = dist.iloc[max_modalities:]
    if not rest.empty:
        bits.append(f"+{len(rest)} autre(s) ({rest.sum():.0f}%)")
    return " · ".join(bits)


def build_profile_sentence(part_df: pd.DataFrame, cat_cols, cluster_id, cluster_col="cluster"):
    """
    Combine les modalités catégorielles majoritaires d'un segment en une phrase
    courte de type "profil type", ex : "Hommes, 25-34 ans, Salarié(e), voyage
    pour Travail, en Été, Seul(e)".
    """
    ordered = ordered_cat_cols(cat_cols)
    bits = []
    for col in ordered:
        top_mod, share = majority_modality(part_df, col, cluster_id, cluster_col)
        if top_mod is None:
            continue
        phrase = _SENTENCE_TEMPLATES.get(col, "{}").format(top_mod)
        bits.append(f"{phrase} ({share:.0f}%)")
    return ", ".join(bits)


def build_persona_label(z_row: pd.Series):
    """Heuristique simple de nommage marketing à partir des z-scores RFM."""
    def lvl(key, seuil=0.5):
        v = z_row.get(key, np.nan)
        if pd.isna(v):
            return 0
        if v > seuil:
            return 1
        if v < -seuil:
            return -1
        return 0

    # Fréquence : récurrence du payeur (nb de commandes) si disponible, sinon nb de
    # billets du participant.
    lf = lvl("n_commandes_payeur") if pd.notna(z_row.get("n_commandes_payeur", np.nan)) else lvl("n_achats")
    lm = lvl("montant_total")
    lp = lvl("panier_moyen")
    lr = lvl("recence_jours")   # positif = achat ancien (mauvais signe)
    ld = lvl("n_destinations")

    if lf >= 1 and lm >= 1 and lr <= 0:
        return ("Clients VIP / fidèles actifs",
                "Fréquence et montant élevés, avec un achat récent : le cœur de cible à "
                "fidéliser et à privilégier pour des offres premium ou un programme VIP.")
    if lf >= 1 and lr >= 1:
        return ("Fidèles en dormance (à réactiver)",
                "Historique d'achats fréquent mais dernier achat ancien : cible prioritaire "
                "pour une campagne de relance / réactivation personnalisée.")
    if lf <= 0 and lp >= 1 and lr <= 0:
        return ("Nouveaux clients à fort potentiel",
                "Peu d'achats mais panier moyen élevé et achat récent : profil premium "
                "à développer via du cross-sell / up-sell.")
    if lr >= 1 and lf <= 0 and lm <= 0:
        return ("Clients à risque de désengagement",
                "Achats peu fréquents, montant faible et dernier achat ancien : segment "
                "à risque d'attrition, à surveiller ou réengager avec une offre incitative.")
    if lf <= -1 and lm <= -1:
        return ("Occasionnels / petits budgets",
                "Faible fréquence et faible montant : segment sensible au prix, potentiel "
                "de montée en gamme limité mais volumétrie intéressante pour des offres d'appel.")
    if ld >= 1:
        return ("Explorateurs multi-destinations",
                "Nombre de destinations distinctes élevé : clientèle mobile et curieuse, "
                "cible pertinente pour des offres multi-destinations ou pass illimité.")
    return ("Segment standard",
            "Profil proche de la moyenne globale sur les indicateurs clés : segment de "
            "volume à analyser plus finement avec des critères complémentaires.")


def to_excel_bytes(sheets: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, d in sheets.items():
            d.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


def fmt_int(x):
    return f"{x:,.0f}".replace(",", " ")


# =============================================================================
# Qualité des données — associations et taille d'effet
# =============================================================================
def v_cramer_corrige(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér corrigé du biais (Bergsma, 2013)."""
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    if n <= 1:
        return np.nan
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2_corr / denom))


def matrice_v_cramer(part_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    cols = [c for c in cols if c in part_df.columns]
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = v_cramer_corrige(part_df[cols[i]], part_df[cols[j]])
            mat.iloc[i, j] = mat.iloc[j, i] = v
    return mat


DELAI_LABELS = ["Même jour", "1-3 jours", "4-7 jours", "8-14 jours", "15-30 jours", "+30 jours"]


def bucket_delai(jours: pd.Series) -> pd.Series:
    """Regroupe un délai en jours dans les mêmes tranches que le rapport
    d'analyse exploratoire (Même jour, 1-3 jours, ...)."""
    bornes = [-np.inf, 1, 4, 8, 15, 31, np.inf]
    return pd.cut(jours, bins=bornes, labels=DELAI_LABELS, right=False)


def bucket_anticipation_heures(heures: pd.Series) -> pd.Series:
    """Regroupe l'anticipation en heures (colonne source) dans les mêmes tranches que
    l'anticipation en jours (bucket_delai), en convertissant chaque seuil en heures
    (1 jour = 24h) : Même jour = <24h, 1-3 jours = 24-95h, ... +30 jours = >=744h."""
    bornes = [-np.inf, 24, 96, 192, 360, 744, np.inf]
    return pd.cut(heures, bins=bornes, labels=DELAI_LABELS, right=False)


@st.cache_data(show_spinner="Préparation de l'anticipation par transaction...")
def build_anticipation_table(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """Table au niveau transaction (1 ligne = 1 achat) : profil du participant,
    anticipation en heures (colonne source du fichier) et sa tranche (mêmes seuils
    que l'anticipation en jours, convertis en heures)."""
    id_col, hcol = colmap["id"], colmap["anticipation_h"]
    anticipation_h = to_float_robust(df[hcol])

    out = pd.DataFrame({"participant_id": df[id_col].astype(str)})
    for key, label in (("genre", "Genre"), ("age", "Tranche d'âge"), ("situation", "Situation")):
        col = colmap.get(key)
        if col:
            out[label] = df[col].values
    if colmap.get("date"):
        out["date_achat"] = to_datetime_robust(df[colmap["date"]])
    if colmap.get("trip_date"):
        out["date_voyage"] = to_datetime_robust(df[colmap["trip_date"]])
    out["anticipation_heures"] = anticipation_h.round(2)
    out["tranche_anticipation"] = bucket_anticipation_heures(anticipation_h)
    return out


def cohens_d(part_df: pd.DataFrame, col: str, cluster_col: str, c0, c1) -> float:
    """Taille d'effet pour une variable numérique : d de Cohen entre 2 clusters."""
    g0 = part_df.loc[part_df[cluster_col] == c0, col].dropna()
    g1 = part_df.loc[part_df[cluster_col] == c1, col].dropna()
    if len(g0) < 2 or len(g1) < 2:
        return np.nan
    n0, n1 = len(g0), len(g1)
    var_poolee = ((n0 - 1) * g0.var() + (n1 - 1) * g1.var()) / (n0 + n1 - 2)
    if var_poolee <= 0:
        return np.nan
    return float((g0.mean() - g1.mean()) / np.sqrt(var_poolee))


def total_variation_distance(part_df: pd.DataFrame, col: str, cluster_col: str, c0, c1) -> float:
    """Taille d'effet pour une variable catégorielle : distance en variation
    totale entre les 2 distributions de clusters (0 = identiques, 1 = disjointes
    -> la variable détermine quasi entièrement l'appartenance au cluster)."""
    p0 = part_df.loc[part_df[cluster_col] == c0, col].value_counts(normalize=True)
    p1 = part_df.loc[part_df[cluster_col] == c1, col].value_counts(normalize=True)
    modalites = p0.index.union(p1.index)
    p0 = p0.reindex(modalites, fill_value=0.0)
    p1 = p1.reindex(modalites, fill_value=0.0)
    return float(0.5 * (p0 - p1).abs().sum())


def pca_loadings(X: np.ndarray, feature_names: list, seed: int = SEED) -> pd.DataFrame:
    n_comp = min(2, X.shape[1]) if X.shape[1] > 0 else 0
    if n_comp == 0:
        return pd.DataFrame(columns=["Composante 1", "Composante 2"])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(X)
    data = pca.components_.T
    if n_comp == 1:
        data = np.hstack([data, np.zeros((data.shape[0], 1))])
    return pd.DataFrame(data, index=feature_names, columns=["Composante 1", "Composante 2"])


def choisir_k_parcimonie(ks: list, sils: list, tolerance: float = 0.02) -> int:
    """Le plus petit k dont la silhouette est à `tolerance` de la meilleure
    silhouette observée (principe de parcimonie, cohérent avec les scripts
    Untitled-1.py / analyse_diagnostic_clustering.py)."""
    valides = [(kk, s) for kk, s in zip(ks, sils) if not np.isnan(s)]
    if not valides:
        return ks[0]
    meilleure = max(s for _, s in valides)
    candidats = [kk for kk, s in valides if s >= meilleure - tolerance]
    return min(candidats)


# =============================================================================
# Récurrence d'achat — calculée par email payeur (colonne PayerEmail)
# =============================================================================
# Un email peut payer pour plusieurs participants (famille, amis, agence) et un
# participant peut utiliser plusieurs emails. La récurrence est donc calculée PAR
# EMAIL (« commande » = un email + un ProductCode si cette colonne est associée, sinon un email + un jour d'achat), puis rattachée à chaque
# participant via son email payeur principal. L'unité de segmentation reste le
# participant (le profil socio-démographique est à ce niveau).
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
# Classe de récurrence du payeur : basée sur le nombre de JOURS D'ACHAT distincts de l'email
# (calculé à partir de la date d'achat, quel que soit le ProductCode), mêmes tranches que le
# graphique de fréquence d'achat.
CLASSES_RECURRENCE = ["1 achat", "2-3 achats", "4-5 achats", "6-10 achats", "+10 achats"]
CLASSE_INTERMEDIAIRE = "Intermédiaire / agence"
CLASSE_SANS_EMAIL = "Email non renseigné"


def tranche_sort_key(label) -> tuple:
    """Ordre naturel d'étiquettes de tranche (« 1 achet » < « 1 a 2 achet » < « 5 a 10 achet »...) :
    d'après les nombres qu'elles contiennent ; sans nombre, à la fin."""
    nums = [int(x) for x in re.findall(r"\d+", str(label))]
    return (0, tuple(nums)) if nums else (1, (0,), str(label))


def normalize_email(s: pd.Series) -> pd.Series:
    """Minuscules, espaces retirés ; toute valeur qui n'a pas la forme x@y.z devient <NA>."""
    e = s.astype("string").str.strip().str.lower()
    return e.where(e.str.match(EMAIL_PATTERN, na=False))


def get_email_salt(file_bytes: bytes) -> str:
    """Sel de pseudonymisation : `EMAIL_SALT` dans .streamlit/secrets.toml si présent,
    sinon sel dérivé du contenu du fichier chargé. C'est une pseudonymisation d'affichage
    et d'export, pas une anonymisation : ne partagez pas le fichier source avec le résultat."""
    try:
        secret = st.secrets.get("EMAIL_SALT")
        if secret:
            return str(secret)
    except Exception:
        pass
    return hashlib.sha256(file_bytes[:1_000_000]).hexdigest()


def pseudonymize_emails(s: pd.Series, salt: str) -> pd.Series:
    """Remplace chaque email valide par un identifiant haché (10 caractères hexadécimaux)."""
    e = normalize_email(s)
    mapping = {u: hashlib.sha256((salt + str(u)).encode("utf-8")).hexdigest()[:10]
               for u in e.dropna().unique()}
    return e.astype(object).map(mapping)


def mask_emails(s: pd.Series) -> pd.Series:
    """Masque un email pour l'affichage : « ab***@gmail.com ». Les valeurs non vides mais
    invalides sont remplacées par « *** »."""
    e = normalize_email(s)
    masked = (e.str.split("@").str[0].str[:2] + "***@" + e.str.split("@").str[-1]).astype(object)
    return masked.mask(s.notna() & e.isna(), "***")


@st.cache_data(show_spinner="Calcul de la récurrence d'achat...")
def build_payer_table(df: pd.DataFrame, colmap: dict, seuil_intermediaire: int = 10):
    """
    Renvoie (payeurs, commandes, qualité).
      - payeurs  : 1 ligne = 1 email (nb de commandes, intervalles, récence, classe...)
      - commandes: 1 ligne = 1 commande (email, jour) avec l'intervalle depuis la précédente
      - qualité  : statistiques de validité de la colonne email et période observée
    Une commande = (email, jour d'achat) : acheter 5 billets d'un coup n'est pas de la
    récurrence.
    """
    raw = df[colmap["payer_email"]]
    email = normalize_email(raw)
    dates = to_datetime_robust(df[colmap["date"]])

    blank = raw.astype("string").str.strip().eq("").fillna(False).astype(bool)
    vides = raw.isna() | blank
    domaines = email.str.split("@").str[-1].value_counts().head(8)
    quality = {
        "n_lignes": int(len(df)),
        "n_vides": int(vides.sum()),
        "n_invalides": int((~vides & email.isna()).sum()),
        "n_valides": int(email.notna().sum()),
        "n_sans_date": int((email.notna() & dates.isna()).sum()),
        "date_min": dates.min(),
        "date_max": dates.max(),
        "domaines_top": {str(k): int(v) for k, v in domaines.items()},
    }

    d = pd.DataFrame({
        "email": email,
        "participant_id": df[colmap["id"]].astype(str),
        "date": dates,
    })
    ref_col = colmap.get("order_ref")
    if ref_col:
        ref = df[ref_col].astype("string").str.strip()
        ref = ref.mask(ref.eq(""))
        d["ref"] = ref
    d = d.dropna(subset=["email", "date"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), quality
    d["jour"] = d["date"].dt.normalize()

    # Définition de la commande :
    #  - avec référence (ex. ProductCode) : un email + une référence = 1 commande ;
    #    une ligne sans référence retombe sur « email + jour » ;
    #  - sans référence : un email + un jour d'achat = 1 commande.
    if ref_col:
        d["cmd"] = d["ref"].astype(object).where(d["ref"].notna(), "J_" + d["jour"].dt.strftime("%Y-%m-%d"))
        quality["n_sans_reference"] = int(d["ref"].isna().sum())
    else:
        d["cmd"] = d["jour"].dt.strftime("%Y-%m-%d")
    quality["mode_commande"] = "reference" if ref_col else "jour"

    orders = (d.groupby(["email", "cmd"])
                .agg(n_billets=("participant_id", "size"), jour=("jour", "min"))
                .reset_index().sort_values(["email", "jour", "cmd"]))
    orders["intervalle_j"] = orders.groupby("email")["jour"].diff().dt.days

    ref_date = dates.max().normalize() + pd.Timedelta(days=1)
    payer = orders.groupby("email").agg(
        n_commandes=("jour", "size"),
        n_billets=("n_billets", "sum"),
        premiere_commande=("jour", "min"),
        derniere_commande=("jour", "max"),
        intervalle_median_jours=("intervalle_j", "median"),
    )
    second = orders[orders.groupby("email").cumcount() == 1].set_index("email")["intervalle_j"]
    payer["delai_2e_commande_jours"] = second
    payer["n_participants_payes"] = d.groupby("email")["participant_id"].nunique()
    payer["billets_par_commande"] = payer["n_billets"] / payer["n_commandes"]
    payer["recence_jours"] = (ref_date - payer["derniere_commande"]).dt.days
    payer["est_recurrent"] = payer["n_commandes"] >= 2
    # Classe : nombre de jours d'achat distincts (par date, indépendant de la référence de commande)
    payer["n_jours_achat"] = d.groupby("email")["jour"].nunique()
    payer["classe_recurrence"] = pd.cut(payer["n_jours_achat"], [0, 1, 3, 5, 10, np.inf],
                                        labels=CLASSES_RECURRENCE)
    tr_col = colmap.get("tranche_cmd")
    if tr_col:
        # Tranche déjà calculée dans le fichier : on la reprend telle quelle (valeur la plus
        # fréquente parmi les lignes de l'email), sans la recalculer.
        trd = pd.DataFrame({"email": email, "tr": clean_tranche_label(df[tr_col])}).dropna()
        best = (trd.groupby(["email", "tr"]).size().rename("n").reset_index()
                   .sort_values(["email", "n", "tr"], ascending=[True, False, True], kind="stable")
                   .drop_duplicates("email").set_index("email")["tr"])
        payer["tranche_fichier"] = best.reindex(payer.index).astype(object)
        payer["classe_recurrence"] = payer["tranche_fichier"].fillna("Tranche non renseignée")
        quality["tranche_fichier"] = str(tr_col).strip()
    payer["intermediaire_probable"] = payer["n_participants_payes"] >= seuil_intermediaire
    return payer, orders, quality


@st.cache_data(show_spinner="Rattachement de la récurrence aux participants...")
def attach_payer_features(part: pd.DataFrame, df: pd.DataFrame, colmap: dict,
                          payer: pd.DataFrame, salt: str, keep_email: bool = False) -> pd.DataFrame:
    """
    Ajoute à la table participants les indicateurs de récurrence de son email payeur
    PRINCIPAL (le plus utilisé ; à égalité, le plus récent) :
      n_emails, n_commandes_payeur, intervalle_median_jours_payeur,
      billets_par_commande_payeur, n_participants_payes_payeur,
      classe_recurrence_payeur, payeur_id (pseudonyme).
    Les participants dont le payeur est un intermédiaire probable (agence) reçoivent la
    classe « Intermédiaire / agence » et des indicateurs vides : la récurrence de l'agence
    ne dit rien sur le participant. Sans email valide : « Email non renseigné ».
    """
    out = part.copy()
    d = pd.DataFrame({
        "participant_id": df[colmap["id"]].astype(str),
        "email": normalize_email(df[colmap["payer_email"]]),
        "date": to_datetime_robust(df[colmap["date"]]),
    }).dropna(subset=["email"])

    per = (d.groupby(["participant_id", "email"])
             .agg(n=("date", "size"), last=("date", "max")).reset_index()
             .sort_values(["participant_id", "n", "last"], ascending=[True, False, False], kind="stable"))
    main = per.drop_duplicates("participant_id").set_index("participant_id")["email"]
    email_p = out["participant_id"].map(main)

    out["n_emails"] = out["participant_id"].map(per.groupby("participant_id")["email"].nunique())

    interm = email_p.isin(payer.index[payer["intermediaire_probable"]])
    for c in ["n_commandes", "n_billets", "intervalle_median_jours", "billets_par_commande", "n_participants_payes"]:
        out[f"{c}_payeur"] = email_p.map(payer[c])
        if c != "n_participants_payes":
            out.loc[interm, f"{c}_payeur"] = np.nan

    classe = email_p.map(payer["classe_recurrence"].astype(object))
    out["classe_recurrence_payeur"] = (
        classe.mask(interm, CLASSE_INTERMEDIAIRE).fillna(CLASSE_SANS_EMAIL)
    )
    out["payeur_id"] = pseudonymize_emails(email_p, salt)
    if keep_email:
        out["email_principal"] = email_p
    return out


# =============================================================================
# Interface Streamlit
# =============================================================================
st.title("🎯 Segmentation marketing des participants — K-Means")
st.caption(
    "Identifiez des groupes homogènes de participants à la tombola à partir de leur "
    "comportement d'achat et de leur profil, puis interprétez-les d'un point de vue marketing."
)

with st.expander("ℹ️ Méthodologie de l'application", expanded=False):
    st.markdown(
        """
1. **Agrégation par participant** — le fichier source contient une ligne par achat ; l'application
   calcule pour chaque participant des indicateurs de type **RFM** (Récence, Fréquence, Montant),
   ainsi que son profil socio-démographique et ses habitudes de voyage dominantes.
2. **Préparation des variables** — les variables numériques sont standardisées (moyenne 0, écart-type 1)
   et les variables catégorielles sont encodées en variables indicatrices (one-hot).
3. **Choix de k** — la méthode du coude (inertie) et le score de silhouette aident à choisir un
   nombre de segments pertinent.
4. **K-Means** — l'algorithme regroupe les participants en k segments homogènes.
5. **Interprétation** — chaque segment est décrit par ses écarts à la moyenne globale (z-scores)
   et ses modalités dominantes, avec une proposition de persona marketing (à affiner avec votre
   connaissance métier).
6. **Récurrence d'achat (optionnel)** — si la colonne d'email payeur est associée, la récurrence est
   calculée **par email** (une commande = un email + un ProductCode si la colonne est associée, sinon un email + un jour d'achat), puis rattachée à chaque
   participant via son email principal. Ces variables servent à **profiler** les segments ; elles
   ne sont pas utilisées dans le clustering par défaut.
        """
    )

# -----------------------------------------------------------------------
# 1. Chargement du fichier
# -----------------------------------------------------------------------
st.sidebar.header("1. Données")
uploaded_file = st.sidebar.file_uploader("Fichier Excel des participants (.xlsx)", type=["xlsx", "xls"])

file_bytes, file_label = None, None
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_label = uploaded_file.name

if file_bytes is None:
    st.info("👆 Chargez un fichier Excel contenant les données des participants pour commencer.")
    st.stop()

sheets = list_sheets(file_bytes)
sheet_name = sheets[0] if len(sheets) == 1 else st.sidebar.selectbox("Feuille Excel", sheets, index=0)
df_raw = load_excel(file_bytes, sheet_name)

# -----------------------------------------------------------------------
# 2. Correspondance des colonnes
# -----------------------------------------------------------------------
cols = df_raw.columns.tolist()
with st.sidebar.expander("🧩 Correspondance des colonnes", expanded=False):
    id_col = select_col("Identifiant participant *", cols, ["Participant", "ParticipantID", "ID"], "map_id")
    date_col = select_col("Date d'achat", cols, ["date_achat", "purchase_date", "date", "OperationDateTime"], "map_date")
    trip_date_col = select_col("Date de voyage", cols, ["date_voyage", "trip_date", "DepartureDateTime"], "map_trip_date")
    price_col = select_col("Montant / Prix", cols, ["Price", "Montant", "Prix"], "map_price")
    genre_col = select_col("Genre", cols, ["Genre", "Sexe"], "map_genre")
    age_col = select_col("Tranche d'âge", cols, ["Tranche d'âge", "Age", "Tranche d'age"], "map_age")
    situation_col = select_col("Situation professionnelle", cols, ["Situation"], "map_situation")
    reason_col = select_col("Raisons de voyage", cols, ["Raisons de voyage", "Raison"], "map_reason")
    period_col = select_col("Périodes de voyage", cols, ["Périodes de voyage", "Periode"], "map_period")
    wish_col = select_col("Souhait / remarque", cols, ["Souhait", "Wish", "Commentaire"], "map_wish")
    company_col = select_col("Accompagnement", cols, ["Accompagnement"], "map_company")
    payment_col = select_col("Moyen de paiement", cols, ["PaymentMethod", "Moyen de paiement"], "map_payment")
    type_col = select_col("Type de produit", cols, ["Type"], "map_type")
    operation_col = select_col("Type d'opération", cols, ["Operation"], "map_operation")
    origin_col = select_col("Origine", cols, ["Origin_Name", "Origine", "OrigineName"], "map_origin")
    dest_col = select_col("Destination", cols, ["Destination_Name", "Destination", "DestinationName"], "map_dest")
    agency_col = select_col("Type d'agence", cols, ["TYPE_Agence", "type_agenceP"], "map_agency")
    saisie_col = select_col("Mode de saisie", cols, ["Mode de saisie du billet"], "map_saisie")
    scan_col = select_col("Scan à l'embarquement", cols, ["Scanné à l'embarquement"], "map_scan")
    anticipation_h_col = select_col(
        "Anticipation (heures)", cols,
        ["Anticipation par Heure", "Anticipation_Heure", "AnticipationHeure", "anticipation_heures"],
        "map_anticipation_h",
    )
    payer_email_col = select_col(
        "Email du payeur (récurrence)", cols,
        ["PayerEmail", "Payer_Email", "Payer Email", "Email"], "map_payer_email",
    )
    tranche_cmd_col = select_col(
        "Tranche d'achat (déjà calculée dans le fichier)", cols,
        ["TRANCHE D'ACHAT", "TRANCHE D’ACHAT", "tranche Nombre commande", "Tranche nombre de commandes",
         "tranche_nombre_commande"],
        "map_tranche_cmd",
    )
    # Optionnel et jamais détecté automatiquement : par défaut, une commande = un email + un jour
    # d'achat. Choisir une colonne ici (ex. ProductCode) fait compter une commande par email + code.
    order_ref_col = select_col(
        "Référence de commande (optionnel — sinon par date d'achat)", cols, [], "map_order_ref",
    )

seuil_intermediaire, emails_clair = 10, False
if payer_email_col != NONE_LABEL:
    with st.sidebar.expander("🔁 Récurrence (email payeur)", expanded=False):
        seuil_intermediaire = int(st.number_input(
            "Seuil « intermédiaire probable » (nb de participants payés par un même email)",
            min_value=2, max_value=1000, value=10, step=1,
            help="Un email qui paie pour au moins ce nombre de participants distincts est traité "
                 "comme un intermédiaire (agence, entreprise...) : sa récurrence ne décrit pas "
                 "le comportement d'un participant, il est donc isolé.",
        ))
        emails_clair = st.checkbox(
            "Afficher / exporter les emails en clair (données personnelles)", value=False,
            help="Par défaut, les emails sont masqués à l'écran et remplacés par un identifiant "
                 "haché (payeur_id) dans les exports. Loi 09-08 (Maroc) / RGPD : ne cochez que "
                 "si l'usage est légitime.",
        )

if id_col == NONE_LABEL:
    st.error("⚠️ Merci de sélectionner une colonne d'identifiant participant : la segmentation "
              "s'effectue par participant.")
    st.stop()

colmap = {
    "id": id_col,
    "date": None if date_col == NONE_LABEL else date_col,
    "trip_date": None if trip_date_col == NONE_LABEL else trip_date_col,
    "price": None if price_col == NONE_LABEL else price_col,
    "genre": None if genre_col == NONE_LABEL else genre_col,
    "age": None if age_col == NONE_LABEL else age_col,
    "situation": None if situation_col == NONE_LABEL else situation_col,
    "reason": None if reason_col == NONE_LABEL else reason_col,
    "period": None if period_col == NONE_LABEL else period_col,
    "wish": None if wish_col == NONE_LABEL else wish_col,
    "company": None if company_col == NONE_LABEL else company_col,
    "payment": None if payment_col == NONE_LABEL else payment_col,
    "type": None if type_col == NONE_LABEL else type_col,
    "operation": None if operation_col == NONE_LABEL else operation_col,
    "origin": None if origin_col == NONE_LABEL else origin_col,
    "dest": None if dest_col == NONE_LABEL else dest_col,
    "agency": None if agency_col == NONE_LABEL else agency_col,
    "saisie": None if saisie_col == NONE_LABEL else saisie_col,
    "scan": None if scan_col == NONE_LABEL else scan_col,
    "anticipation_h": None if anticipation_h_col == NONE_LABEL else anticipation_h_col,
    "payer_email": None if payer_email_col == NONE_LABEL else payer_email_col,
    "order_ref": None if order_ref_col == NONE_LABEL else order_ref_col,
    "tranche_cmd": None if tranche_cmd_col == NONE_LABEL else tranche_cmd_col,
}

part_df = build_participant_table(df_raw, colmap)
reasons_multi = build_reasons_multihot(df_raw, colmap) if colmap.get("reason") else None

# Récurrence d'achat par email payeur, rattachée aux participants (variables illustratives).
payer_df = orders_df = payer_quality = None
email_salt = get_email_salt(file_bytes)
if colmap.get("payer_email") and colmap.get("date"):
    payer_df, orders_df, payer_quality = build_payer_table(df_raw, colmap, seuil_intermediaire)
    if not payer_df.empty:
        part_df = attach_payer_features(part_df, df_raw, colmap, payer_df, email_salt, emails_clair)
elif colmap.get("payer_email"):
    st.sidebar.warning("Associez aussi la « Date d'achat » pour calculer la récurrence.")

anticipation_df = None
if colmap.get("anticipation_h"):
    anticipation_df = build_anticipation_table(df_raw, colmap)

# -----------------------------------------------------------------------
# 3. Sélection des variables de segmentation
# -----------------------------------------------------------------------
NUMERIC_CANDIDATES = [c for c in NUMERIC_LABELS
                      if c in part_df.columns and c not in ILLUSTRATIVE_ONLY_NUMERIC]
# Variables utilisées pour profiler les segments (actives ou non dans le clustering).
PROFILE_NUM = profile_numeric_cols(part_df)
CAT_CANDIDATES = [c for c in CAT_LABELS if c in part_df.columns]
# Mêmes variables actives que Untitled-1.py par défaut : profil (Genre, Âge,
# Situation) + canal/paiement + réponses multiples — pas de RFM numérique.
DEFAULT_CAT = [c for c in ["genre", "tranche_age", "situation", "moyen_paiement",
                            "type_agence", "raison_principale", "periode_principale",
                            "accompagnement_principal"] if c in CAT_CANDIDATES]
DEFAULT_NUMERIC = []

st.sidebar.header("2. Variables de segmentation")
with st.sidebar.expander("🎛️ Variables", expanded=True):
    numeric_cols = st.multiselect(
        "Variables numériques (comportement d'achat)",
        NUMERIC_CANDIDATES, default=DEFAULT_NUMERIC,
        format_func=lambda c: NUMERIC_LABELS.get(c, c),
        help="Vide par défaut, comme Untitled-1.py (aucune variable RFM) — ajoutez-en si besoin.",
    )
    cat_cols = st.multiselect(
        "Variables catégorielles (profil)",
        CAT_CANDIDATES, default=DEFAULT_CAT,
        format_func=lambda c: CAT_LABELS.get(c, c),
    )
    cat_weight = st.slider(
        "Poids des variables catégorielles", 0.0, 2.0, 1.0, 0.1,
        help="Pondère l'influence des variables catégorielles (encodées en 0/1) par rapport "
             "aux variables numériques standardisées, pour éviter qu'un grand nombre de "
             "modalités ne domine la distance euclidienne.",
    )
    max_categories = st.slider(
        "Nb max de modalités par variable (le reste → 'Autres')", 3, 40, 15, 1,
        help="Limite le nombre de colonnes indicatrices créées pour les variables à forte "
             "cardinalité (ex. Origine, Destination) : seules les N modalités les plus "
             "fréquentes sont conservées, le reste est regroupé sous 'Autres'.",
    )

if not numeric_cols and not cat_cols:
    st.warning("Sélectionnez au moins une variable dans la barre latérale pour lancer la segmentation.")
    st.stop()


X, feature_names, numeric_used, cat_used = prepare_features(
    part_df, numeric_cols, cat_cols, cat_weight, max_categories
)

# -----------------------------------------------------------------------
# 4. Choix de k
# -----------------------------------------------------------------------
st.sidebar.header("3. Nombre de clusters")
max_k_possible = max(2, min(10, len(part_df) - 1))

# k recommandé automatiquement par la même règle de parcimonie que Untitled-1.py
# (plus petit k dont la silhouette est à 0.02 de la meilleure sur k=2..max_k_possible).
if X.shape[1] > 0:
    ks_auto, _, sils_auto = compute_elbow_silhouette(X, 2, max_k_possible)
    k_recommande = choisir_k_parcimonie(ks_auto, [0 if np.isnan(s) else s for s in sils_auto])
else:
    k_recommande = min(4, max_k_possible)

with st.sidebar.expander("🔢 Choix de k", expanded=True):
    st.caption(f"💡 k recommandé (règle de parcimonie, comme Untitled-1.py) : **{k_recommande}**")
    k = st.slider("k (nombre de segments)", min_value=2, max_value=max_k_possible,
                   value=k_recommande)
    show_elbow = st.checkbox("Afficher les courbes d'aide au choix de k", value=True)

run_btn = st.sidebar.button("🚀 Lancer la segmentation K-Means", type="primary", width="stretch")

# -----------------------------------------------------------------------
# Onglets principaux
# -----------------------------------------------------------------------
(tab_overview, tab_k, tab_kcompare, tab_results,
 tab_recurrence, tab_export) = st.tabs(
    ["📄 Aperçu des données", "📈 Choix de k",
     "🔬 Comparaison k=2 à k=5", "🗺️ Résultats & interprétation marketing",
     "🔁 Récurrence d'achat", "💾 Export"]
)

with tab_overview:
    st.subheader("Aperçu des données brutes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes (transactions)", fmt_int(len(df_raw)))
    c2.metric("Participants uniques", fmt_int(df_raw[id_col].nunique()))
    c3.metric("Colonnes", len(df_raw.columns))
    df_preview = df_raw.head(50).copy()
    if colmap.get("payer_email") and not emails_clair:
        df_preview[colmap["payer_email"]] = mask_emails(df_preview[colmap["payer_email"]])
    st.dataframe(df_preview, width="stretch")

    st.divider()
    st.subheader("Table agrégée par participant")
    st.caption("Chaque participant correspond à une seule ligne. Les indicateurs de fréquence, "
               "montant et récence sont calculés à partir de l'ensemble de ses achats.")
    c1, c2 = st.columns(2)
    c1.metric("Participants", fmt_int(len(part_df)))
    c2.metric("Variables disponibles", len(numeric_used) + len(cat_used))
    st.dataframe(part_df.head(50), width="stretch")

    st.divider()
    st.subheader("Qualité des données et répartitions")
    st.caption("Répartitions calculées sur la table agrégée par participant "
               "(une ligne = un participant, modalité la plus fréquente pour les variables catégorielles).")

    repartition_cols = [c for c in ["genre", "tranche_age", "situation", "raison_principale",
                                    "type_agence", "mode_saisie"]
                         if c in part_df.columns]
    if not repartition_cols:
        st.info("Associez au moins une colonne de profil (genre, âge, situation, raisons de voyage, "
                "type d'agence, mode de saisie) dans la barre latérale pour afficher les répartitions.")
    else:
        n_cols = 2
        rows = [repartition_cols[i:i + n_cols] for i in range(0, len(repartition_cols), n_cols)]
        for row in rows:
            cols_ui = st.columns(len(row))
            for col_ui, col in zip(cols_ui, row):
                counts = part_df[col].value_counts(dropna=True)
                pct = (counts / counts.sum() * 100).round(1)
                fig = px.bar(
                    x=pct.values, y=pct.index.astype(str), orientation="h",
                    labels={"x": "% des participants", "y": ""},
                    title=CAT_LABELS.get(col, col), text=pct.values,
                    color_discrete_sequence=["#2a78d6"],
                )
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(t=40, b=10))
                col_ui.plotly_chart(fig, width="stretch")

    if reasons_multi is not None and not reasons_multi.empty:
        st.markdown("**Raisons de voyage — toutes les mentions**")
        n_part = len(part_df)
        pct_r = (reasons_multi.sum() / n_part * 100).round(1).sort_values()
        fig_r = px.bar(
            x=pct_r.values, y=pct_r.index.astype(str), orientation="h", text=pct_r.values,
            labels={"x": "% des participants qui citent cette raison", "y": ""},
            title="Raisons de voyage citées (un participant peut en citer plusieurs)",
            color_discrete_sequence=["#eb6834"],
        )
        fig_r.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_r, width="stretch")
        st.caption("Un participant compte dans chaque raison qu'il a citée au moins une fois : la somme "
                   "dépasse 100 %. Contrairement au graphique « Raison de voyage principale », aucune "
                   "raison n'est écartée.")

    if "delai_reservation_jours" in part_df.columns:
        st.markdown("**Délai entre achat et voyage**")
        delai_cat = bucket_delai(part_df["delai_reservation_jours"])
        counts = delai_cat.value_counts(dropna=True)
        counts = counts.reindex([o for o in DELAI_LABELS if o in counts.index])
        pct = (counts / counts.sum() * 100).round(1)
        fig = px.bar(
            x=pct.index.astype(str), y=pct.values,
            labels={"x": "", "y": "% des participants"},
            title="Délai moyen entre achat et voyage (par participant)", text=pct.values,
            color_discrete_sequence=["#2a78d6"],
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, width="stretch")
        st.caption("Calculé sur la moyenne des délais de chaque participant (et non par billet) — "
                   "à lire comme une tendance, pas comme la répartition exacte par billet du rapport source.")

    if anticipation_df is not None:
        st.markdown("**Anticipation par transaction (à partir de la colonne heures)**")
        st.caption("Calculé transaction par transaction (et non moyenné par participant comme "
                   "ci-dessus) à partir de votre colonne d'anticipation en heures, regroupée dans "
                   "les mêmes tranches que le délai en jours ci-dessus (1 jour = 24h).")

        tranche_counts = anticipation_df["tranche_anticipation"].value_counts(dropna=False)
        tranche_counts = tranche_counts.reindex(
            [o for o in DELAI_LABELS if o in tranche_counts.index]
        )
        pct_h = (tranche_counts / tranche_counts.sum() * 100).round(1)
        fig_h = px.bar(
            x=pct_h.index.astype(str), y=pct_h.values,
            labels={"x": "", "y": "% des transactions"},
            title="Anticipation entre achat et voyage (par transaction, à partir des heures)", text=pct_h.values,
            color_discrete_sequence=["#2a78d6"],
        )
        fig_h.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_h, width="stretch")

        st.markdown("**Table détaillée (profil, anticipation)**")
        st.dataframe(anticipation_df.head(50), width="stretch")

        anticipation_excel = to_excel_bytes({"Anticipation": anticipation_df})
        st.download_button(
            "⬇️ Télécharger la table d'anticipation complète (Excel)",
            data=anticipation_excel, file_name="anticipation_heures.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    profil_cols_cramer = [c for c in ["genre", "tranche_age", "situation"] if c in part_df.columns]
    if len(profil_cols_cramer) >= 2:
        st.markdown("**Associations entre variables de profil (V de Cramér corrigé du biais)**")
        mat = matrice_v_cramer(part_df, profil_cols_cramer)
        fig = px.imshow(
            mat.round(2), text_auto=True, color_continuous_scale="Blues", zmin=0, zmax=1,
            labels=dict(color="V de Cramér"),
            x=[CAT_LABELS.get(c, c) for c in mat.columns],
            y=[CAT_LABELS.get(c, c) for c in mat.index],
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("0 = variables indépendantes, 1 = association parfaite. Version corrigée du biais "
                   "(Bergsma 2013), qui compense la sur-estimation liée au nombre de modalités.")

with tab_k:
    st.subheader("Choisir le nombre de segments")
    if X.shape[1] == 0:
        st.warning("Aucune variable sélectionnée.")
    elif show_elbow:
        ks, inertias, sils = compute_elbow_silhouette(X, 2, max_k_possible)
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers"))
            fig.update_layout(title="Méthode du coude", xaxis_title="k",
                               yaxis_title="Inertie intra-cluster")
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig2 = go.Figure(go.Scatter(x=ks, y=sils, mode="lines+markers",
                                         line=dict(color="orange")))
            fig2.update_layout(title="Score de silhouette", xaxis_title="k",
                                yaxis_title="Silhouette moyenne")
            st.plotly_chart(fig2, width="stretch")
        if np.any(~np.isnan(sils)):
            best_k = ks[int(np.nanargmax(sils))]
            st.info(f"💡 Le score de silhouette est maximal pour **k = {best_k}**. "
                    f"Ajustez le slider « k » dans la barre latérale si besoin, puis lancez la segmentation.")
    else:
        st.caption("Cochez la case dans la barre latérale pour afficher les courbes d'aide au choix de k.")

    if X.shape[1] > 0:
        st.markdown("**Table Inertie et Silhouette (k = 2 à k = 8)**")
        k_table_max = min(8, max_k_possible)
        ks_t, inertias_t, sils_t = compute_elbow_silhouette(X, 2, k_table_max)
        table_k = pd.DataFrame({
            "k": ks_t,
            "Inertie": [round(v, 1) for v in inertias_t],
            "Silhouette": [round(v, 3) if not np.isnan(v) else np.nan for v in sils_t],
        })
        st.dataframe(table_k, width="stretch", hide_index=True)
        if k_table_max < 8:
            st.caption(f"Limité à k = {k_table_max} : le nombre de participants ne permet pas d'aller jusqu'à k = 8.")

with tab_kcompare:
    st.subheader("Comparaison des segmentations k = 2 à k = 5")
    st.caption(
        "Cet onglet fonctionne indépendamment du bouton « Lancer la segmentation » : il "
        "recalcule K-Means pour k = 2 à 5 sur les variables actuellement sélectionnées "
        "dans la barre latérale, afin de comparer rapidement plusieurs scénarios de "
        "segmentation avant de choisir un k définitif."
    )

    if X.shape[1] == 0:
        st.warning("Aucune variable sélectionnée dans la barre latérale.")
    else:
        k_results = compute_k_comparison(X, 2, 5)

        recap_rows = []
        for kk_, res in k_results.items():
            sizes = pd.Series(res["labels"]).value_counts().sort_index()
            recap_rows.append({
                "k": kk_,
                "Inertie": round(res["inertia"], 1),
                "Silhouette": round(res["silhouette"], 3) if not np.isnan(res["silhouette"]) else np.nan,
                "Nb participants par segment": " | ".join(f"Seg{i}: {v}" for i, v in sizes.items()),
            })
        recap_df = pd.DataFrame(recap_rows)
        st.markdown("**Tableau récapitulatif**")
        st.dataframe(recap_df, width="stretch", hide_index=True)

        st.markdown("### Élasticité de la taille des segments (k → k+1)")
        show_elasticite = st.toggle(
            "Afficher l'élasticité de la taille des segments",
            value=False, key="show_elasticite_kcompare",
        )
        if show_elasticite:
            st.caption(
                "Variation en % de l'effectif de chaque segment (par indice Seg0, Seg1, ...) entre "
                "deux valeurs de k consécutives. À lire comme un indicateur de stabilité globale des "
                "tailles de segments, pas comme un suivi garanti de la même sous-population : K-Means "
                "recalcule une partition entière à chaque k, l'indice d'un segment n'a donc pas de "
                "correspondance assurée d'un k à l'autre."
            )
            k_list = sorted(k_results.keys())
            elastic_rows = []
            for k_a, k_b in zip(k_list[:-1], k_list[1:]):
                sizes_a = pd.Series(k_results[k_a]["labels"]).value_counts().sort_index()
                sizes_b = pd.Series(k_results[k_b]["labels"]).value_counts().sort_index()
                for seg in sorted(set(sizes_a.index) | set(sizes_b.index)):
                    eff_a = int(sizes_a.get(seg, 0))
                    eff_b = int(sizes_b.get(seg, 0))
                    variation_pct = ((eff_b - eff_a) / eff_a * 100) if eff_a > 0 else np.nan
                    elastic_rows.append({
                        "Transition": f"k={k_a} → k={k_b}",
                        "Segment": f"Seg{seg}",
                        "Effectif (k initial)": eff_a,
                        "Effectif (k final)": eff_b,
                        "Variation (%)": round(variation_pct, 1) if not np.isnan(variation_pct) else np.nan,
                    })
            elastic_df = pd.DataFrame(elastic_rows)

            vue_elasticite = st.radio(
                "Regrouper l'élasticité par :",
                ["Scénario (k → k+1)", "Segment"],
                horizontal=True,
                key="vue_elasticite_kcompare",
            )

            if vue_elasticite == "Scénario (k → k+1)":
                for k_a, k_b in zip(k_list[:-1], k_list[1:]):
                    transition_label = f"k={k_a} → k={k_b}"
                    sub = elastic_df[elastic_df["Transition"] == transition_label]

                    st.markdown(f"**Scénario {transition_label}**")
                    st.dataframe(
                        sub[["Segment", "Effectif (k initial)", "Effectif (k final)", "Variation (%)"]],
                        width="stretch", hide_index=True,
                    )

                    sub_valid = sub.dropna(subset=["Variation (%)"])
                    if not sub_valid.empty:
                        fig = px.bar(
                            sub_valid, x="Segment", y="Variation (%)",
                            title=f"Variation (%) de l'effectif par segment — {transition_label}",
                            color_discrete_sequence=["#2a78d6"],
                        )
                        st.plotly_chart(fig, width="stretch")
                        st.caption(f"Variation moyenne absolue sur ce scénario : "
                                   f"{sub_valid['Variation (%)'].abs().mean():.1f}%")

                    st.divider()
            else:
                segments = sorted(elastic_df["Segment"].unique(), key=lambda s: int(s.replace("Seg", "")))
                for seg in segments:
                    sub = elastic_df[elastic_df["Segment"] == seg]

                    st.markdown(f"**{seg}**")
                    st.dataframe(
                        sub[["Transition", "Effectif (k initial)", "Effectif (k final)", "Variation (%)"]],
                        width="stretch", hide_index=True,
                    )

                    sub_valid = sub.dropna(subset=["Variation (%)"])
                    if not sub_valid.empty:
                        fig = px.bar(
                            sub_valid, x="Transition", y="Variation (%)",
                            title=f"Variation (%) de l'effectif — {seg}, selon la transition",
                            color_discrete_sequence=["#eb6834"],
                        )
                        st.plotly_chart(fig, width="stretch")
                        st.caption(f"Variation moyenne absolue pour {seg} : "
                                   f"{sub_valid['Variation (%)'].abs().mean():.1f}%")

                    st.divider()

            elastic_valid = elastic_df.dropna(subset=["Variation (%)"])
            if not elastic_valid.empty:
                mean_abs_var = elastic_valid["Variation (%)"].abs().mean()
                st.metric("Variation moyenne absolue de l'effectif par segment (toutes transitions confondues)",
                          f"{mean_abs_var:.1f}%")

        st.markdown("### Fiches segments condensées par scénario")
        for kk_, res in k_results.items():
            labels = res["labels"]
            sil = res["silhouette"]
            sil_txt = f"{sil:.3f}" if not np.isnan(sil) else "N/A"

            clustered_k = part_df.copy()
            clustered_k["cluster"] = labels
            cluster_mean_k, z_k = (
                numeric_profile(clustered_k, PROFILE_NUM) if PROFILE_NUM else (pd.DataFrame(), pd.DataFrame())
            )

            # Fichier Excel du scénario : données BRUTES (une ligne par transaction,
            # toutes les colonnes du fichier source) + le numéro de segment du
            # participant correspondant. Généré à la demande (callable) pour ne pas
            # reconstruire les Excel à chaque interaction.
            def _excel_scenario(labels_k=labels, k_val=kk_):
                seg_par_participant = dict(zip(part_df["participant_id"], labels_k))
                export_k = df_raw.copy()
                if colmap.get("payer_email") and not emails_clair:
                    # Données personnelles : l'email payeur est remplacé par un pseudonyme haché.
                    pcol = colmap["payer_email"]
                    export_k[pcol] = pseudonymize_emails(export_k[pcol], email_salt)
                    export_k = export_k.rename(columns={pcol: f"{pcol}_pseudonyme"})
                segment = export_k[id_col].astype(str).map(seg_par_participant)
                export_k.insert(export_k.columns.get_loc(id_col) + 1, "segment", segment)
                return to_excel_bytes({f"Brut_k{k_val}": export_k})

            st.download_button(
                f"⬇️ Télécharger l'Excel — k = {kk_} (données brutes + segment)",
                data=_excel_scenario,
                file_name=f"participants_segmentes_k{kk_}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_scenario_k{kk_}",
            )

            with st.expander(f"k = {kk_} — Inertie = {res['inertia']:.1f} — Silhouette = {sil_txt}"):
                for c in range(kk_):
                    sub = clustered_k[clustered_k["cluster"] == c]
                    pct = len(sub) / len(clustered_k) * 100
                    z_row = z_k.loc[c] if (not z_k.empty and c in z_k.index) else pd.Series(dtype=float)
                    persona, explanation = build_persona_label(z_row)

                    st.markdown(f"**🟢 Segment {c} — {persona} · {len(sub)} participants ({pct:.1f}%)**")
                    st.caption(explanation)

                    if cat_used:
                        sentence = build_profile_sentence(clustered_k, cat_used, c)
                        if sentence:
                            st.markdown(f"_Carte d'identité : {sentence}_")

                        for col in ordered_cat_cols(cat_used):
                            detail = modality_breakdown(clustered_k, col, c)
                            if detail:
                                st.caption(f"{CAT_LABELS.get(col, col)} : {detail}")

                    if reasons_multi is not None:
                        line_r = reasons_mentions_line(reasons_multi, sub["participant_id"], with_global=False)
                        if line_r:
                            st.caption(f"Raisons citées (toutes mentions) : {line_r}")

                    if c < kk_ - 1:
                        st.divider()

# -----------------------------------------------------------------------
# Exécution K-Means (stockée en session pour persister entre les onglets)
# -----------------------------------------------------------------------
if run_btn:
    labels, km_model = run_kmeans(X, k)
    clustered = part_df.copy()
    clustered["cluster"] = labels
    coords2, var_exp2 = compute_pca(X, 2)
    coords3, var_exp3 = compute_pca(X, 3)
    clustered["pca_1"] = coords2[:, 0]
    clustered["pca_2"] = coords2[:, 1]
    clustered["pca_3d_1"] = coords3[:, 0]
    clustered["pca_3d_2"] = coords3[:, 1]
    clustered["pca_3d_3"] = coords3[:, 2]

    st.session_state["part_df_clustered"] = clustered
    st.session_state["numeric_used"] = numeric_used
    st.session_state["cat_used"] = cat_used
    st.session_state["var_exp"] = var_exp2
    st.session_state["var_exp_3d"] = var_exp3
    st.session_state["k"] = k
    st.session_state["inertia"] = km_model.inertia_
    st.session_state["silhouette"] = (
        silhouette_score(X, labels) if 1 < k < len(part_df) else np.nan
    )

with tab_results:
    st.subheader("Résultats de la segmentation")
    if "part_df_clustered" not in st.session_state:
        st.info("👈 Configurez vos variables puis cliquez sur **Lancer la segmentation K-Means** "
                 "dans la barre latérale.")
    else:
        pdf = st.session_state["part_df_clustered"]
        kk = st.session_state["k"]
        var_exp = st.session_state["var_exp"]
        num_used = st.session_state["numeric_used"]
        c_used = st.session_state["cat_used"]

        # Profilage sur TOUTES les variables numériques disponibles (actives + illustratives,
        # dont la récurrence du payeur) : les personas ne dépendent plus du réglage du clustering.
        prof_cols = profile_numeric_cols(pdf)
        cluster_mean, z = numeric_profile(pdf, prof_cols) if prof_cols else (pd.DataFrame(), pd.DataFrame())

        # Les résultats détaillés se masquent / s'affichent ; les fiches segments restent toujours visibles.
        show_details = st.toggle(
            "Afficher les résultats détaillés (métriques, PCA, profils, répartition des variables catégorielles)",
            value=False, key="show_details_results",
        )
        if show_details:
            c1, c2, c3 = st.columns(3)
            c1.metric("Nombre de segments", kk)
            c2.metric("Inertie intra-cluster", f"{st.session_state['inertia']:.1f}")
            sil = st.session_state["silhouette"]
            c3.metric("Score de silhouette", f"{sil:.3f}" if not np.isnan(sil) else "N/A")

            size_df = pdf["cluster"].value_counts().sort_index().reset_index()
            size_df.columns = ["cluster", "effectif"]
            size_df["part_%"] = (size_df["effectif"] / size_df["effectif"].sum() * 100).round(1)

            colA, colB = st.columns([1, 2])
            with colA:
                st.markdown("**Taille des segments**")
                st.dataframe(size_df, width="stretch", hide_index=True)
            with colB:
                fig_bar = px.bar(size_df, x="cluster", y="effectif", text="part_%",
                                  title="Effectif par segment", color="cluster")
                fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_bar, width="stretch")

            hover_cols = ["participant_id"] + [c for c in num_used if c in pdf.columns]

            col2d, col3d = st.columns(2)
            with col2d:
                st.markdown("**Projection PCA — 2 dimensions**")
                fig_pca2 = px.scatter(pdf, x="pca_1", y="pca_2", color=pdf["cluster"].astype(str),
                                       hover_data=hover_cols, labels={"color": "Segment"})
                var_txt2 = f"{var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% = {(var_exp[0]+var_exp[1])*100:.1f}%"
                fig_pca2.update_layout(title=f"Variance expliquée : {var_txt2}",
                                        margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_pca2, width="stretch")

            with col3d:
                st.markdown("**Projection PCA — 3 dimensions**")
                var_exp_3d = st.session_state.get("var_exp_3d", np.zeros(3))
                fig_pca3 = px.scatter_3d(
                    pdf, x="pca_3d_1", y="pca_3d_2", z="pca_3d_3",
                    color=pdf["cluster"].astype(str), hover_data=hover_cols,
                    labels={"color": "Segment", "pca_3d_1": "PC1", "pca_3d_2": "PC2", "pca_3d_3": "PC3"},
                )
                var_txt3 = (f"{var_exp_3d[0]*100:.1f}% + {var_exp_3d[1]*100:.1f}% + {var_exp_3d[2]*100:.1f}% "
                            f"= {var_exp_3d.sum()*100:.1f}%")
                fig_pca3.update_layout(title=f"Variance expliquée : {var_txt3}",
                                        margin=dict(l=0, r=0, t=40, b=0))
                fig_pca3.update_traces(marker=dict(size=4))
                st.plotly_chart(fig_pca3, width="stretch")


            if prof_cols:
                st.markdown("**Profil moyen des segments (variables numériques)**")
                display_mean = cluster_mean.rename(columns=NUMERIC_LABELS).round(1)
                st.dataframe(display_mean, width="stretch")

                fig_heat = px.imshow(
                    z[prof_cols].T, aspect="auto", color_continuous_scale="RdBu_r",
                    y=[NUMERIC_LABELS.get(c, c) + (" ✓" if c in num_used else "") for c in prof_cols],
                    labels=dict(x="Segment", y="", color="Écart (z-score)"),
                    title="Écart de chaque segment à la moyenne globale (z-score)",
                )
                st.plotly_chart(fig_heat, width="stretch")
                st.caption("✓ = variable utilisée dans le clustering ; les autres sont illustratives "
                           "(elles décrivent les segments sans les avoir construits).")

            st.divider()
            st.subheader("Interprétation marketing des segments")

            if c_used:
                st.markdown("### Répartition des variables catégorielles par segment")
                for col in c_used:
                    tab_pct = categorical_profile(pdf, col)
                    fig = px.imshow(
                        tab_pct, aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x=CAT_LABELS.get(col, col), y="Segment", color="%"),
                        title=f"{CAT_LABELS.get(col, col)} — répartition (%) par segment",
                    )
                    st.plotly_chart(fig, width="stretch")

        st.markdown("### Fiches segments")
        for c in range(kk):
            sub = pdf[pdf["cluster"] == c]
            pct = len(sub) / len(pdf) * 100
            z_row = z.loc[c] if (not z.empty and c in z.index) else pd.Series(dtype=float)
            persona, explanation = build_persona_label(z_row)

            with st.expander(f"🟢 Segment {c} — {persona}  ·  {len(sub)} participants ({pct:.1f}%)",
                              expanded=(c == 0)):
                if c_used:
                    sentence = build_profile_sentence(pdf, c_used, c)
                    if sentence:
                        st.markdown(f"**Carte d'identité du segment (profil majoritaire) :** _{sentence}_")

                    if st.checkbox("Afficher la répartition complète par variable "
                                   "(toutes les modalités du segment)", key=f"detail_cat_{c}"):
                        for col in ordered_cat_cols(c_used):
                            detail = modality_breakdown(pdf, col, c)
                            if detail:
                                st.markdown(f"- **{CAT_LABELS.get(col, col)}** : {detail}")

with tab_recurrence:
    st.subheader("Récurrence d'achat par email payeur")
    if not colmap.get("payer_email"):
        st.info("👈 Associez la colonne **Email du payeur** (ex. `PayerEmail`) dans « 🧩 Correspondance "
                "des colonnes » de la barre latérale pour mesurer la récurrence d'achat.")
    elif not colmap.get("date"):
        st.warning("Associez aussi la **Date d'achat** : la récurrence se calcule sur les dates de commande.")
    elif payer_df is None or payer_df.empty:
        st.warning("Aucun email valide associé à une date d'achat n'a été trouvé : vérifiez les "
                   "colonnes choisies dans la barre latérale.")
    else:
        q = payer_quality
        if colmap.get("order_ref"):
            cmd_def = f"1 email + 1 {colmap['order_ref']}"
            cmd_expl = (f"Une **commande** = un email + un **{colmap['order_ref']}** "
                        "(plusieurs lignes avec le même code = 1 commande). ")
        else:
            cmd_def = "1 email + 1 jour"
            cmd_expl = "Une **commande** = un email + un jour d'achat (5 billets achetés d'un coup = 1 commande). "
        st.caption(cmd_expl + f"Période observée : {q['date_min']:%d/%m/%Y} → {q['date_max']:%d/%m/%Y}.")

        with st.expander("Domaines les plus fréquents (repérer fautes de frappe et emails génériques)"):
            st.dataframe(
                pd.DataFrame({"Domaine": list(q["domaines_top"]), "Lignes": list(q["domaines_top"].values())}),
                hide_index=True, width="stretch",
            )

        n_interm = int(payer_df["intermediaire_probable"].sum())
        excl_interm = st.checkbox(
            f"Exclure les intermédiaires probables ({n_interm} email(s) payant pour "
            f"≥ {seuil_intermediaire} participants)",
            value=True, key="rec_excl_interm",
            help="Une agence ou une entreprise qui paie pour de nombreux participants fausserait "
                 "les statistiques de récurrence. Le seuil se règle dans la barre latérale.",
        )
        pv = payer_df[~payer_df["intermediaire_probable"]] if excl_interm else payer_df

        if pv.empty:
            st.warning("Aucun payeur à afficher avec ces réglages.")
        else:
            unite = st.radio(
                "Compter :",
                ["Billets (chaque ligne = 1 achat)", f"Commandes ({cmd_def} = 1 commande)"],
                horizontal=True, key="rec_unite",
                help="« Billets » mesure la taille des achats : 3 lignes comptent pour 3. "
                     "« Commandes » mesure le retour du client : plusieurs lignes d'une même commande "
                     f"({cmd_def}) comptent pour 1, seule une nouvelle commande fait un récurrent.",
            )
            col_n = "n_billets" if unite.startswith("Billets") else "n_commandes"
            mot = "achats" if col_n == "n_billets" else "commandes"

            if "tranche_fichier" in pv.columns:
                # Tranches déjà calculées dans le fichier (colonne associée dans la barre latérale).
                # Les payeurs sans tranche (« - » ou vide) ne sont pas affichés dans ces graphiques.
                cls = pv["tranche_fichier"].dropna().value_counts()
                cls = cls.reindex(sorted(cls.index, key=tranche_sort_key)).astype(int)
                titre_tr = f"Tranche « {q['tranche_fichier']} »"
                n_rec = int(cls.iloc[1:].sum())
                st.caption(f"Tranches reprises de la colonne **{q['tranche_fichier']}** du fichier (non "
                           "recalculées ; le sélecteur « Compter » ne s'applique pas à ces deux graphiques). "
                           f"**{n_rec / cls.sum() * 100:.1f} %** des payeurs ne sont pas dans la première tranche "
                           f"« {cls.index[0]} » ({fmt_int(n_rec)} sur {fmt_int(int(cls.sum()))}).")
            else:
                bornes = [0, 1, 3, 5, 10, np.inf]
                tranches = [f"1 {mot[:-1]}", f"2-3 {mot}", f"4-5 {mot}", f"6-10 {mot}", f"+10 {mot}"]
                cls = (pd.cut(pv[col_n], bornes, labels=tranches).value_counts()
                         .reindex(tranches).fillna(0).astype(int))
                titre_tr = f"Tranche de récurrence ({mot})"
                n_rec = int((pv[col_n] > 1).sum())
                st.caption(f"**{n_rec / len(pv) * 100:.1f} %** des payeurs ont plus "
                           f"{'d’un achat' if col_n == 'n_billets' else 'd’une commande'} "
                           f"({fmt_int(n_rec)} sur {fmt_int(len(pv))}).")
            pct_cls = (cls / cls.sum() * 100).round(1)

            colA, colB = st.columns(2)
            with colA:
                fig_n = px.bar(
                    x=cls.index.astype(str), y=cls.values, text=cls.values,
                    labels={"x": "", "y": "Nb de payeurs"},
                    title=f"{titre_tr} — nombre de payeurs",
                    color_discrete_sequence=["#2a78d6"],
                )
                fig_n.update_traces(textposition="outside")
                st.plotly_chart(fig_n, width="stretch")
            with colB:
                fig_p = px.bar(
                    x=pct_cls.index.astype(str), y=pct_cls.values, text=pct_cls.values,
                    labels={"x": "", "y": "% des payeurs"},
                    title=f"{titre_tr} — % des payeurs",
                    color_discrete_sequence=["#eb6834"],
                )
                fig_p.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_p, width="stretch")

            ov = orders_df[orders_df["email"].isin(pv.index)]

            # ---- 1. Billets par commande -------------------------------------------
            st.markdown("**Billets par commande**")
            cnt_b = ov["n_billets"].clip(upper=5).value_counts().reindex([1, 2, 3, 4, 5]).fillna(0).astype(int)
            pct_b = (cnt_b / cnt_b.sum() * 100).round(1)
            fig_b = px.bar(
                x=["1 billet", "2 billets", "3 billets", "4 billets", "5+ billets"], y=cnt_b.values,
                text=[f"{n} ({p}%)" for n, p in zip(cnt_b.values, pct_b.values)],
                labels={"x": "", "y": "Nb de commandes"},
                title=f"Taille des commandes (1 commande = {cmd_def})",
                color_discrete_sequence=["#2a78d6"],
            )
            fig_b.update_traces(textposition="outside")
            st.plotly_chart(fig_b, width="stretch")
            st.caption(f"**{(cnt_b.iloc[1:].sum() / cnt_b.sum() * 100):.1f} %** des commandes contiennent "
                       "plusieurs billets (famille, amis, groupe) : c'est ce qui sépare le comptage en "
                       "billets du comptage en commandes.")

            # ---- 2. Nouveaux vs récurrents par semaine -------------------------------
            st.markdown("**Nouveaux clients vs clients récurrents, par semaine**")
            ow = ov.copy()
            ow["Type"] = np.where(ow["jour"] == ow.groupby("email")["jour"].transform("min"),
                                  "Nouveau client", "Client récurrent")
            ow["Semaine"] = ow["jour"].dt.to_period("W-SUN").dt.start_time
            ow["valeur"] = ow["n_billets"] if col_n == "n_billets" else 1
            wk = ow.groupby(["Semaine", "Type"])["valeur"].sum().reset_index()
            fig_w = px.bar(
                wk, x="Semaine", y="valeur", color="Type", barmode="stack",
                labels={"valeur": "Nb de billets" if col_n == "n_billets" else "Nb de commandes"},
                title=f"{'Billets' if col_n == 'n_billets' else 'Commandes'} par semaine : nouveaux vs récurrents",
                color_discrete_map={"Nouveau client": "#2a78d6", "Client récurrent": "#eb6834"},
                category_orders={"Type": ["Nouveau client", "Client récurrent"]},
            )
            st.plotly_chart(fig_w, width="stretch")
            st.caption("« Nouveau client » = première commande de l'email dans le fichier. Les premières "
                       "semaines paraissent entièrement « nouvelles » simplement parce que l'historique "
                       "d'avant le fichier n'est pas connu.")

            st.markdown("**Tous les payeurs**")
            allp = pv.reset_index()[["email", "n_commandes", "n_participants_payes",
                                     "derniere_commande"]].copy()
            allp = allp.sort_values("n_commandes", ascending=False)
            allp = allp.rename(columns={"n_commandes": "nombre des Billets",
                                        "n_participants_payes": "nombre de participants par email"})
            allp["email"] = allp["email"].astype(object) if emails_clair else mask_emails(allp["email"])
            allp["derniere_commande"] = allp["derniere_commande"].dt.date
            st.caption(f"{fmt_int(len(allp))} payeurs, triés par nombre de billets. "
                       "Les emails sont masqués par défaut (case « Afficher / exporter les emails en clair » "
                       "dans la barre latérale).")
            st.dataframe(allp, hide_index=True, width="stretch")

with tab_export:
    st.subheader("Export des résultats")
    if "part_df_clustered" not in st.session_state:
        st.info("👈 Lancez d'abord la segmentation dans la barre latérale.")
    else:
        pdf = st.session_state["part_df_clustered"]
        num_used = st.session_state["numeric_used"]
        c_used = st.session_state["cat_used"]
        kk = st.session_state["k"]

        # Colonnes de récurrence (si disponibles) : pratiques pour cibler les segments en CRM.
        # `payeur_id` est un pseudonyme ; `email_principal` n'existe que si l'option « emails
        # en clair » est cochée dans la barre latérale.
        extra_cols = [c for c in ["payeur_id", "email_principal", "n_commandes_payeur",
                                   "classe_recurrence_payeur"] if c in pdf.columns]
        export_cols = ["participant_id", "cluster"] + num_used + c_used + extra_cols
        export_cols = list(dict.fromkeys(c for c in export_cols if c in pdf.columns))
        export_df = pdf[export_cols]

        st.markdown("**Aperçu du fichier à exporter**")
        st.dataframe(export_df.head(30), width="stretch")

        prof_cols_exp = profile_numeric_cols(pdf)
        if prof_cols_exp:
            cluster_mean, _ = numeric_profile(pdf, prof_cols_exp)
            summary_df = cluster_mean.round(1)
        else:
            summary_df = pd.DataFrame(index=range(kk))
        summary_df.insert(0, "effectif", pdf["cluster"].value_counts().sort_index())
        summary_df.insert(1, "part_%", (summary_df["effectif"] / summary_df["effectif"].sum() * 100).round(1))

        excel_bytes = to_excel_bytes({
            "Participants_segmentes": export_df,
            "Profil_segments": summary_df.reset_index(),
        })

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Télécharger le fichier Excel", data=excel_bytes,
                file_name="participants_segmentes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        with col2:
            csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Télécharger le fichier CSV", data=csv_bytes,
                file_name="participants_segmentes.csv", mime="text/csv",
                width="stretch",
            )
