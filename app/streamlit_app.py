"""Application Streamlit - Telecom Customer Churn Prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "telco_customer_churn.csv"
TEST_X_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
TEST_Y_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"

THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.60

# Les 19 variables utilisées pendant l'entraînement.
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Palette
NAVY = "#0F2540"
BLUE = "#1F6FEB"
TEAL = "#14B8A6"
RED = "#E5484D"
ORANGE = "#F59E0B"
GREEN = "#22A06B"
GREY = "#64748B"
LIGHT_BG = "#F5F7FB"


# =========================================================
# STYLE
# =========================================================

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        /* Bandeau d'en-tête */
        .app-header {{
            background: linear-gradient(120deg, {NAVY} 0%, #1B3F73 60%, {BLUE} 100%);
            padding: 1.6rem 2rem;
            border-radius: 16px;
            color: #fff;
            margin-bottom: 1.4rem;
            box-shadow: 0 6px 18px rgba(15, 37, 64, 0.18);
        }}
        .app-header h1 {{
            color: #fff;
            font-size: 1.8rem;
            margin: 0 0 .25rem 0;
            padding: 0;
        }}
        .app-header p {{
            color: #CBD7EA;
            margin: 0;
            font-size: .95rem;
        }}

        /* Titres de section */
        .section-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {NAVY};
            margin: 1.2rem 0 .6rem 0;
            padding-left: .7rem;
            border-left: 4px solid {BLUE};
        }}

        /* Cartes KPI */
        div[data-testid="stMetric"] {{
            background: #fff;
            border: 1px solid #E3E8F0;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: 0 2px 8px rgba(15, 37, 64, 0.05);
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {GREY};
            font-weight: 600;
            font-size: .85rem;
            text-transform: uppercase;
            letter-spacing: .03em;
        }}
        div[data-testid="stMetricValue"] {{
            color: {NAVY};
            font-weight: 700;
        }}

        /* Conteneurs de graphiques */
        div[data-testid="stPlotlyChart"] {{
            background: #fff;
            border: 1px solid #E3E8F0;
            border-radius: 14px;
            padding: .5rem;
            box-shadow: 0 2px 8px rgba(15, 37, 64, 0.04);
        }}

        /* Formulaire */
        div[data-testid="stForm"] {{
            background: #fff;
            border: 1px solid #E3E8F0;
            border-radius: 14px;
            padding: 1.4rem;
        }}

        /* Boutons */
        .stButton > button, .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button {{
            background: {BLUE};
            color: #fff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: .6rem 1rem;
            transition: all .15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {{
            background: {NAVY};
            color: #fff;
            transform: translateY(-1px);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {NAVY};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E6EDF8 !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,.15);
        }}

        /* Badges de risque */
        .risk-card {{
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            margin: .6rem 0 1rem 0;
            border: 1px solid;
        }}
        .risk-high {{ background:#FDECEC; border-color:{RED}; color:#8A1C20; }}
        .risk-mid  {{ background:#FEF4E0; border-color:{ORANGE}; color:#8A5A00; }}
        .risk-low  {{ background:#E7F6EF; border-color:{GREEN}; color:#14573B; }}
        .risk-card h4 {{ margin:0 0 .3rem 0; font-size:1.05rem; }}
        .risk-card p  {{ margin:0; font-size:.93rem; }}

        .note {{
            color: {GREY};
            font-size: .82rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Applique un style homogène à tous les graphiques Plotly."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=NAVY),
        title=dict(font=dict(size=15, color=NAVY)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#EDF1F7", zeroline=False)
    return fig


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>Telecom Customer Churn Prediction</h1>
            <p>Plateforme de Business Intelligence et de Machine Learning
            pour l'analyse et la prédiction de la résiliation des clients.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_banner(level: str, title: str, text: str) -> None:
    css = {"high": "risk-high", "mid": "risk-mid", "low": "risk-low"}[level]
    st.markdown(
        f'<div class="risk-card {css}"><h4>{title}</h4><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": " %", "font": {"size": 42, "color": NAVY}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": GREY},
                "bar": {"color": NAVY, "thickness": 0.28},
                "steps": [
                    {"range": [0, THRESHOLD * 100], "color": "#D5F0E3"},
                    {
                        "range": [THRESHOLD * 100, HIGH_RISK_THRESHOLD * 100],
                        "color": "#FCE8BF",
                    },
                    {"range": [HIGH_RISK_THRESHOLD * 100, 100], "color": "#F8D0D2"},
                ],
                "threshold": {
                    "line": {"color": RED, "width": 4},
                    "thickness": 0.8,
                    "value": THRESHOLD * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    return fig


inject_css()


# =========================================================
# CHARGEMENT DES RESSOURCES
# =========================================================

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def load_data(data_path: str):
    return pd.read_csv(data_path)


@st.cache_data
def load_test_data(x_path: str, y_path: str):
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze("columns")
    return X_test, y_test


if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Impossible de charger le modèle : {error}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset introuvable : {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Impossible de charger le dataset : {error}")
    st.stop()

missing_dataset_columns = [
    column for column in ["Churn"] + FEATURE_COLUMNS if column not in df.columns
]
if missing_dataset_columns:
    st.error(
        "Le dataset ne contient pas toutes les colonnes attendues : "
        + ", ".join(missing_dataset_columns)
    )
    st.stop()

if "Yes" not in list(model.classes_):
    st.error("La classe 'Yes' est introuvable dans le modèle sauvegardé.")
    st.stop()

YES_INDEX = list(model.classes_).index("Yes")


# =========================================================
# NAVIGATION
# =========================================================

st.sidebar.markdown("## 📊 Telecom Churn")
st.sidebar.caption("Plateforme d'analyse et de prédiction")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔎 Prédiction individuelle",
        "📂 Prédiction en masse",
        "🤖 Performance du modèle",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("**Informations modèle**")
st.sidebar.caption("Algorithme : Gradient Boosting")
st.sidebar.caption(f"Seuil de décision : {THRESHOLD:.0%}")
st.sidebar.caption(f"Variables utilisées : {len(FEATURE_COLUMNS)}")

render_header()


# =========================================================
# PAGE 1 - DASHBOARD
# =========================================================

if page == "🏠 Dashboard":
    st.header("Dashboard — Vue générale")

    total_customers = len(df)
    churn_customers = int((df["Churn"] == "Yes").sum())
    retained_customers = int((df["Churn"] == "No").sum())
    churn_rate = (
        churn_customers / total_customers * 100 if total_customers else 0
    )
    retention_rate = (
        retained_customers / total_customers * 100 if total_customers else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total clients", f"{total_customers:,}")
    col2.metric("Clients résiliés", f"{churn_customers:,}")
    col3.metric("Clients fidèles", f"{retained_customers:,}")
    col4.metric("Taux de résiliation", f"{churn_rate:.2f}%")

    section_title("Répartition et contrats")
    left, right = st.columns(2)

    with left:
        churn_distribution = (
            df["Churn"]
            .value_counts()
            .rename_axis("Statut")
            .reset_index(name="Nombre")
        )
        fig_pie = px.pie(
            churn_distribution,
            names="Statut",
            values="Nombre",
            hole=0.55,
            title="Clients résiliés et fidèles",
            color="Statut",
            color_discrete_map={"No": TEAL, "Yes": RED},
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
        st.plotly_chart(style_fig(fig_pie), use_container_width=True)

    with right:
        contract_rate = (
            df.groupby("Contract", observed=True)["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .reset_index(name="Churn_Rate")
        )
        fig_contract = px.bar(
            contract_rate,
            x="Contract",
            y="Churn_Rate",
            text_auto=".2f",
            title="Taux de résiliation selon le contrat",
            labels={
                "Contract": "Type de contrat",
                "Churn_Rate": "Taux de résiliation (%)",
            },
            color_discrete_sequence=[BLUE],
        )
        fig_contract.update_traces(marker_line_width=0, textposition="outside")
        st.plotly_chart(style_fig(fig_contract), use_container_width=True)

    section_title("Méthode de paiement et ancienneté")
    left2, right2 = st.columns(2)

    with left2:
        payment_rate = (
            df.groupby("PaymentMethod", observed=True)["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .reset_index(name="Churn_Rate")
            .sort_values("Churn_Rate", ascending=False)
        )
        fig_payment = px.bar(
            payment_rate,
            x="PaymentMethod",
            y="Churn_Rate",
            text_auto=".2f",
            title="Taux de résiliation par méthode de paiement",
            labels={
                "PaymentMethod": "Méthode de paiement",
                "Churn_Rate": "Taux de résiliation (%)",
            },
            color_discrete_sequence=[NAVY],
        )
        fig_payment.update_traces(marker_line_width=0, textposition="outside")
        st.plotly_chart(style_fig(fig_payment), use_container_width=True)

    with right2:
        tenure_df = df.copy()
        tenure_df["Tenure_Group"] = pd.cut(
            tenure_df["tenure"],
            bins=[-1, 12, 24, 36, 48, 60, 72],
            labels=[
                "0-12 mois",
                "13-24 mois",
                "25-36 mois",
                "37-48 mois",
                "49-60 mois",
                "61-72 mois",
            ],
        )
        tenure_rate = (
            tenure_df.groupby("Tenure_Group", observed=True)["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .reset_index(name="Churn_Rate")
        )
        fig_tenure = px.line(
            tenure_rate,
            x="Tenure_Group",
            y="Churn_Rate",
            markers=True,
            title="Taux de résiliation selon l'ancienneté",
            labels={
                "Tenure_Group": "Ancienneté",
                "Churn_Rate": "Taux de résiliation (%)",
            },
            color_discrete_sequence=[RED],
        )
        fig_tenure.update_traces(line=dict(width=3), marker=dict(size=9))
        st.plotly_chart(style_fig(fig_tenure), use_container_width=True)

    st.markdown(
        f'<p class="note">Taux de fidélisation observé dans le dataset : '
        f"{retention_rate:.2f}%. Ces statistiques décrivent les données "
        "historiques et ne démontrent pas une relation de causalité.</p>",
        unsafe_allow_html=True,
    )


# =========================================================
# PAGE 2 - PREDICTION INDIVIDUELLE
# =========================================================

elif page == "🔎 Prédiction individuelle":
    st.header("Prédiction individuelle")
    st.write(
        "Renseigne les caractéristiques d'un client pour estimer "
        "sa probabilité de résiliation."
    )

    with st.form("customer_form"):
        st.markdown("##### Profil et abonnement du client")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Genre", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input(
                "Ancienneté (mois)", min_value=0, max_value=72, value=12
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input(
                "Monthly Charges", min_value=0.0, value=70.0
            )
            total = st.number_input(
                "Total Charges", min_value=0.0, value=840.0
            )

        submitted = st.form_submit_button(
            "Prédire le risque de résiliation",
            use_container_width=True,
        )

    if submitted:
        customer = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly,
                    "TotalCharges": total,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        probability = float(model.predict_proba(customer)[0, YES_INDEX])
        prediction = "Yes" if probability >= THRESHOLD else "No"

        section_title("Résultat de la prédiction")

        gauge_col, result_col = st.columns([1, 1])

        with gauge_col:
            st.plotly_chart(
                probability_gauge(probability), use_container_width=True
            )

        with result_col:
            st.metric(
                "Probabilité estimée de résiliation", f"{probability:.2%}"
            )
            if prediction == "Yes":
                level = "high" if probability >= HIGH_RISK_THRESHOLD else "mid"
                risk_banner(
                    level,
                    "Client identifié comme à risque",
                    "Contacter le client, comprendre ses difficultés et "
                    "proposer une offre de fidélisation adaptée.",
                )
            else:
                risk_banner(
                    "low",
                    "Client non identifié comme à risque",
                    "Continuer à suivre la satisfaction du client et "
                    "maintenir une relation commerciale de qualité.",
                )

        st.markdown(
            f'<p class="note">Seuil de décision : {THRESHOLD:.0%}. '
            "La probabilité est une estimation du modèle, pas une "
            "certitude.</p>",
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE 3 - PREDICTION EN MASSE
# =========================================================

elif page == "📂 Prédiction en masse":
    st.header("Prédiction en masse")
    st.write(
        "Importe un fichier CSV contenant les informations de plusieurs "
        "clients. Le fichier doit inclure les 19 variables utilisées "
        "pendant l'entraînement."
    )

    st.info(
        "La colonne Churn n'est pas nécessaire. Les colonnes supplémentaires "
        "(par exemple customerID) seront conservées dans le fichier de sortie "
        "mais ne seront pas utilisées par le modèle."
    )

    with st.expander("Voir la liste des colonnes attendues"):
        st.code(", ".join(FEATURE_COLUMNS), language="text")

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"],
        key="bulk_csv",
    )

    if uploaded_file is not None:
        try:
            customers_df = pd.read_csv(uploaded_file)

            missing_columns = [
                column
                for column in FEATURE_COLUMNS
                if column not in customers_df.columns
            ]

            if missing_columns:
                st.error(
                    "Colonnes manquantes : " + ", ".join(missing_columns)
                )
            elif customers_df.empty:
                st.warning("Le fichier CSV ne contient aucune ligne.")
            else:
                X_customers = customers_df[FEATURE_COLUMNS].copy()

                # Certains CSV Telco contiennent TotalCharges sous forme
                # de texte ou avec des cellules vides.
                X_customers["TotalCharges"] = pd.to_numeric(
                    X_customers["TotalCharges"], errors="coerce"
                )

                probabilities = model.predict_proba(X_customers)[:, YES_INDEX]
                predictions = np.where(
                    probabilities >= THRESHOLD, "Yes", "No"
                )

                results_df = customers_df.copy()
                results_df["Churn_Probability"] = (
                    probabilities * 100
                ).round(2)
                results_df["Churn_Prediction"] = predictions
                results_df["Risk_Level"] = np.select(
                    [
                        probabilities < THRESHOLD,
                        probabilities < HIGH_RISK_THRESHOLD,
                    ],
                    ["Faible", "Moyen"],
                    default="Élevé",
                )

                churn_count = int((predictions == "Yes").sum())
                risk_rate = float((predictions == "Yes").mean() * 100)

                st.success(
                    f"{len(results_df):,} clients analysés avec succès."
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Clients analysés", f"{len(results_df):,}")
                col2.metric("Clients identifiés à risque", f"{churn_count:,}")
                col3.metric("Part identifiée à risque", f"{risk_rate:.2f}%")

                section_title("Répartition des niveaux de risque")
                order = ["Faible", "Moyen", "Élevé"]
                risk_counts = (
                    results_df["Risk_Level"]
                    .value_counts()
                    .reindex(order, fill_value=0)
                    .rename_axis("Niveau de risque")
                    .reset_index(name="Clients")
                )
                chart_left, chart_right = st.columns([1, 1])
                with chart_left:
                    fig_risk = px.bar(
                        risk_counts,
                        x="Niveau de risque",
                        y="Clients",
                        text_auto=True,
                        color="Niveau de risque",
                        color_discrete_map={
                            "Faible": GREEN,
                            "Moyen": ORANGE,
                            "Élevé": RED,
                        },
                        title="Nombre de clients par niveau de risque",
                    )
                    fig_risk.update_layout(showlegend=False)
                    st.plotly_chart(
                        style_fig(fig_risk, 320), use_container_width=True
                    )
                with chart_right:
                    fig_hist = px.histogram(
                        results_df,
                        x="Churn_Probability",
                        nbins=20,
                        title="Distribution des probabilités de résiliation",
                        labels={"Churn_Probability": "Probabilité (%)"},
                        color_discrete_sequence=[BLUE],
                    )
                    fig_hist.add_vline(
                        x=THRESHOLD * 100,
                        line_dash="dash",
                        line_color=RED,
                        annotation_text="Seuil",
                    )
                    st.plotly_chart(
                        style_fig(fig_hist, 320), use_container_width=True
                    )

                section_title("Résultats des prédictions")
                selected_levels = st.multiselect(
                    "Filtrer par niveau de risque",
                    options=order,
                    default=order,
                )
                filtered_df = results_df[
                    results_df["Risk_Level"].isin(selected_levels)
                ].sort_values("Churn_Probability", ascending=False)
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Churn_Probability": st.column_config.ProgressColumn(
                            "Churn_Probability",
                            format="%.2f %%",
                            min_value=0,
                            max_value=100,
                        )
                    },
                )
                st.caption(
                    f"{len(filtered_df):,} ligne(s) affichée(s) sur "
                    f"{len(results_df):,}. Le fichier téléchargé contient "
                    "tous les clients."
                )

                csv_output = results_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 Télécharger les résultats CSV",
                    data=csv_output,
                    file_name="telecom_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Erreur lors du traitement du fichier : {error}")


# =========================================================
# PAGE 4 - PERFORMANCE DU MODELE
# =========================================================

elif page == "🤖 Performance du modèle":
    st.header("Performance du modèle")
    st.write(
        "Évaluation du pipeline Gradient Boosting sur le jeu de test "
        "exporté depuis le notebook."
    )

    if not TEST_X_PATH.exists() or not TEST_Y_PATH.exists():
        st.warning(
            "Les fichiers de test sont introuvables. Vérifie la présence de "
            "`data/processed/X_test.csv` et `data/processed/y_test.csv`."
        )
        st.stop()

    try:
        X_test_app, y_test_app = load_test_data(
            str(TEST_X_PATH), str(TEST_Y_PATH)
        )

        missing_test_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in X_test_app.columns
        ]
        if missing_test_columns:
            st.error(
                "Colonnes manquantes dans X_test.csv : "
                + ", ".join(missing_test_columns)
            )
            st.stop()

        X_test_app = X_test_app[FEATURE_COLUMNS]
        y_test_proba = model.predict_proba(X_test_app)[:, YES_INDEX]
        y_test_pred = np.where(
            y_test_proba >= THRESHOLD, "Yes", "No"
        )

        accuracy = accuracy_score(y_test_app, y_test_pred)
        precision = precision_score(
            y_test_app, y_test_pred, pos_label="Yes", zero_division=0
        )
        recall = recall_score(
            y_test_app, y_test_pred, pos_label="Yes", zero_division=0
        )
        f1 = f1_score(
            y_test_app, y_test_pred, pos_label="Yes", zero_division=0
        )
        roc_auc = roc_auc_score(
            (y_test_app == "Yes").astype(int), y_test_proba
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")
        col4.metric("F1-score", f"{f1:.2%}")
        col5.metric("ROC-AUC", f"{roc_auc:.2%}")

        st.markdown(
            f'<p class="note">Seuil de décision utilisé : '
            f"{THRESHOLD:.2f}</p>",
            unsafe_allow_html=True,
        )

        section_title("Matrice de confusion et courbe ROC")
        cm_col, roc_col = st.columns(2)

        with cm_col:
            cm = confusion_matrix(
                y_test_app, y_test_pred, labels=["No", "Yes"]
            )
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Prédit No", "Prédit Yes"],
                y=["Réel No", "Réel Yes"],
                color_continuous_scale=[[0, "#EAF1FD"], [1, BLUE]],
                labels={
                    "x": "Prédiction",
                    "y": "Valeur réelle",
                    "color": "Nombre de clients",
                },
                title="Matrice de confusion",
            )
            fig_cm.update_traces(textfont_size=18)
            fig_cm.update_coloraxes(showscale=False)
            st.plotly_chart(style_fig(fig_cm, 420), use_container_width=True)

        with roc_col:
            fpr, tpr, _ = roc_curve(
                (y_test_app == "Yes").astype(int), y_test_proba
            )
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"Modèle (AUC = {roc_auc:.3f})",
                    line=dict(color=BLUE, width=3),
                    fill="tozeroy",
                    fillcolor="rgba(31,111,235,0.10)",
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Hasard",
                    line=dict(color=GREY, dash="dash"),
                )
            )
            fig_roc.update_layout(
                title=f"Courbe ROC (AUC = {roc_auc:.3f})",
                xaxis_title="Taux de faux positifs",
                yaxis_title="Taux de vrais positifs",
            )
            st.plotly_chart(style_fig(fig_roc, 420), use_container_width=True)

        st.info(
            "Note méthodologique : ce jeu de test a déjà été consulté lors "
            "des premières évaluations du projet. Ces résultats sont donc "
            "indicatifs et ne constituent pas une validation totalement "
            "indépendante."
        )

    except Exception as error:
        st.error(f"Erreur pendant l'évaluation du modèle : {error}")