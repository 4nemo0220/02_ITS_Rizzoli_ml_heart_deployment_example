# heart_ml_app.py
"""
Heart Disease – ML Demo

Piccola applicazione Streamlit che:
- carica un dataset pubblico su malattia cardiaca
- allena un modello binario (malattia sì/no)
- mostra alcune metriche di performance
- permette di fare una previsione per un singolo paziente
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# *PAGE CONFIG -------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease – ML Demo",
    layout="wide",
    page_icon="❤️",
)

# *VARS  -------------------------------------------------------------
# Data path
DATA_PATH = Path("data/heart.csv")

# Colonne selezionate come feature
FEATURE_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# Etichette leggibili in italiano per l'UI
FEATURE_LABELS = {
    "age": "Età (anni)",
    "trestbps": "Pressione a riposo (mm Hg)",
    "chol": "Colesterolo (mg/dl)",
    "thalch": "Freq. cardiaca max (bpm)",
    "oldpeak": "Depressione ST (oldpeak)",
}
TARGET_LABEL = "Presenza di malattia (target)"

# *UTILS -------------------------------------------------------------


@st.cache_data
def load_data() -> pd.DataFrame:
    """Legge il csv e crea la colonna target binaria."""
    df = pd.read_csv(DATA_PATH)

    # num: 0 = sano, 1–4 = malattia
    df["target"] = (df["num"] > 0).astype(int)

    cols = FEATURE_COLS + ["target"]
    return df[cols]


@st.cache_resource
def train_model(
    df: pd.DataFrame,
    max_depth: int,
    min_samples_leaf: int,
):
    """
    Allena un RandomForest e calcola:
    - accuracy su train e test
    - baseline (classe più frequente)
    - importanza delle feature
    """
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # baseline: predire sempre la classe più frequente
    majority_class = int(y_test.value_counts().idxmax())
    baseline_pred = [majority_class] * len(y_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    metrics = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "baseline_acc": baseline_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # importanza delle feature con etichette italiane
    fi = pd.Series(
        model.feature_importances_,
        index=[FEATURE_LABELS[c] for c in FEATURE_COLS],
    ).sort_values(ascending=True)

    return model, metrics, fi


######################################################################
# ------------------------------ ST APP ----------------------------- #
######################################################################

df = load_data()

st.title("❤️ Heart Disease – ML Demo")
st.caption(
    "Esempio didattico: modello binario (malattia sì/no) su 5 feature "
    "numeriche. Non è uno strumento medico reale."
)

# *SIDEBAR: PARAMETRI MODELLO ----------------------------------------

st.sidebar.header("⚙️ Parametri del modello")

max_depth = st.sidebar.slider(
    "Profondità massima degli alberi",
    min_value=2,
    max_value=10,
    value=4,
    step=1,
    help="Valori più alti = modello più complesso, più rischio di overfitting.",
)

min_samples_leaf = st.sidebar.slider(
    "Minimo pazienti per foglia",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Valori più alti = foglie meno estreme, modello più regolare.",
)

# *PANORAMICA --------------------------------------------------------

st.subheader("🔍 Panoramica del dataset")

col_a, col_b, col_c = st.columns(3)

n_patients = len(df)
positive_rate = df["target"].mean()

col_a.metric("Numero pazienti", n_patients)
col_b.metric("Con malattia (%)", f"{positive_rate:.1%}")
col_c.metric(
    "Sani vs malati",
    f"{(1 - positive_rate):.1%} / {positive_rate:.1%}",
)

# EDA: variabile selezionabile + istogramma + violin + statistiche
with st.expander("Esplora la distribuzione delle variabili"):
    # selectbox mostra le label in italiano ma restituisce il nome colonna originale
    selected_feature = st.selectbox(
        "Scegli una variabile numerica",
        options=FEATURE_COLS,
        format_func=lambda c: FEATURE_LABELS[c],
    )

    serie = df[selected_feature]
    label = FEATURE_LABELS[selected_feature]

    col_plot_left, col_plot_right = st.columns(2)

    with col_plot_left:
        st.markdown(f"**Istogramma – {label} (tutti i pazienti)**")
        fig_hist, ax_hist = plt.subplots(figsize=(4, 3))
        ax_hist.hist(serie, bins=15)
        ax_hist.set_xlabel(label)
        ax_hist.set_ylabel("Numero pazienti")
        st.pyplot(fig_hist)

    with col_plot_right:
        st.markdown(f"**Violin plot – {label} per sani vs malati**")
        fig_violin, ax_violin = plt.subplots(figsize=(4, 3))
        sns.violinplot(
            data=df,
            x="target",
            y=selected_feature,
            ax=ax_violin,
        )
        ax_violin.set_xlabel("Presenza di malattia (0 = no, 1 = sì)")
        ax_violin.set_ylabel(label)
        st.pyplot(fig_violin)

    # statistiche riassuntive
    st.markdown("**Statistiche riassuntive**")
    stats = serie.describe()[["min", "25%", "50%", "75%", "max", "mean", "std"]]
    stats = stats.rename(
        index={
            "min": "Min",
            "25%": "1° quartile",
            "50%": "Mediana",
            "75%": "3° quartile",
            "max": "Max",
            "mean": "Media",
            "std": "Deviazione standard",
        }
    )
    st.table(stats.to_frame(name=label))

with st.expander("Mostra prime righe del dataset"):
    st.dataframe(df.head())

# * RF PERFORMANCE ---------------------------------------------------

model, metrics, feature_importances = train_model(
    df,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
)

st.subheader("📏 Performance del modello")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy su train", f"{metrics['acc_train']:.2%}")
col2.metric("Accuracy su test", f"{metrics['acc_test']:.2%}")
col3.metric(
    "Baseline (classe più frequente)",
    f"{metrics['baseline_acc']:.2%}",
)

st.caption(
    "Se l'accuracy su test è simile a quella su train e migliore della "
    "baseline, il modello sta generalizzando in modo ragionevole."
)

# Messaggio extra su possibile overfitting
gap = metrics["acc_train"] - metrics["acc_test"]
if gap > 0.2:
    st.warning(
        "Possibile overfitting: il modello va molto meglio su train "
        f"({metrics['acc_train']:.0%}) che su test "
        f"({metrics['acc_test']:.0%}). Prova ad abbassare la profondità "
        "degli alberi o ad aumentare il minimo di pazienti per foglia."
    )
elif gap > 0.1:
    st.info(
        "Leggero overfitting: il modello è più preciso su train "
        f"({metrics['acc_train']:.0%}) che su test "
        f"({metrics['acc_test']:.0%}), ma il gap è ancora accettabile."
    )
else:
    st.success(
        "Train e test hanno performance simili: il modello sembra "
        "generalizzare bene."
    )

# *CORR & FEATURE IMPORTANCE -----------------------------------------

st.subheader("📈 Correlazioni e importanza delle variabili")

col_corr, col_imp = st.columns(2)

with col_corr:
    st.markdown("**Correlazione tra variabili e target**")

    df_corr = df.copy()
    rename_map = {col: FEATURE_LABELS[col] for col in FEATURE_COLS}
    rename_map["target"] = TARGET_LABEL
    df_corr = df_corr.rename(columns=rename_map)

    corr = df_corr.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax_corr,
    )
    ax_corr.set_title("Matrice di correlazione")
    st.pyplot(fig_corr)

with col_imp:
    st.markdown("**Importanza delle variabili (RandomForest)**")

    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    feature_importances.plot(kind="barh", ax=ax_imp)
    ax_imp.set_xlabel("Importanza (Gini)")
    ax_imp.set_ylabel("Variabile")
    ax_imp.set_title("Importanza delle variabili")
    plt.tight_layout()
    st.pyplot(fig_imp)

# Variabile più importante + boxplot compatto per target
most_important_label = feature_importances.idxmax()
most_important_col = None
for key, value in FEATURE_LABELS.items():
    if value == most_important_label:
        most_important_col = key
        break

if most_important_col is not None:
    with st.expander(
        f"Distribuzione di {most_important_label} per sani vs malati"
    ):
        st.write(
            f"La variabile più importante per il modello è: "
            f"**{most_important_label}**."
        )

        fig_box, ax_box = plt.subplots(figsize=(4, 3))
        sns.boxplot(
            data=df,
            x="target",
            y=most_important_col,
            ax=ax_box,
        )
        ax_box.set_xlabel("Presenza di malattia (0 = no, 1 = sì)")
        ax_box.set_ylabel(most_important_label)
        st.pyplot(fig_box)

# Challenge Correlazioni: variabile più importante + stripplot
most_important_label = feature_importances.idxmax()
most_important_col = None
for key, value in FEATURE_LABELS.items():
    if value == most_important_label:
        most_important_col = key
        break

if most_important_col is not None:
    st.write(
        f"La variabile più importante per il modello è: "
        f"**{most_important_label}**."
    )

    fig_strip, ax_strip = plt.subplots(figsize=(5, 3))
    sns.stripplot(
        data=df,
        x="target",
        y=most_important_col,
        ax=ax_strip,
        alpha=0.6,
    )
    ax_strip.set_xlabel("Presenza di malattia (0 = no, 1 = sì)")
    ax_strip.set_ylabel(most_important_label)
    ax_strip.set_title(
        f"{most_important_label} per pazienti sani e con malattia"
    )
    st.pyplot(fig_strip)

# *FORM PAZIENTE -----------------------------------------------------

st.subheader("🧪 Inserisci i dati del paziente")

cols = st.columns(3)
user_input: dict[str, float] = {}

for i, col_name in enumerate(FEATURE_COLS):
    serie = df[col_name]
    min_val = float(serie.min())
    max_val = float(serie.max())
    default = float(serie.median())

    label = FEATURE_LABELS[col_name]

    with cols[i % 3]:
        user_input[col_name] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
        )

if st.button("Predici rischio"):
    input_df = pd.DataFrame([user_input])
    proba = model.predict_proba(input_df)[0]
    pred = int(proba[1] > 0.5)

    col_res1, col_res2 = st.columns(2)
    label_risk = "ALTO" if pred == 1 else "BASSO"
    col_res1.metric("Rischio stimato", label_risk)
    col_res2.metric("Probabilità di malattia", f"{proba[1]:.1%}")

    st.write("Valori inseriti:")
    pretty_input = {FEATURE_LABELS[k]: v for k, v in user_input.items()}
    st.json(pretty_input)

    st.info(
        "⚠️ Esempio didattico su un dataset pubblico. "
        "Non è uno strumento clinico e non va usato per decisioni reali."
    )