
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----------------------------------------------------------------------------
# Optional dependencies — the app degrades gracefully if they are missing.
# ----------------------------------------------------------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAP_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:  # pragma: no cover
    WANDB_AVAILABLE = False


# ----------------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Taxi Fare — Linear Regression Lab",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)
sns.set_theme(style="whitegrid")

TARGET = "total"                       # total trip cost, in US dollars
NUM_FEATURES = ["distance", "trip_minutes", "passengers", "hour"]
CAT_FEATURES = ["weekday", "payment", "color", "pickup_borough", "dropoff_borough"]
FEATURES = CAT_FEATURES + NUM_FEATURES     # model-input order (matches encoder)

# Columns that make up `total` — excluded from the model to avoid leakage
LEAKAGE = ["fare", "tip", "tolls"]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

FEATURE_LABELS = {
    "distance": "Trip distance (miles)",
    "trip_minutes": "Trip duration (minutes)",
    "passengers": "Passengers",
    "hour": "Pickup hour (0–23)",
    "weekday": "Day of week",
    "payment": "Payment type",
    "color": "Taxi type",
    "pickup_borough": "Pickup borough",
    "dropoff_borough": "Drop-off borough",
}


# ----------------------------------------------------------------------------
# Data loading (cached so it runs once)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the NYC taxi dataset.

    Prefers the bundled ``nyc_taxis.csv`` (works fully offline / on deployment).
    Falls back to building it from seaborn's copy of the NYC TLC data.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "nyc_taxis.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:  # pragma: no cover - fallback build
        df = sns.load_dataset("taxis")
        df["pickup"] = pd.to_datetime(df["pickup"])
        df["dropoff"] = pd.to_datetime(df["dropoff"])
        df["hour"] = df["pickup"].dt.hour
        df["weekday"] = df["pickup"].dt.day_name()
        df["trip_minutes"] = ((df["dropoff"] - df["pickup"]).dt.total_seconds() / 60).round(2)
        keep = NUM_FEATURES + CAT_FEATURES + LEAKAGE + [TARGET]
        df = df[keep].dropna().reset_index(drop=True)
        df = df[(df.distance > 0) & (df.total > 0) &
                (df.trip_minutes > 0) & (df.trip_minutes < 180)].reset_index(drop=True)

    df["weekday"] = pd.Categorical(df["weekday"], categories=WEEKDAY_ORDER, ordered=True)
    return df


@st.cache_data(show_spinner=False)
def split_data(test_size: float = 0.2, random_state: int = 42):
    df = load_data()
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def make_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical columns, standard-scale the numeric ones."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )


def build_model(name: str, **params) -> Pipeline:
    """Return an untrained estimator wrapped with preprocessing."""
    if name == "Linear Regression":
        est = LinearRegression()
    elif name == "Ridge Regression":
        est = Ridge(alpha=params.get("alpha", 1.0))
    elif name == "Random Forest":
        est = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            random_state=42, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("prep", make_preprocessor()), ("model", est)])


@st.cache_resource(show_spinner=False)
def train_model(name: str, _params_key: str = ""):
    """Train and cache a model. ``_params_key`` busts the cache when params change."""
    X_train, X_test, y_train, y_test = split_data()
    params = st.session_state.get("model_params", {})
    model = build_model(name, **params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "R²": r2_score(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAE": mean_absolute_error(y_test, preds),
    }
    return model, metrics, (X_train, X_test, y_train, y_test, preds)


def feature_names(model: Pipeline):
    """Readable transformed-feature names (prefixes stripped)."""
    raw = model.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[-1] for n in raw]


def transform_named(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    arr = model.named_steps["prep"].transform(X)
    return pd.DataFrame(np.asarray(arr), columns=feature_names(model), index=X.index)


def metric_cards(metrics: dict):
    cols = st.columns(len(metrics))
    helptext = {
        "R²": "Share of fare variance explained (1.0 = perfect).",
        "RMSE": "Typical error in US$. Lower is better.",
        "MAE": "Average absolute error in US$. Lower is better.",
    }
    fmt = {"R²": "{:.3f}", "RMSE": "${:,.2f}", "MAE": "${:,.2f}"}
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, fmt[k].format(v), help=helptext.get(k))


# ----------------------------------------------------------------------------
# Sidebar navigation ("Dashboard", like the course example)
# ----------------------------------------------------------------------------
st.sidebar.title("🚕 Dashboard")
st.sidebar.caption("NYC Taxi Fare · Linear Regression Lab")

PAGES = [
    "1 · Business Case & Data 💼",
    "2 · Data Visualization 📊",
    "3 · Prediction 🤖",
    "4 · Explainable AI 🔍",
    "5 · Hyperparameter Tuning ⚙️",
    "6 · Conclusion 🎯",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(
    "**Stack:** Streamlit · Pandas · Scikit-Learn · SHAP · Weights & Biases"
)
if not SHAP_AVAILABLE:
    st.sidebar.warning("`shap` not installed — page 4 uses a fallback.")
if not WANDB_AVAILABLE:
    st.sidebar.info("`wandb` not installed — page 5 logs experiments locally.")

df = load_data()


# ============================================================================
# PAGE 1 — BUSINESS CASE & DATA
# ============================================================================
if page.startswith("1"):
    st.title("🚕 NYC Taxi Fare — Linear Regression Lab")
    st.header("💼 Business Case & Data Presentation")

    st.markdown(
        """
        Every day New Yorkers take **hundreds of thousands of taxi trips**. Riders
        increasingly expect to know the **price before they get in** — exactly the
        up-front pricing ride-hailing apps popularised. Taxi fleets and apps need a
        fast, transparent way to estimate trip cost from trip characteristics.

        > **Business question:** *Given the details of a NYC taxi trip, what will
        > the total cost be — and which factors drive it the most?*

        A reliable estimator lets us:
        - 💵 **Quote an up-front fare** before the trip starts.
        - 📊 **Power dynamic / surge pricing** with an objective baseline.
        - 🚨 **Flag anomalous charges** (over- or under-billing) for review.

        We frame this as a **supervised regression** problem and predict the
        continuous target `total` (total trip cost, in **US dollars**).
        """
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (trips)", f"{len(df):,}")
    c2.metric("Model features", f"{len(FEATURES)}")
    c3.metric("Target", "total ($)")
    c4.metric("Missing values", f"{int(df.isna().sum().sum())}")

    st.markdown(
        """
        **Source:** NYC TLC Yellow & Green taxi trip records (via seaborn's
        `taxis` sample of the public NYC Taxi data). Each row is one trip. It fits
        the project guidelines: 9 model features (4 numerical + 5 categorical),
        ~6.3k rows, a mix of categorical & numerical columns, and a continuous
        target to predict.

        ⚠️ `fare`, `tip` and `tolls` literally **add up to `total`**, so they are
        kept for exploration but **excluded from the model** to avoid leakage.
        """
    )

    st.subheader("Feature dictionary")
    st.dataframe(
        pd.DataFrame({
            "Column": ["distance", "trip_minutes", "passengers", "hour", "weekday",
                       "payment", "color", "pickup_borough", "dropoff_borough",
                       "fare / tip / tolls", "total"],
            "Description": [
                "Trip distance in miles",
                "Trip duration in minutes (engineered)",
                "Number of passengers",
                "Hour of pickup, 0–23 (engineered)",
                "Day of week (engineered)",
                "Payment type: credit card / cash",
                "Taxi type: yellow / green",
                "Borough where the trip started",
                "Borough where the trip ended",
                "Components of total — EXCLUDED from model (leakage)",
                "TARGET — total trip cost in US dollars",
            ],
            "Role": ["Feature (num)", "Feature (num)", "Feature (num)",
                     "Feature (num)", "Feature (cat)", "Feature (cat)",
                     "Feature (cat)", "Feature (cat)", "Feature (cat)",
                     "Excluded", "Target"],
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Description")
    num_cols = NUM_FEATURES + LEAKAGE + [TARGET]
    st.dataframe(df[num_cols].describe().T, use_container_width=True)

    with st.expander("Category counts"):
        c1, c2, c3 = st.columns(3)
        c1.write("**Payment**"); c1.dataframe(df["payment"].value_counts())
        c2.write("**Pickup borough**"); c2.dataframe(df["pickup_borough"].value_counts())
        c3.write("**Taxi type**"); c3.dataframe(df["color"].value_counts())


# ============================================================================
# PAGE 2 — DATA VISUALIZATION
# ============================================================================
elif page.startswith("2"):
    st.title("📊 Data Visualization & Insights")
    st.caption("Exploring what drives the cost of a NYC taxi trip.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distributions", "Correlation Matrix", "Cost vs Distance", "By category"]
    )

    with tab1:
        st.subheader("Distribution of a variable")
        col = st.selectbox("Variable", NUM_FEATURES + [TARGET],
                           index=len(NUM_FEATURES))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], kde=True, ax=ax[0], color="#4C72B0")
        ax[0].set_title(f"Histogram — {col}")
        sns.boxplot(x=df[col], ax=ax[1], color="#55A868")
        ax[1].set_title(f"Boxplot — {col}")
        st.pyplot(fig); plt.close(fig)
        st.info(
            "💡 **Insight:** `total` is **right-skewed** — most trips are short and "
            "cheap, with a long tail of long-distance (e.g. airport) rides."
        )

    with tab2:
        st.subheader("Correlation Matrix")
        corr = df[NUM_FEATURES + LEAKAGE + [TARGET]].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig); plt.close(fig)
        st.info(
            f"💡 **Insight:** **distance & duration drive cost** "
            f"(distance↔total ≈ {corr.loc['distance', TARGET]:.2f}). Passenger "
            "count barely matters — fares are per-trip, not per-person."
        )

    with tab3:
        st.subheader("Total cost vs. distance (coloured by payment)")
        sample = df.sample(min(4000, len(df)), random_state=1)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(data=sample, x="distance", y="total", hue="payment",
                        palette="Set1", alpha=0.5, s=18, ax=ax)
        ax.set_title("Cost rises almost linearly with distance")
        ax.set_xlabel("Distance (miles)"); ax.set_ylabel("Total ($)")
        st.pyplot(fig); plt.close(fig)
        st.info(
            "💡 **Insight:** the near-linear distance→cost band is exactly what a "
            "linear model captures. The flat cluster of long trips at ~$70 is the "
            "**JFK airport flat fare**."
        )

    with tab4:
        st.subheader("Average cost by category")
        c1, c2, c3 = st.columns(3)
        for col_obj, feat, pal in [(c1, "pickup_borough", "Blues"),
                                   (c2, "weekday", "Greens"),
                                   (c3, "payment", "Oranges")]:
            with col_obj:
                fig, ax = plt.subplots(figsize=(4.2, 4))
                order = (df[feat].cat.categories if str(df[feat].dtype) == "category"
                         else df.groupby(feat)[TARGET].mean().sort_values().index)
                sns.barplot(data=df, x=feat, y=TARGET, order=order, ax=ax,
                            palette=pal, estimator=np.mean, errorbar=None)
                ax.set_title(f"by {feat}")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig); plt.close(fig)
        st.info(
            "💡 **Insight:** trips **starting outside Manhattan cost more on "
            "average** (longer rides into the city), and credit-card trips show a "
            "higher recorded total (cash tips go unrecorded)."
        )


# ============================================================================
# PAGE 3 — PREDICTION (2+ models, switchable)
# ============================================================================
elif page.startswith("3"):
    st.title("🤖 Prediction — Estimate a trip's cost")
    st.caption("Train a model, compare performance, then predict on custom inputs.")

    model_name = st.selectbox(
        "Choose a model", ["Linear Regression", "Ridge Regression", "Random Forest"],
        help="Switch between models to compare. The first two are linear; "
             "Random Forest is a non-linear benchmark.",
    )

    params = {}
    if model_name == "Ridge Regression":
        params["alpha"] = st.slider("Ridge α (regularisation)", 0.0, 100.0, 1.0, 0.5)
    elif model_name == "Random Forest":
        params["n_estimators"] = st.slider("Number of trees", 50, 400, 200, 50)
        params["max_depth"] = st.slider("Max depth (0 = unlimited)", 0, 30, 0) or None
    st.session_state["model_params"] = params

    with st.spinner(f"Training {model_name}…"):
        model, metrics, data = train_model(model_name, f"{model_name}-{params}")
    X_train, X_test, y_train, y_test, preds = data

    st.subheader("Model performance (held-out test set)")
    metric_cards(metrics)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, preds, alpha=0.25, s=12, color="#4C72B0")
        lims = [0, max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "r--", lw=2)
        ax.set_xlabel("Actual total ($)"); ax.set_ylabel("Predicted total ($)")
        ax.set_title("Predicted vs. actual")
        st.pyplot(fig); plt.close(fig)
    with c2:
        residuals = y_test - preds
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(preds, residuals, alpha=0.25, s=12, color="#C44E52")
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Predicted total ($)"); ax.set_ylabel("Residual ($)")
        ax.set_title("Residuals (should be centred on 0)")
        st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader("🎛️ Estimate the cost of a custom trip")

    c1, c2, c3 = st.columns(3)
    distance = c1.slider(FEATURE_LABELS["distance"], 0.1, 35.0, 2.0, 0.1)
    trip_minutes = c2.slider(FEATURE_LABELS["trip_minutes"], 1.0, 120.0, 12.0, 1.0)
    passengers = c3.slider(FEATURE_LABELS["passengers"], 1, 6, 1)

    c1, c2, c3 = st.columns(3)
    hour = c1.slider(FEATURE_LABELS["hour"], 0, 23, 18)
    weekday = c2.selectbox(FEATURE_LABELS["weekday"], WEEKDAY_ORDER, index=4)
    payment = c3.selectbox(FEATURE_LABELS["payment"], sorted(df["payment"].unique()))

    c1, c2, c3 = st.columns(3)
    color = c1.selectbox(FEATURE_LABELS["color"], sorted(df["color"].unique()))
    boroughs = sorted(df["pickup_borough"].dropna().unique())
    pickup_borough = c2.selectbox(FEATURE_LABELS["pickup_borough"], boroughs,
                                  index=boroughs.index("Manhattan") if "Manhattan" in boroughs else 0)
    dboroughs = sorted(df["dropoff_borough"].dropna().unique())
    dropoff_borough = c3.selectbox(FEATURE_LABELS["dropoff_borough"], dboroughs,
                                   index=dboroughs.index("Manhattan") if "Manhattan" in dboroughs else 0)

    if st.button("Estimate fare 🚀", type="primary"):
        row = {
            "weekday": weekday, "payment": payment, "color": color,
            "pickup_borough": pickup_borough, "dropoff_borough": dropoff_borough,
            "distance": distance, "trip_minutes": trip_minutes,
            "passengers": passengers, "hour": hour,
        }
        x = pd.DataFrame([row])[FEATURES]
        pred = float(model.predict(x)[0])
        st.success(f"### Estimated total cost: **${pred:,.2f}**")


# ============================================================================
# PAGE 4 — EXPLAINABLE AI
# ============================================================================
elif page.startswith("4"):
    st.title("🔍 Explainable AI — what drives the fare?")
    st.caption("Understanding *why* the model predicts what it predicts.")

    model_name = st.selectbox(
        "Model to explain", ["Random Forest", "Linear Regression", "Ridge Regression"],
    )
    st.session_state["model_params"] = {}
    with st.spinner(f"Training {model_name}…"):
        model, metrics, data = train_model(model_name, f"explain-{model_name}")
    X_train, X_test, y_train, y_test, preds = data

    n_sample = st.slider("Sample size for SHAP (smaller = faster)", 100, 1000, 300, 100)
    X_raw = X_test.sample(min(n_sample, len(X_test)), random_state=0)
    X_enc = transform_named(model, X_raw)
    estimator = model.named_steps["model"]

    shap_ok = False
    if SHAP_AVAILABLE:
        with st.spinner("Computing SHAP values…"):
            try:
                if model_name == "Random Forest":
                    explainer = shap.TreeExplainer(estimator)
                else:
                    explainer = shap.LinearExplainer(estimator, X_enc)
                shap_values = explainer.shap_values(X_enc)

                st.subheader("Global feature importance")
                fig = plt.figure()
                shap.summary_plot(shap_values, X_enc, plot_type="bar",
                                  max_display=15, show=False)
                st.pyplot(fig, bbox_inches="tight"); plt.close(fig)
                st.info(
                    "💡 Bars show each feature's **average impact** on the predicted "
                    "cost. **distance** and **trip duration** dominate."
                )

                st.subheader("Impact distribution (beeswarm)")
                fig = plt.figure()
                shap.summary_plot(shap_values, X_enc, max_display=15, show=False)
                st.pyplot(fig, bbox_inches="tight"); plt.close(fig)
                st.caption(
                    "Red = high feature value, blue = low. Points to the right push "
                    "the predicted cost **up**."
                )
                shap_ok = True
            except Exception as e:  # pragma: no cover
                st.error(f"SHAP failed ({e}); showing a fallback below.")

    if not shap_ok:
        st.subheader("Feature importance (permutation — fallback)")
        from sklearn.inspection import permutation_importance
        with st.spinner("Computing permutation importance…"):
            r = permutation_importance(model, X_raw, y_test.loc[X_raw.index],
                                       n_repeats=5, random_state=0)
        imp = pd.Series(r.importances_mean, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot.barh(ax=ax, color="#4C72B0")
        ax.set_title("Permutation importance")
        st.pyplot(fig); plt.close(fig)

    if model_name in ("Linear Regression", "Ridge Regression"):
        st.subheader("Top linear coefficients (standardised)")
        coefs = pd.Series(estimator.coef_, index=feature_names(model))
        top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(12).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#C44E52" if v < 0 else "#55A868" for v in top]
        top.plot.barh(ax=ax, color=colors)
        ax.set_title("Largest effects on cost (per 1 std-dev increase)")
        st.pyplot(fig); plt.close(fig)
        st.caption("Green raises the predicted cost; red lowers it.")


# ============================================================================
# PAGE 5 — HYPERPARAMETER TUNING (Weights & Biases)
# ============================================================================
elif page.startswith("5"):
    st.title("⚙️ Hyperparameter Tuning & Experiment Tracking")
    st.caption("Search hyperparameters, track every run, and select the best model.")


    family = st.selectbox("Model family to tune", ["Ridge Regression", "Random Forest"])

    with st.expander("🔗 Weights & Biases logging (optional)"):
        if WANDB_AVAILABLE:
            use_wandb = st.checkbox("Log this sweep to Weights & Biases")
            wandb_key = st.text_input("W&B API key", type="password",
                                      help="From https://wandb.ai/authorize")
            wandb_project = st.text_input("W&B project", value="ds4e-nyc-taxi-final")
        else:
            use_wandb, wandb_key, wandb_project = False, "", ""
            st.info("`wandb` is not installed. `pip install wandb` to enable logging.")

    if family == "Ridge Regression":
        grid = [{"alpha": a} for a in [0.01, 0.1, 1, 10, 50, 100]]
    else:
        grid = [{"n_estimators": n, "max_depth": d}
                for n in [100, 200] for d in [6, 12, 20]]
    st.write(f"**{len(grid)} configurations** to evaluate.")

    if st.button("Run sweep 🧪", type="primary"):
        X_train, X_test, y_train, y_test = split_data()
        results = []
        if use_wandb and wandb_key:
            try:
                os.environ["WANDB_API_KEY"] = wandb_key
                wandb.login(key=wandb_key)
            except Exception as e:
                st.warning(f"W&B login failed ({e}); continuing locally.")
                use_wandb = False

        progress = st.progress(0.0, text="Starting sweep…")
        for i, cfg in enumerate(grid, start=1):
            model = build_model(family, **cfg)
            model.fit(X_train, y_train)
            p = model.predict(X_test)
            row = {
                **cfg,
                "R²": r2_score(y_test, p),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
                "MAE": mean_absolute_error(y_test, p),
            }
            results.append(row)
            if use_wandb and wandb_key:
                try:
                    run = wandb.init(project=wandb_project, name=f"{family}-{i}",
                                     config=cfg, reinit=True)
                    wandb.log({k: row[k] for k in ("R²", "RMSE", "MAE")})
                    run.finish()
                except Exception as e:
                    st.warning(f"W&B logging error: {e}")
                    use_wandb = False
            progress.progress(i / len(grid), text=f"Evaluated {i}/{len(grid)}")
        progress.empty()

        res_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
        st.session_state["sweep_results"] = res_df

    if "sweep_results" in st.session_state:
        res_df = st.session_state["sweep_results"]
        st.subheader("📋 Experiment results (sorted by R²)")
        st.dataframe(
            res_df.style.highlight_max(subset=["R²"], color="#b6e3b6")
                       .highlight_min(subset=["RMSE", "MAE"], color="#b6e3b6"),
            use_container_width=True,
        )

        best = res_df.iloc[0]
        hp_cols = [c for c in res_df.columns if c not in ("R²", "RMSE", "MAE")]
        st.success(
            "🏆 **Best config:** "
            + ", ".join(f"{k}={best[k]}" for k in hp_cols)
            + f"  →  R² = {best['R²']:.3f}, RMSE = ${best['RMSE']:,.2f}"
        )

        if len(hp_cols) == 1:
            d = res_df.sort_values(hp_cols[0])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(d[hp_cols[0]], d["R²"], "o-", color="#4C72B0")
            ax.set_xlabel(hp_cols[0]); ax.set_ylabel("R²")
            ax.set_title(f"R² vs. {hp_cols[0]}")
            st.pyplot(fig); plt.close(fig)
        elif len(hp_cols) == 2:
            pivot = res_df.pivot_table(index=hp_cols[0], columns=hp_cols[1], values="R²")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
            ax.set_title("R² across the hyperparameter grid")
            st.pyplot(fig); plt.close(fig)

        if WANDB_AVAILABLE:
            st.caption("Tip: enable W&B logging above for a shareable online dashboard.")
    else:
        st.info("Configure the options above and click **Run sweep** to begin.")


# ============================================================================
# PAGE 6 — CONCLUSION
# ============================================================================
elif page.startswith("6"):
    st.title("🎯 Conclusion")
    st.caption("What we built, what we learned, and what it means for the business.")

    st.header("The problem, recapped")
    st.markdown(
        """
        We set out to answer one question for a ride-hailing / taxi platform:

        > *Given the details of a NYC taxi trip, what will the **total cost** be —
        > and which factors drive it the most?*

        Using ~6,300 real NYC taxi trips, we built an end-to-end Streamlit app that
        explores the data, predicts the fare with **linear regression**, explains
        the drivers with **SHAP**, and tunes models with experiment tracking.
        """
    )

    # --- Live model comparison so the conclusion reflects real numbers ---
    st.header("How well can we predict the fare?")
    st.session_state["model_params"] = {}
    rows = []
    with st.spinner("Evaluating the three models…"):
        for name in ["Linear Regression", "Ridge Regression", "Random Forest"]:
            _, m, _ = train_model(name, f"conclusion-{name}")
            rows.append({"Model": name, **m})
    results = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    best = results.iloc[0]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(
            results.style.format({"R²": "{:.3f}", "RMSE": "${:,.2f}", "MAE": "${:,.2f}"})
                   .highlight_max(subset=["R²"], color="#b6e3b6")
                   .highlight_min(subset=["RMSE", "MAE"], color="#b6e3b6"),
            use_container_width=True, hide_index=True,
        )
    with c2:
        st.metric("🏆 Best model", best["Model"], help="Highest R² on the held-out test set")
        st.metric("R²", f"{best['R²']:.3f}")
        st.metric("Typical error (RMSE)", f"${best['RMSE']:,.2f}")

    st.success(
        f"Our best model (**{best['Model']}**) explains **{best['R²']*100:.1f}%** of the "
        f"variation in trip cost, with a typical error of only **${best['RMSE']:,.2f}** — "
        "accurate enough to power an up-front fare estimate."
    )

    st.header("Key findings")
    st.markdown(
        """
        - 📏 **Distance and trip duration are the dominant drivers** of cost — confirmed
          by both the correlation analysis and the SHAP feature-importance ranking.
        - 👥 **Passenger count barely matters**: NYC fares are charged per *trip*, not
          per person.
        - 🗺️ **Pickup borough shifts the baseline**: trips starting outside Manhattan
          tend to cost more (longer rides into the city); the **JFK airport flat fare**
          shows up as a distinct cluster around \\$70.
        - 💳 **Payment type correlates with recorded total** — cash tips go unrecorded,
          so credit-card trips show a higher logged total.
        - ⚖️ **Linear models are already strong here** because the core relationship
          (distance → fare) is close to linear; Random Forest adds a modest edge by
          capturing non-linear effects like the airport flat fare.
        """
    )

    st.header("Business recommendations")
    st.markdown(
        """
        1. 💵 **Ship up-front pricing** using the model — distance + duration + route
           are enough for a reliable quote before the trip starts.
        2. 🚨 **Flag anomalies**: trips whose actual cost deviates far from the predicted
           value are candidates for over-/under-billing review.
        3. 📈 **Feed surge/dynamic pricing** with the model's baseline so surcharges are
           applied on top of an objective, explainable estimate.
        """
    )

    st.header("Limitations & next steps")
    st.markdown(
        """
        - The sample is a **snapshot of NYC trips**; retrain on the full, current
          TLC dataset before production use.
        - We deliberately **excluded `fare`/`tip`/`tolls`** (they sum to `total`) to
          avoid leakage — a production model could instead predict the metered fare
          and model tips separately.
        - Next: add **traffic & weather** features, try a **log-transformed target**
          for the long right tail, and run a larger **W&B sweep** to squeeze out more
          accuracy.
        """
    )

    st.divider()
    st.markdown(
        "**Thanks for visiting!** Built with Streamlit · Pandas · Scikit-Learn · "
        "SHAP · Weights & Biases — NYU DS-4-Everyone Final Project."
    )

#streamlit run streamlit_app.py 