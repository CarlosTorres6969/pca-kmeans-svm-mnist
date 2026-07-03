import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import (
    load_data, preprocess, apply_pca, apply_kmeans,
    train_svm, evaluate_model
)

st.set_page_config(
    page_title="PCA + K-Means + SVM - MNIST",
    layout="wide"
)

st.title("Reducción de Dimensionalidad y Clasificación")
st.markdown("**PCA + K-Means + SVM sobre MNIST Digit Recognizer**")
st.markdown("---")
st.markdown("**Estudiante:** José Carlos Torres Donaire — 20211920141")
st.markdown("**Materia:** Inteligencia Artificial")

# -------------------------------------------------------------------
# 1. Carga de datos
# -------------------------------------------------------------------
df = load_data()

if df is None:
    st.warning(
        "No se encontró el archivo `data/train.csv`. "
        "Descárgalo desde [Kaggle MNIST](https://www.kaggle.com/competitions/digit-recognizer/data) "
        "y colócalo en la carpeta `data/`."
    )
    st.stop()

st.sidebar.header("Configuración")

# -------------------------------------------------------------------
# 2. Preprocesamiento
# -------------------------------------------------------------------
SAMPLE_SIZE = st.sidebar.slider(
    "Tamaño de muestra (filas)",
    min_value=1000, max_value=len(df), value=10000, step=1000
)

with st.spinner("Preprocesando datos..."):
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df, sample_size=SAMPLE_SIZE
    )

st.sidebar.success(f"Datos cargados: {len(df):,} registros")
st.sidebar.write(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# -------------------------------------------------------------------
# 3. PCA
# -------------------------------------------------------------------
st.sidebar.subheader("PCA")
MAX_COMPONENTS = min(X_train.shape[0], X_train.shape[1], 100)
N_COMPONENTS = st.sidebar.slider(
    "Número de componentes principales",
    min_value=2, max_value=MAX_COMPONENTS, value=10, step=1
)

with st.spinner("Aplicando PCA..."):
    X_train_pca, X_test_pca, pca_model, expl_var, cum_var = apply_pca(
        X_train, X_test, n_components=N_COMPONENTS
    )

# -------------------------------------------------------------------
# 4. K-Means
# -------------------------------------------------------------------
st.sidebar.subheader("K-Means")
N_CLUSTERS = st.sidebar.slider(
    "Número de clústeres (k)",
    min_value=2, max_value=15, value=10, step=1
)

with st.spinner("Aplicando K-Means..."):
    kmeans_model, cluster_labels = apply_kmeans(X_train_pca, n_clusters=N_CLUSTERS)

# -------------------------------------------------------------------
# 5. SVM
# -------------------------------------------------------------------
st.sidebar.subheader("SVM")
SVM_KERNEL = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
SVM_C = st.sidebar.select_slider("C (regularización)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
SVM_GAMMA = st.sidebar.select_slider(
    "Gamma", options=["scale", "auto", 0.001, 0.01, 0.1, 1.0], value="scale"
)

RUN_CLASSIFICATION = st.sidebar.button("Entrenar SVM", type="primary")

# ===================================================================
# CUERPO PRINCIPAL
# ===================================================================

tab_data, tab_pca, tab_kmeans, tab_svm, tab_predict = st.tabs([
    "📊 Datos", "📉 PCA", "🔵 K-Means", "🤖 SVM", "🔮 Predecir"
])

# ========================== TAB: DATOS =============================
with tab_data:
    st.subheader("Vista previa del dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", f"{len(df):,}")
    col2.metric("Columnas (píxeles)", f"{df.shape[1] - 1:,}")
    col3.metric("Clases (0-9)", "10")

    st.dataframe(df.head(10))

    st.subheader("Distribución de dígitos")
    fig, ax = plt.subplots(figsize=(10, 4))
    df["label"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de clases en el dataset")
    st.pyplot(fig)

    st.subheader("Ejemplos de imágenes")
    fig2, axes = plt.subplots(2, 10, figsize=(12, 3))
    for digit in range(10):
        idx = df[df["label"] == digit].index[0]
        img = df.drop(columns=["label"]).iloc[idx].values.reshape(28, 28)
        for row_idx in range(2):
            axes[row_idx][digit].imshow(img, cmap="gray")
            axes[row_idx][digit].axis("off")
            if row_idx == 0:
                axes[row_idx][digit].set_title(str(digit))
    fig2.suptitle("Ejemplos de dígitos manuscritos", fontsize=14, y=1.05)
    st.pyplot(fig2)

# ========================== TAB: PCA ===============================
with tab_pca:
    st.subheader("Análisis de Componentes Principales (PCA)")

    col1, col2 = st.columns(2)
    col1.metric("Componentes seleccionados", N_COMPONENTS)
    col2.metric(
        "Varianza explicada acumulada",
        f"{cum_var[-1] * 100:.2f}%"
    )

    st.subheader("Varianza explicada por componente")
    fig_pca1, ax1 = plt.subplots(figsize=(10, 4))
    bars = ax1.bar(range(1, N_COMPONENTS + 1), expl_var * 100, alpha=0.7, label="Individual")
    ax1.plot(range(1, N_COMPONENTS + 1), cum_var * 100, "r-o", label="Acumulada")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Varianza explicada (%)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig_pca1)

    st.subheader("Proyección 2D de los datos")
    X_2d = X_train_pca[:, :2] if N_COMPONENTS >= 2 else np.column_stack([X_train_pca[:, 0], np.zeros_like(X_train_pca[:, 0])])
    df_2d = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Dígito": y_train.astype(str)
    })
    fig_pca2 = px.scatter(
        df_2d, x="PC1", y="PC2", color="Dígito",
        title="Proyección PCA en 2 dimensiones",
        width=900, height=600,
        opacity=0.6
    )
    st.plotly_chart(fig_pca2, use_container_width=True)

# ========================== TAB: K-MEANS ===========================
with tab_kmeans:
    st.subheader("Agrupamiento con K-Means")

    inertia = kmeans_model.inertia_
    st.metric("Inercia del modelo", f"{inertia:,.0f}")

    st.subheader("Visualización de clústeres")
    X_2d = X_train_pca[:, :2] if N_COMPONENTS >= 2 else np.column_stack([X_train_pca[:, 0], np.zeros_like(X_train_pca[:, 0])])
    df_kmeans = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Cluster": cluster_labels.astype(str),
        "Dígito real": y_train.astype(str)
    })
    col_k1, col_k2 = st.columns(2)

    with col_k1:
        fig_km1 = px.scatter(
            df_kmeans, x="PC1", y="PC2", color="Cluster",
            title="Clústeres formados por K-Means",
            width=500, height=450,
            opacity=0.6
        )
        st.plotly_chart(fig_km1, use_container_width=True)

    with col_k2:
        fig_km2 = px.scatter(
            df_kmeans, x="PC1", y="PC2", color="Dígito real",
            title="Etiquetas reales (para comparación)",
            width=500, height=450,
            opacity=0.6
        )
        st.plotly_chart(fig_km2, use_container_width=True)

    st.subheader("Composición de cada clúster")
    cross_tab = pd.crosstab(cluster_labels, y_train, margins=True)
    st.dataframe(cross_tab)

    st.subheader("Interpretación")
    st.info(
        "Cada clúster se asigna al dígito mayoritario. "
        "La diagonal principal de la tabla muestra aciertos del agrupamiento "
        "respecto a las etiquetas reales."
    )

# ========================== TAB: SVM ===============================
with tab_svm:
    st.subheader("Clasificación con Support Vector Machine")

    if RUN_CLASSIFICATION:
        with st.spinner("Entrenando SVM..."):
            svm_model = train_svm(
                X_train_pca, y_train,
                kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA
            )
            accuracy, cm, report, y_pred = evaluate_model(svm_model, X_test_pca, y_test)

        st.session_state["svm_model"] = svm_model
        st.session_state["svm_accuracy"] = accuracy
        st.session_state["svm_cm"] = cm
        st.session_state["svm_report"] = report
        st.session_state["svm_y_pred"] = y_pred
        st.session_state["svm_trained"] = True
        st.session_state["svm_n_components"] = N_COMPONENTS

    if st.session_state.get("svm_trained"):
        accuracy = st.session_state["svm_accuracy"]
        cm = st.session_state["svm_cm"]
        report = st.session_state["svm_report"]
        y_pred = st.session_state["svm_y_pred"]

        st.success(f"**Accuracy del modelo:** {accuracy * 100:.2f}%")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.subheader("Matriz de Confusión")
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
            ax_cm.set_xlabel("Predicción")
            ax_cm.set_ylabel("Valor real")
            ax_cm.set_title("Matriz de Confusión")
            st.pyplot(fig_cm)

        with col_s2:
            st.subheader("Reporte de Clasificación")
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
            st.dataframe(report_df)

        st.subheader("Precisión por dígito")
        digit_acc = []
        for d in range(10):
            mask = y_test == d
            digit_acc.append({
                "Dígito": d,
                "Muestras": mask.sum(),
                "Aciertos": (y_pred[mask] == d).sum(),
                "Precisión": f"{(y_pred[mask] == d).sum() / mask.sum() * 100:.1f}%"
            })
        st.dataframe(pd.DataFrame(digit_acc))

        st.subheader("Análisis de resultados")
        st.markdown(
            f"""
            - **Componentes PCA:** {N_COMPONENTS}
            - **Kernel SVM:** {SVM_KERNEL}
            - **Accuracy global:** {accuracy * 100:.2f}%
            - La reducción de dimensionalidad con PCA permite entrenar SVM más rápido
              manteniendo un alto rendimiento.
            """
        )
    else:
        st.info("Haz clic en **'Entrenar SVM'** en la barra lateral para comenzar la clasificación.")

# ========================== TAB: PREDECIR ==========================
with tab_predict:
    st.subheader("Predecir un dígito manualmente")

    if not st.session_state.get("svm_trained"):
        st.info("Primero entrena el modelo SVM en la pestaña anterior.")
    elif st.session_state.get("svm_n_components") != N_COMPONENTS:
        st.warning(
            "El número de componentes PCA ha cambiado desde que se entrenó el modelo. "
            "Vuelve a entrenar el SVM para usar la configuración actual."
        )
    else:
        svm_model = st.session_state["svm_model"]

        st.markdown(
            "Selecciona un índice del conjunto de prueba para ver la predicción:"
        )
        test_idx = st.number_input(
            "Índice de prueba",
            min_value=0, max_value=len(X_test_pca) - 1, value=0, step=1
        )

        sample_pixels = X_test[test_idx].reshape(1, -1)
        sample_pca = pca_model.transform(sample_pixels)
        prediction = svm_model.predict(sample_pca)[0]
        true_label = y_test[test_idx]

        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            img_display = X_test[test_idx].reshape(28, 28)
            fig_img, ax_img = plt.subplots(figsize=(4, 4))
            ax_img.imshow(img_display, cmap="gray")
            ax_img.axis("off")
            st.pyplot(fig_img)

        with col_p2:
            st.metric("Valor real", str(true_label))
            st.metric("Predicción del modelo", str(prediction))
            st.metric("¿Coincide?", "✅ Sí" if prediction == true_label else "❌ No")

        with col_p3:
            if prediction == true_label:
                st.success("El modelo clasificó correctamente el dígito.")
            else:
                st.error("El modelo no clasificó correctamente el dígito.")
