import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


@st.cache_data
def load_data():
    if os.path.exists(TRAIN_PATH):
        df = pd.read_csv(TRAIN_PATH)
        return df
    return None


def preprocess(df, sample_size=None, test_size=0.2, random_state=42):
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    if sample_size is not None and sample_size < len(X):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), sample_size, replace=False)
        X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler


def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    return X_train_pca, X_test_pca, pca, explained_variance, cumulative_variance


def apply_kmeans(X, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale"):
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, cm, report, y_pred
