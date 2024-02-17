import streamlit as st

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from lime import lime_tabular

# Load the data
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df["WineType"] = [wine.target_names[typ] for typ in wine.target]

X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=123
)

# load model
rf_classif = load("rf_classif.model")

y_test_preds = rf_classif.predict(X_test)

# Dashboard
st.title("Wine Type :red[Prediction] :chart_with_upwards_trend: :bar_chart: :tea: :coffee:")
st.markdown("Predict wine type based on the ingredient values")

tab1, tab2, tab3 = st.tabs(["Data :clipboard:", "Global Performance :weight_lifter:", "Local Performance :bicyclist:"])

with tab1:
    st.header("Wine Dataset")
    st.write(wine_df)

with tab2:
    conf_mat_fig = plt.figure(figsize=(6,6))
    ax1 = conf_mat_fig.add_subplot(111)
    skplt.metrics.plot_confusion_matrix(y_test, y_test_preds, ax= ax1)
    st.pyplot(conf_mat_fig, use_container_width=True)

    