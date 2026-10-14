import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide",
)

# --------------------------------------------------
# Load the model and dataset
# --------------------------------------------------
@st.cache_resource
def load_artifact():
    return joblib.load("breast_cancer_knn.joblib")

@st.cache_data
def load_dataset():
    return load_breast_cancer()


artifact = load_artifact()
dataset = load_dataset()

model = artifact["model"]
feature_names = artifact["feature_names"]
target_names = artifact["target_name"]

# Check the feature names against the saved dataset
if feature_names != list(dataset.feature_names):
    st.error(
        "Feature names or their order do not match the "
        "Wisconsin Diagnostic Breast Cancer dataset. "
        "Check your training notebook."
    )
    st.stop()

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("🔬 Breast Cancer Detection")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict",
        "Model Performance",
        "About the Model",
    ],
)
st.sidebar.caption(
    "Educational project — not a medical diagnostic tool."
)

# --------------------------------------------------
# Home
# --------------------------------------------------
if page == "Home":
    st.title("Breast Cancer Classification Dashboard")
    st.write(
        "Explore a K-Nearest Neighbors model trained on "
        "the Wisconsin Diagnostic Breast Cancer dataset."
    )

    st.warning(
        "This application is for educational purposes only. "
        "It's predictions must not be used for medical decisions."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset samples", len(dataset.data))

    with col2:
        st.metric("Input features", len(feature_names))

    with col3:
        st.metric("Classes", len(target_names))

    st.subheader("Class distribution")

    class_counts = pd.Series(
        dataset.target
    ).map(
        dict(enumerate(dataset.target_names))
    ).value_counts()

    st.bar_chart(class_counts)

    st.subheader("Explore the application")
    st.markdown(
        """
        - **Predict:** Enter measurements and explore a prediction.
        - **Model Performance:** Review model validation information.
        - **About the Model:** Understand KNN and GridSearchCV.
        """
    )

# --------------------------------------------------
# Prediction page
# --------------------------------------------------
elif page == "Predict":
    st.title("Make a Prediction")

    st.write(
        "Enter all 30 numerical features. For an initial demo, "
        "the fields start with the dataset's median values."
    )

    st.info(
        "These measurements are technical dataset features, "
        "not ordinary measurements that a patient can enter "
        "without the appropriate clinical data."
    )
    medians = np.median(dataset.data, axis=0)

    with st.form("prediction_form"):
        user_values = {}

        st.subheader("Mean features")
        mean_features = [
            (i, name) for i, name in enumerate(feature_names)
            if name.startswith("mean")
        ]

        cols = st.columns(3)

        for j, (i, name) in enumerate(mean_features):
            with cols[j % 3]:
                user_values[name] = st.number_input(
                    name,
                    value=float(medians[i]),
                    format="%.5f",
                    key=f"feature_{i}",
                )

        st.subheader("Error and worst features")

        remaining_features = [
            (i, name) for i, name in enumerate(feature_names)
            if not name.startswith("mean")
        ]

        for start in range(0, len(remaining_features), 3):
            cols = st.columns(3)

            for j, (i, name) in enumerate(
                remaining_features[start:start + 3]
            ):
                with cols[j]:
                    user_values[name] = st.number_input(
                        name,
                        value=float(medians[i]),
                        format="%.5f",
                        key=f"feature_{i}",
                    )

        submitted = st.form_submit_button(
            "Predict classification",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [[user_values[name] for name in feature_names]],
            columns=feature_names,
        )

        prediction = model.predict(input_df)[0]

        # 0 = malignant and 1 = benign.
        if isinstance(prediction, (int, np.integer)):
            predicted_label = target_names[int(prediction)]
        else:
            predicted_label = str(prediction)

        st.subheader("Prediction result")

        if predicted_label.lower() == "benign":
            st.success("Model classification: Benign")
        elif predicted_label.lower() == "malignant":
            st.error("Model classification: Malignant")
        else:
            st.info(f"Model classification: {predicted_label}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            st.subheader("Model output scores")

            for class_label, probability in zip(
                classes, probabilities
            ):
                if isinstance(class_label, (int, np.integer)):
                    class_name = target_names[int(class_label)]
                else:
                    class_name = str(class_label)

                st.write(
                    f"{class_name.title()}: "
                    f"{probability:.1%}"
                )
                st.progress(float(probability))

            st.caption(
                "These are model-estimated class probabilities, "
                "not calibrated medical risk estimates."
            )

# --------------------------------------------------
# Model performance page
# --------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    st.write(
        "The score below is the best mean cross-validation "
        "score reported by the GridSearchCV run."
    )

    best_cv_score = artifact.get("best_cv")

    if best_cv_score is not None:
        st.metric(
            "Best mean CV score",
            f"{best_cv_score:.2%}",
        )
    else:
        st.info("No cross-validation score was saved.")

    st.subheader("Best hyperparameters")

    best_params = artifact.get("best_params", {})

    if best_params:
        st.json(best_params)
    else:
        st.info("No best-parameter information was saved.")

    st.subheader("Class distribution")

    counts = pd.Series(
        dataset.target
    ).map(
        dict(enumerate(dataset.target_names))
    ).value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.set_title("Dataset class distribution")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Cross-validation score is not the same as held-out "
        "test accuracy. We can add your notebook's test-set "
        "metrics and confusion matrix after saving them."
    )

# --------------------------------------------------
# About the model
# --------------------------------------------------
elif page == "About the Model":
    st.title("About the Model")

    st.subheader("What is KNN?")

    st.write(
        "K-Nearest Neighbors classifies a new sample by "
        "looking at nearby training samples in feature space."
    )

    st.subheader("Why scale the features?")

    st.write(
        "KNN uses distances between samples. StandardScaler "
        "helps prevent features with larger numerical ranges "
        "from dominating those distances."
    )

    st.subheader("What does GridSearchCV do?")

    st.write(
        "GridSearchCV evaluates specified hyperparameter "
        "combinations using cross-validation and identifies "
        "the combination with the best mean validation score."
    )

    st.subheader("Dataset")

    st.markdown(
        """
        The Wisconsin Diagnostic Breast Cancer dataset is
        available through scikit-learn. Its 30 numerical
        features describe characteristics of cell nuclei
        computed from digitized fine-needle aspirate images.
        """
    )

    st.caption(
        "This application is an educational demonstration, "
        "not a substitute for professional medical evaluation."
    )
