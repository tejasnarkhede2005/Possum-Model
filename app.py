import streamlit as st
import pandas as pd
import joblib

# Load the trained XGBoost model
model = joblib.load("xgboost_possum_model.pkl")

# Configure the Streamlit page
st.set_page_config(page_title="Possum Classifier", layout="centered")


# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

# Home page
if page == "Home":
    st.title("🦡 Possum Population Classifier")
    st.markdown("Predict whether a possum is from **Victoria (Vic)** or **other** populations based on biological features.")

    # User input form
    def user_input():
        st.subheader("📥 Enter Possum Data")

        site = st.slider("Site", 0, 7, 1)
        sex = st.selectbox("Sex", ["male", "female"])
        hdlngth = st.number_input("Head Length (hdlngth)", value=95.0)
        skullw = st.number_input("Skull Width (skullw)", value=55.0)
        totlngth = st.number_input("Total Length (totlngth)", value=900.0)
        taill = st.number_input("Tail Length (taill)", value=360.0)
        footlgth = st.number_input("Foot Length (footlgth)", value=80.0)
        earconch = st.number_input("Ear Conch Length (earconch)", value=45.0)
        eye = st.number_input("Eye Width (eye)", value=15.0)
        chest = st.number_input("Chest Girth (chest)", value=180.0)
        belly = st.number_input("Belly Girth (belly)", value=140.0)

        sex_num = 1 if sex == "male" else 0

        # Create input dictionary
        input_data = {
            'site': site,
            'sex': sex_num,
            'hdlngth': hdlngth,
            'skullw': skullw,
            'totlngth': totlngth,
            'taill': taill,
            'footlgth': footlgth,
            'earconch': earconch,
            'eye': eye,
            'chest': chest,
            'belly': belly
        }

        # Create DataFrame
        df = pd.DataFrame([input_data])

        # Add dummy columns to match training feature set
        df['case'] = 0
        df['Pop'] = 0

        # Reorder columns to match the model's training data
        expected_columns = ['case', 'site', 'Pop', 'sex', 'hdlngth', 'skullw', 'totlngth',
                            'taill', 'footlgth', 'earconch', 'eye', 'chest', 'belly']
        df = df[expected_columns]

        return df

    # Get user input
    input_df = user_input()

    # Predict
    if st.button("🧪 Predict"):
        try:
            prediction = model.predict(input_df)[0]
            label = "Victoria (Vic)" if prediction == 0 else "Other"
            st.success(f"🌏 Predicted Population: **{label}**")
        except ValueError as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# About page
elif page == "About":
    st.title("📄 About This App")
    st.markdown("""
    This Streamlit app uses a trained **XGBoost Classifier** to predict whether a possum belongs to the **Victoria (Vic)** population or another population based on biological measurements.

    ### 🔢 Features used in model:
    - case (dummy)
    - site
    - Pop (dummy)
    - sex (0 = female, 1 = male)
    - hdlngth (head length)
    - skullw (skull width)
    - totlngth (total length)
    - taill (tail length)
    - footlgth (foot length)
    - earconch (ear conch length)
    - eye (eye width)
    - chest (chest girth)
    - belly (belly girth)

    **Author**: Tejas Narkhede
    **Email**: tejasnarkhede03@gmail.com  
    """)
