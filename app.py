
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("modele_velib_rf.pkl")
le = joblib.load("label_encoder_station.pkl")
stations_connues = list(le.classes_)

st.set_page_config(page_title="📍 Prédiction Vélib'", layout="centered")
st.title("🚲 Prédire si une station Vélib’ sera vide")

nom_station = st.selectbox("📌 Nom de la station", stations_connues)
heure = st.slider("🕒 Heure de la journée", 0, 23, 8)
jour_semaine = st.selectbox("📅 Jour de la semaine", list(range(7)), format_func=lambda x: ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][x])
mois = st.selectbox("🗓️ Mois", list(range(1, 13)), format_func=lambda x: ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"][x-1])

station_id = le.transform([nom_station])[0]
X_input = pd.DataFrame([[heure, jour_semaine, mois, station_id]], columns=["heure", "jour_semaine", "mois", "station_id"])

prediction = model.predict(X_input)[0]
proba = model.predict_proba(X_input)[0][1]

st.markdown("### Résultat de la prédiction :")
if prediction == 1:
    st.error(f"🚨 La station **{nom_station}** risque d’être **vide** à {heure}h.")
else:
    st.success(f"✅ La station **{nom_station}** **ne sera pas vide** à {heure}h.")

st.markdown(f"**Probabilité que la station soit vide** : {proba:.2%}")
