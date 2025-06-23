# app.py
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model/logistic_regression_model.pkl")  # Pastikan path dan nama file model sesuai
class_names = ["Tidak Diabetes", "Diabetes"]

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Diabetes", layout="centered")
st.markdown("<h1 style='text-align: center; color: purple;'>Masukkan Semua Detail</h1>", unsafe_allow_html=True)

# Form Input
with st.form("form_diabetes"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Age")
        pregnancies = st.text_input("Pregnancies")
        glucose = st.text_input("Glucose")
        blood_pressure = st.text_input("Blood Pressure")
    
    with col2:
        insulin = st.text_input("Insulin")
        bmi = st.text_input("BMI")
        skin_thickness = st.text_input("Skin Thickness")
        dpf = st.text_input("DPF")

    submitted = st.form_submit_button("Prediksi!", type="primary")

if submitted:
    try:
        input_data = np.array([[
            float(pregnancies), float(glucose), float(blood_pressure),
            float(skin_thickness), float(insulin), float(bmi),
            float(dpf), float(age)
        ]])

        prediction = model.predict(input_data)[0]
        result_text = class_names[prediction]
        st.success(f"Hasil prediksi model: **{result_text}**")

        st.markdown("### Detail Input yang Anda Masukkan:")
        st.markdown(f"""
        - **Umur (Age)**: {age}
        - **Jumlah Kehamilan (Pregnancies)**: {pregnancies}
        - **Kadar Glukosa (Glucose)**: {glucose}
        - **Tekanan Darah (Blood Pressure)**: {blood_pressure}
        - **Ketebalan Kulit (Skin Thickness)**: {skin_thickness}
        - **Insulin**: {insulin}
        - **Indeks Massa Tubuh (BMI)**: {bmi}
        - **Fungsi Riwayat Diabetes (DPF)**: {dpf}
        """)

    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid.")

st.caption("Aplikasi Prediksi Diabetes menggunakan Streamlit & Scikit-learn")
