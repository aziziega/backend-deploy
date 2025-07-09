# app.py
import streamlit as st
import numpy as np
import joblib   

# Load model dan scaler
model = joblib.load("model/logistic_regression_model.pkl")
scaler = joblib.load("model/scaler.pkl")
class_names = ["Tidak Diabetes", "Diabetes"]

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Diabetes", layout="centered")
st.markdown("<h1 style='text-align: center; color: purple;'>Masukkan Semua Detail</h1>", unsafe_allow_html=True)

# Form Input
with st.form("form_diabetes"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Age")
        pregnancies_option = st.selectbox("Apakah pernah hamil?", ["No", "Yes"])
        if pregnancies_option == "Yes":
            pregnancies = st.text_input("Jumlah Kehamilan")
        else:
            pregnancies = "0"

        glucose = st.text_input("Glucose (mg/dL) [Normal: 70-100]")
        blood_pressure = st.text_input("Blood Pressure (mmHg) [Normal: 80-120]")
        skin_thickness = st.text_input("Skin Thickness (mm) [Normal: 10-50]")

    with col2:
        insulin = st.text_input("Insulin (µIU/mL) [Normal: 5-25]")
        dpf = st.text_input("Diabetes Pedigree Function [Normal: 0.1 - 2.5]")
        height_cm = st.text_input("Height (cm)")
        weight_kg = st.text_input("Weight (kg)")

    submitted = st.form_submit_button("Prediksi!", type="primary")

if submitted:
    try:
        # Validasi input jumlah kehamilan jika "Yes"
        if pregnancies_option == "Yes" and pregnancies.strip() == "":
            st.error("Silakan isi jumlah kehamilan.")
        else:
            # Hitung BMI
            height_m = float(height_cm) / 100
            bmi = float(weight_kg) / (height_m ** 2)

            # Ambil input dan ubah ke float
            input_data = np.array([[ 
                float(pregnancies), float(glucose), float(blood_pressure),
                float(skin_thickness), float(insulin), float(bmi),
                float(dpf), float(age)
            ]])

            # Scaling data
            input_scaled = scaler.transform(input_data)

            # Probabilitas prediksi
            proba = model.predict_proba(input_scaled)[0]
            threshold = 0.4  # bisa diubah sesuai performa model
            prediction = 1 if proba[1] > threshold else 0
            result_text = class_names[prediction]

            # Tampilkan hasil
            st.success(f"Hasil prediksi model: **{result_text}**")
            st.info(f"Probabilitas Tidak Diabetes: **{proba[0]:.2f}**, Diabetes: **{proba[1]:.2f}**")

            # Tampilkan detail input
            st.markdown("### Detail Input yang Anda Masukkan:")
            st.markdown(f"""
            - **Umur (Age)**: {age}
            - **Apakah Pernah Hamil?**: {pregnancies_option}    
            - **Jumlah Kehamilan (Pregnancies)**: {pregnancies}
            - **Kadar Glukosa (Glucose)**: {glucose}
            - **Tekanan Darah (Blood Pressure)**: {blood_pressure}
            - **Ketebalan Kulit (Skin Thickness)**: {skin_thickness}
            - **Insulin**: {insulin}
            - **Indeks Massa Tubuh (BMI)**: {bmi:.2f}
            - **Fungsi Riwayat Diabetes (DPF)**: {dpf}
            - **Tinggi Badan**: {height_cm} cm
            - **Berat Badan**: {weight_kg} kg
            """)

    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid dan lengkap.")

st.caption("Aplikasi Prediksi Diabetes menggunakan Streamlit & Scikit-learn")
