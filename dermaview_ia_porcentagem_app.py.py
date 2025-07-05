
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="DermaView – Simulador Estético com IA", layout="centered")
st.title("💄 DermaView – Simulador Estético com IA Avançada")
st.markdown("Ajuste procedimentos estéticos faciais com precisão de forma natural.")

uploaded_file = st.file_uploader("📸 Faça upload da imagem (rosto de frente, boa iluminação):", type=["jpg", "jpeg", "png"])

# Sliders para controle de porcentagem dos efeitos
olheira_pct = st.slider("Reduzir olheiras (%)", 0, 100, 0)
bochecha_pct = st.slider("Aumentar bochechas (%)", 0, 100, 0)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_copy = img_np.copy()

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = img_np.shape

                # Posição dos olhos para olheiras
                if olheira_pct > 0:
                    under_eye_indices = [145, 159, 160, 144, 153, 154]
                    for idx in under_eye_indices:
                        x = int(face_landmarks.landmark[idx].x * iw)
                        y = int(face_landmarks.landmark[idx].y * ih)
                        radius = int(10 + (olheira_pct / 100) * 15)
                        cv2.circle(img_copy, (x, y+10), radius, (255, 255, 255), -1)

                # Região das bochechas
                if bochecha_pct > 0:
                    cheek_indices = [50, 280]
                    for idx in cheek_indices:
                        x = int(face_landmarks.landmark[idx].x * iw)
                        y = int(face_landmarks.landmark[idx].y * ih)
                        radius = int(8 + (bochecha_pct / 100) * 20)
                        cv2.circle(img_copy, (x, y), radius, (255, 180, 200), -1)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Antes", use_column_width=True)
    with col2:
        st.image(img_copy, caption="Depois", use_column_width=True)
