import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


# Carregar o modelo treinado
model = load_model('melanona.h5')  # Substitua pelo caminho correto

# Função para preparar a imagem para a predição
def prepare_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))  # Ajuste o tamanho conforme necessário
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar batch dimension
    img_array /= 255.0  # Normalização, se necessário
    return img_array

# Função de predição
def predict_image(img_array):
    prediction = model.predict(img_array)
    if prediction[0][0] >= 0.5:
        return "Maligno", prediction[0][0]
    else:
        return "Benigno", prediction[0][0]

# Configurações da página Streamlit
st.set_page_config(page_title="Melanona Classification", layout="wide")

st.title("Classificação de imagens de melanoma")
st.write("Faça o upload de uma imagem de melanoma para predição.")

# Upload de imagem pelo usuário
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar a imagem carregada
    st.image(uploaded_file, caption="Imagem carregada.", use_container_width=True)

    # Preparar a imagem
    img_array = prepare_image(uploaded_file)

    # Fazer a predição
    label, confidence = predict_image(img_array)

    # Exibir o resultado da predição
    st.write(f"**Predição**: {label} ({confidence*100:.2f}% de confiança)")

# Rodapé com link para o LinkedIn
st.markdown(
    """
    <hr style='border:1px solid #e3e3e3;margin-top:40px'>
    <div style='text-align: center;'>
        Desenvolvida por 
        <a href='https://www.linkedin.com/in/tairone-amaral/' target='_blank'>
            Tairone Leandro do Amaral
        </a>
    </div>
    """,
    unsafe_allow_html=True
)