import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from PIL import Image
import numpy as np
import pickle

# Tải mô hình và tokenizer
@st.cache_resource
def load_caption_model():
    return load_model('inception_caption_model_lr_0.0001.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_extractor():
    # Sử dụng InceptionV3 để trích xuất đặc trưng, không bao gồm lớp cuối
    model = InceptionV3(weights='imagenet')
    model = tf.keras.Model(model.input, model.layers[-2].output)
    return model

# Hàm trích xuất đặc trưng từ hình ảnh
def extract_features(image, feature_extractor):
    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_extractor.predict(image, verbose=0)
    return feature

# Hàm dự đoán mô tả hình ảnh
def generate_caption(model, image_features, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Ứng dụng Streamlit
st.title("Image Caption Generator")

# Tải hình ảnh từ người dùng
uploaded_file = st.file_uploader("Chọn một hình ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị hình ảnh đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Hình ảnh đã tải lên.', use_column_width=True)
    
    # Tải mô hình
    model = load_caption_model()
    tokenizer = load_tokenizer()
    feature_extractor = load_feature_extractor()
    
    # Trích xuất đặc trưng từ ảnh
    image_features = extract_feature
