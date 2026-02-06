import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "aya99ma/shifaa-bert-classifier"

st.set_page_config(page_title="Shifaa Question Classifier", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🩺 تصنيف أسئلة شفاء الطبية")
text = st.text_area("اكتب السؤال الطبي هنا:", height=120)

top_k = st.slider("عدد التصنيفات المعروضة", 1, 5, 3)

if st.button("صنّف السؤال"):
    if not text.strip():
        st.warning("من فضلك اكتب سؤالًا أولًا.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top_idx = probs.argsort()[::-1][:top_k]
        st.subheader("النتائج:")
        for i in top_idx:
            st.write(f"- **{model.config.id2label[i]}** : {probs[i]*100:.2f}%")
