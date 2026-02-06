import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "aya99ma/shifaa-bert-classifier"

# تحويل LABEL_x إلى أسماء عربية
LABELS_AR = {
    "LABEL_0": "أمراض الأطفال ومشاكلهم",
    "LABEL_1": "أمراض الباطنية والصدر",
    "LABEL_2": "أمراض الجلدية",
    "LABEL_3": "أمراض الجهاز البولي والتناسلي",
    "LABEL_4": "أمراض الجهاز العصبي",
    "LABEL_5": "أمراض الدم والأورام",
    "LABEL_6": "أمراض الرأس",
    "LABEL_7": "أمراض العضلات",
    "LABEL_8": "أمراض العظام",
    "LABEL_9": "أمراض الغدد والهرمونات",
    "LABEL_10": "أمراض النساء والولادة",
    "LABEL_11": "الأدوية والمستحضرات",
    "LABEL_12": "الجراحة العامة والتجميل",
    "LABEL_13": "الصحة البدنية",
    "LABEL_14": "الطب البديل",
    "LABEL_15": "شئون طبية ومشاكل متفرقة",
}

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
            label_en = model.config.id2label[i]
            label_ar = LABELS_AR.get(label_en, label_en)
            st.write(f"- **{label_ar}** : {probs[i]*100:.2f}%")
