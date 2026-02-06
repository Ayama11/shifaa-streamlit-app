import os
import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# أداء/استقرار على Streamlit Cloud
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

MODEL_ID = "aya99ma/shifaa-bert-classifier"

# تحويل LABEL_x إلى أسماء عربية (مطابق لترتيب label_encoder)
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

st.set_page_config(
    page_title="Shifaa Question Classifier",
    page_icon="🩺",
    layout="centered"
)

# CSS لتحسين المظهر + RTL
st.markdown("""
<style>
html, body, [class*="css"]  { direction: rtl; text-align: right; }
.block-container { padding-top: 2rem; max-width: 900px; }

h1, h2, h3 { letter-spacing: 0.2px; }

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    font-weight: 700;
}

.result-card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    background: rgba(255,255,255,0.03);
}

.small-muted { opacity: 0.75; font-size: 0.92rem; line-height: 1.35; }

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

with st.spinner("جاري تحميل الموديل... قد يستغرق ذلك بعض الوقت في أول تشغيل"):
    tokenizer, model = load_model()

# Header
st.title("🩺 تصنيف أسئلة شفاء الطبية")
st.markdown(
    '<div class="small-muted">'
    'نموذج لتصنيف الأسئلة الطبية إلى 16 فئة. '
    'مخصص للعرض الأكاديمي/البحثي ولا يُعد تشخيصًا طبيًا.'
    '</div>',
    unsafe_allow_html=True
)
st.divider()

# Layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    text = st.text_area(
        "اكتب السؤال الطبي هنا:",
        height=150,
        placeholder="مثال: لدي صداع شديد منذ يومين مع دوخة، ما السبب المحتمل؟"
    )

with col2:
    top_k = st.slider("عدد التصنيفات المعروضة", 1, 5, 3)
    show_all_probs = st.checkbox("عرض جميع الاحتمالات", value=False)
    classify = st.button("صنّف السؤال")

if classify:
    if not text.strip():
        st.warning("من فضلك اكتب سؤالًا أولًا.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        order = probs.argsort()[::-1]
        top_idx = order[:top_k]

        st.subheader("النتائج")

        # عرض Top-k كبطاقات + progress
        for i in top_idx:
            label_en = model.config.id2label[i]
            label_ar = LABELS_AR.get(label_en, label_en)
            p = float(probs[i])

            st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between; gap:10px; align-items:center;">
                    <div style="font-weight:800; font-size:1.05rem;">{label_ar}</div>
                    <div style="font-weight:800;">{p*100:.2f}%</div>
                </div>
                <div class="small-muted" style="margin-top:6px;">
                    التصنيف المتوقع بناءً على السؤال المُدخل.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(p)

        if show_all_probs:
            st.divider()
            st.markdown("### جميع الاحتمالات")
            for i in order:
                label_en = model.config.id2label[i]
                label_ar = LABELS_AR.get(label_en, label_en)
                p = float(probs[i])
                st.write(f"- {label_ar}: {p*100:.2f}%")
