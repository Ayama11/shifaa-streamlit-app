import os
import time
import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===== Performance / Stability =====
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

MODEL_ID = "aya99ma/shifaa-bert-classifier"

GITHUB_REPO_URL = "https://github.com/Ayama11/shifaa-streamlit-app"
HF_MODEL_URL = f"https://huggingface.co/{MODEL_ID}"

# Model metrics (as provided)
METRICS = {
    "Accuracy": 0.82,
    "F1-macro": 0.70,
}

# Arabic labels mapping (from your label_encoder)
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
    layout="centered",
)

# ===== Minimal, responsive RTL styling (no fixed widths) =====
st.markdown("""
<style>
html, body, [class*="css"]  { direction: rtl; text-align: right; }
.block-container { padding-top: 1.8rem; max-width: 920px; }

.small-muted { opacity: 0.78; font-size: 0.95rem; line-height: 1.5; }
.kpi { opacity: 0.88; font-size: 0.92rem; }

.card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.03);
    margin-bottom: 12px;
}

.card-strong {
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.05);
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
    font-size: 0.85rem;
    opacity: 0.9;
}

div.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.7rem 1rem;
    font-weight: 800;
}

/* Link buttons (Streamlit) - small polish without breaking responsiveness */
div[data-testid="stLinkButton"] a {
    border-radius: 14px !important;
    font-weight: 800 !important;
    padding: 0.7rem 1rem !important;
}

a.cleanlink { text-decoration: none; }
a.cleanlink:hover { text-decoration: underline; }

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    mdl.eval()
    return tok, mdl

with st.spinner("جاري تحميل الموديل... (قد يستغرق ذلك أول مرة)"):
    tokenizer, model = load_model()

# ===== Header =====
st.title("🩺 تصنيف أسئلة موقع شفاء الطبية")
st.markdown(
    '<div class="small-muted">'
    'نموذج لتصنيف الأسئلة الطبية إلى <b>16</b> فئة باستخدام BERT. '
    'للأغراض البحثية/العرض فقط.'
    '</div>',
    unsafe_allow_html=True
)

# ✅ Responsive interactive links (won't overflow on mobile/desktop)
st.link_button("🐙 GitHub Repo", GITHUB_REPO_URL, use_container_width=True)
st.link_button("🤗 HuggingFace Model", HF_MODEL_URL, use_container_width=True)

st.divider()

# ===== Input Section (Stacked for mobile friendliness) =====
st.markdown('<div class="card">', unsafe_allow_html=True)

question = st.text_area(
    "اكتب السؤال الطبي هنا:",
    height=160,
    placeholder="مثال: لدي صداع شديد منذ يومين مع دوخة ؟"
)

top_k = st.slider("عدد التصنيفات المعروضة", 1, 5, 3)
show_all_probs = st.checkbox("عرض جميع الاحتمالات", value=False)

classify = st.button("صنّف السؤال")

st.markdown('</div>', unsafe_allow_html=True)

# ===== Inference =====
if classify:
    if not question.strip():
        st.warning("من فضلك اكتب سؤالًا أولًا.")
    else:
        t0 = time.perf_counter()

        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        ms = (time.perf_counter() - t0) * 1000.0

        order = probs.argsort()[::-1]
        top_idx = order[:top_k]

        # Top-1
        i0 = int(top_idx[0])
        label0_en = model.config.id2label[i0]
        label0_ar = LABELS_AR.get(label0_en, label0_en)
        p0 = float(probs[i0])

        st.subheader("النتائج")

        st.markdown(
            f"""
            <div class="card card-strong">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                <div style="font-weight:900; font-size:1.08rem;">{label0_ar}</div>
                <div class="badge">{p0*100:.2f}%</div>
              </div>
              <div class="kpi" style="margin-top:8px;">
                زمن الاستجابة: <b>{ms:.0f} ms</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(p0)

        # Remaining Top-k
        if len(top_idx) > 1:
            st.markdown("##### أعلى تصنيفات أخرى")
            for i in top_idx[1:]:
                i = int(i)
                label_en = model.config.id2label[i]
                label_ar = LABELS_AR.get(label_en, label_en)
                p = float(probs[i])

                st.markdown(
                    f"""
                    <div class="card">
                      <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                        <div style="font-weight:800;">{label_ar}</div>
                        <div class="badge">{p*100:.2f}%</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(p)

        # Optional: show all probabilities
        if show_all_probs:
            st.divider()
            st.markdown("### جميع الاحتمالات")
            for i in order:
                i = int(i)
                label_en = model.config.id2label[i]
                label_ar = LABELS_AR.get(label_en, label_en)
                p = float(probs[i])
                st.write(f"- {label_ar}: {p*100:.2f}%")

# ===== Footer / Project Info (Bottom, responsive) =====
st.divider()

st.markdown(
    f"""
<div class="card">
  <div style="font-weight:900; font-size:1.05rem; margin-bottom:6px;">عن المشروع</div>

  <div class="small-muted">
    هذا العمل يقدّم نموذجًا لتصنيف أسئلة منصة شفاء الطبية إلى 16 فئة باستخدام BERT.
    الهدف هو عرض تجربة NLP كاملة تشمل التدريب، التقييم، ثم نشر واجهة تفاعلية عبر Streamlit.
  </div>

  <div style="margin-top:12px;">
    <span class="badge">Accuracy ≈ {METRICS["Accuracy"]:.2f}</span>
    &nbsp;
    <span class="badge">F1-macro ≈ {METRICS["F1-macro"]:.2f}</span>
  </div>

  <div class="small-muted" style="margin-top:12px;">
    <b>الموديل:</b> <a class="cleanlink" href="{HF_MODEL_URL}" target="_blank">{MODEL_ID}</a>
    &nbsp; | &nbsp;
    <b>المصدر:</b> <a class="cleanlink" href="{GITHUB_REPO_URL}" target="_blank">GitHub</a>
  </div>

  <div class="small-muted" style="margin-top:10px;">
    تنبيه: هذا النموذج يصنّف السؤال حسب الفئة الطبية المتوقعة فقط، ولا يقدّم تشخيصًا أو توصيات علاجية.
  </div>
</div>
""",
    unsafe_allow_html=True
)
