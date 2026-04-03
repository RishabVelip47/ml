import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# Cache heavy resources — loaded once, reused every run
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

@st.cache_resource
def load_stopwords():
    return set(stopwords.words('english'))  # set = O(1) lookup vs list O(n)

STOP_WORDS  = load_stopwords()
PUNCTUATION = set(string.punctuation)

def transform_text(text):
    tokens  = nltk.word_tokenize(text.lower())
    cleaned = [
        ps.stem(w)
        for w in tokens
        if w.isalnum() and w not in STOP_WORDS and w not in PUNCTUATION
    ]
    return " ".join(cleaned)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,80,80,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,160,50,0.08) 0%, transparent 60%),
        #0a0a0f !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { max-width: 860px !important; padding: 2rem 3rem 4rem !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #ff5050;
    border: 1px solid rgba(255,80,80,0.35);
    border-radius: 2px;
    padding: 4px 12px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f0f0fa;
    margin: 0 0 0.6rem;
    white-space: nowrap;
}
.hero-title span { color: #ff5050; }
.hero-sub {
    font-size: 0.82rem;
    color: #6a6a80;
    letter-spacing: 0.04em;
}

/* ── Divider ── */
.slash-divider {
    text-align: center;
    color: rgba(255,80,80,0.3);
    font-size: 1.2rem;
    letter-spacing: 0.6em;
    margin: 1.6rem 0;
}

/* ── Textarea override ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 4px !important;
    color: #d0d0e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
    caret-color: #ff5050;
    transition: border-color 0.2s;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: rgba(255,80,80,0.5) !important;
    box-shadow: 0 0 0 3px rgba(255,80,80,0.06) !important;
}
.stTextArea label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #5a5a70 !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: #ff5050 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    background: #e03a3a !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result cards ── */
.result-card {
    margin-top: 2rem;
    padding: 2rem 2.2rem;
    border-radius: 4px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
}
.result-spam {
    background: rgba(255, 50, 50, 0.06);
    border: 1px solid rgba(255, 50, 50, 0.3);
}
.result-spam::before {
    box-shadow: inset 0 0 60px rgba(255,50,50,0.04);
}
.result-safe {
    background: rgba(50, 210, 130, 0.06);
    border: 1px solid rgba(50, 210, 130, 0.3);
}
.result-safe::before {
    box-shadow: inset 0 0 60px rgba(50,210,130,0.04);
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-spam .result-label { color: rgba(255,100,100,0.7); }
.result-safe .result-label { color: rgba(50,210,130,0.7); }

.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}
.result-spam .result-verdict { color: #ff5050; }
.result-safe .result-verdict { color: #32d282; }

.result-desc {
    font-size: 0.78rem;
    color: #5a5a70;
    margin-top: 0.7rem;
    line-height: 1.6;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}
.stat-pill {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 3px;
    padding: 0.7rem 1rem;
    text-align: center;
}
.stat-pill .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0f0;
}
.stat-pill .key {
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a4a60;
    margin-top: 2px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3.5rem;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #2e2e42;
    text-transform: uppercase;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">ML-Powered Detection</div>
    <div class="hero-title">Spam<span>Shield</span></div>
    <div class="hero-sub">paste any message below — we'll tell you if it's a threat</div>
</div>
<div class="slash-divider">/ / /</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
input_sms = st.text_area(
    "Message content",
    placeholder="Paste the email or SMS message here…",
    height=180,
    label_visibility="visible"
)

analyze_clicked = st.button("Analyze Message →")

# ── Prediction ────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not input_sms.strip():
        st.warning("Please enter a message to analyze.")
    else:
        try:
            tfidf, model = load_model()

            # Preprocess
            transformed = transform_text(input_sms)
            token_count = len(transformed.split())
            char_count  = len(input_sms)

            # Vectorize & predict
            vector_input = tfidf.transform([transformed])
            result       = model.predict(vector_input)[0]

            # Try to get probability if model supports it
            confidence_str = "—"
            try:
                proba = model.predict_proba(vector_input)[0]
                confidence_str = f"{max(proba)*100:.1f}%"
            except Exception:
                pass

            # ── Result card ──
            if result == 1:
                st.markdown(f"""
                <div class="result-card result-spam">
                    <div class="result-label">⚠ Classification result</div>
                    <div class="result-verdict">SPAM</div>
                    <div class="result-desc">
                        This message exhibits characteristics commonly associated with spam or
                        phishing attempts. Do not click any links or provide personal information.
                    </div>
                    <div class="stats-row">
                        <div class="stat-pill">
                            <div class="val">{confidence_str}</div>
                            <div class="key">Confidence</div>
                        </div>
                        <div class="stat-pill">
                            <div class="val">{token_count}</div>
                            <div class="key">Tokens</div>
                        </div>
                        <div class="stat-pill">
                            <div class="val">{char_count}</div>
                            <div class="key">Characters</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-safe">
                    <div class="result-label">✓ Classification result</div>
                    <div class="result-verdict">NOT SPAM</div>
                    <div class="result-desc">
                        This message appears to be legitimate. No spam indicators were detected
                        by the classifier. Always stay cautious with unexpected messages.
                    </div>
                    <div class="stats-row">
                        <div class="stat-pill">
                            <div class="val">{confidence_str}</div>
                            <div class="key">Confidence</div>
                        </div>
                        <div class="stat-pill">
                            <div class="val">{token_count}</div>
                            <div class="key">Tokens</div>
                        </div>
                        <div class="stat-pill">
                            <div class="val">{char_count}</div>
                            <div class="key">Characters</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.error("⚠ Model files not found. Make sure `vectorizer.pkl` and `model.pkl` are in the same directory.")
        except Exception as e:
            st.error(f"⚠ An error occurred: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">SpamShield · NLP Spam Classifier · Model: TF-IDF + ML</div>
""", unsafe_allow_html=True)