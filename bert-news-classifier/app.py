import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ── Load model directly ──────────────────────────────────────
MODEL_PATH = "zoha12/bert-ag-news-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Print actual labels from model config so we know exactly what they are
print("Model labels:", model.config.id2label)

# ── Label styling map (keyed by exact model label strings) ───
def get_label_style(label_str):
    label_lower = label_str.lower().strip()
    if "world" in label_lower:
        return ("World",    "🌍", "#3B82F6", "Geopolitics & International Affairs")
    elif "sport" in label_lower:
        return ("Sports",   "⚽", "#10B981", "Athletics & Competitions")
    elif "business" in label_lower or "finance" in label_lower:
        return ("Business", "💼", "#F59E0B", "Finance, Economy & Markets")
    elif "sci" in label_lower or "tech" in label_lower:
        return ("Sci/Tech", "🔬", "#8B5CF6", "Science, Technology & Innovation")
    else:
        return (label_str,  "📰", "#64748b", "News")

# ── Prediction function ──────────────────────────────────────
def classify_headline(text):
    if not text.strip():
        return ""

    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]

        # Build scores using actual model label strings
        num_labels = len(model.config.id2label)
        scores = []
        for i in range(num_labels):
            label_str = model.config.id2label[i]
            score = probs[i].item()
            scores.append((label_str, score))

        # Get top prediction
        top_label, top_score = max(scores, key=lambda x: x[1])
        name, icon, color, desc = get_label_style(top_label)
        confidence = top_score * 100

        # Build confidence bars
        bars_html = ""
        for label_str, score in scores:
            n, ic, col, _ = get_label_style(label_str)
            pct = score * 100
            bars_html += f"""
            <div style='margin: 8px 0;'>
              <div style='display:flex; justify-content:space-between;
                          font-family:monospace; font-size:13px;
                          color:#e2e8f0; margin-bottom:4px;'>
                <span>{ic} {n}</span>
                <span style='color:{col}; font-weight:700;'>{pct:.1f}%</span>
              </div>
              <div style='background:#1e293b; border-radius:999px; height:8px; overflow:hidden;'>
                <div style='width:{pct}%; background:{col}; height:100%;
                            border-radius:999px;'></div>
              </div>
            </div>"""

        # Build result card
        result_html = f"""
        <div style='font-family:sans-serif; background:#0f172a; border:1px solid #334155;
                    border-radius:16px; padding:28px; margin-top:8px;'>
            <div style='background:{color}22; border:1px solid {color}55;
                        border-radius:12px; padding:20px; margin-bottom:24px;
                        text-align:center;'>
                <div style='font-size:48px;'>{icon}</div>
                <div style='font-size:26px; font-weight:800; color:{color};'>{name}</div>
                <div style='font-size:13px; color:#94a3b8; margin-top:4px;'>{desc}</div>
                <div style='margin-top:12px; background:{color}33;
                            border:1px solid {color}66; border-radius:999px;
                            padding:4px 16px; font-size:14px; color:{color};
                            font-weight:700; display:inline-block;'>
                    {confidence:.1f}% confidence
                </div>
            </div>
            <div style='font-size:12px; color:#64748b; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:12px;'>
                All Category Scores
            </div>
            {bars_html}
        </div>"""

        return result_html

    except Exception as e:
        import traceback
        return f"<div style='color:red; padding:20px; font-family:monospace;'>❌ Error: {str(e)}<br><br>{traceback.format_exc()}</div>"


# ── Custom CSS ───────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;800&family=DM+Mono&display=swap');

body, .gradio-container {
    background: #0f172a !important;
    font-family: 'DM Sans', sans-serif !important;
}
.gradio-container {
    max-width: 720px !important;
    margin: 0 auto !important;
}
h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    background: linear-gradient(135deg, #60a5fa, #a78bfa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
}
.prose p {
    color: #94a3b8 !important;
    text-align: center !important;
    font-size: 15px !important;
}
textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
}
textarea:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 3px #3b82f622 !important;
}
button.primary {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.3px !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.88 !important; }
.examples table { background: #1e293b !important; border-radius: 10px !important; }
.examples td { color: #cbd5e1 !important; font-size: 13px !important; }
.examples tr:hover td { background: #334155 !important; color: #f1f5f9 !important; }
label span { color: #94a3b8 !important; font-size: 13px !important; }
"""

# ── Build Interface ──────────────────────────────────────────
with gr.Blocks(title="News Classifier") as demo:

    gr.Markdown("""
    # 📰 News Topic Classifier
    **BERT fine-tuned on AG News** — paste any headline to instantly classify it
    """)

    inp = gr.Textbox(
        lines=3,
        placeholder="e.g. Fed raises interest rates amid inflation fears...",
        label="News Headline",
        show_label=True
    )

    btn = gr.Button("Classify →", variant="primary")
    out = gr.HTML(label="Result")

    gr.Examples(
        examples=[
            ["Federal Reserve raises interest rates for third time this year"],
            ["Lionel Messi scores hat-trick to lead Argentina to victory"],
            ["Apple unveils M4 chip with breakthrough on-device AI features"],
            ["UN Security Council holds emergency session on Gaza conflict"],
            ["Tesla stock surges 12% after record quarterly deliveries"],
            ["Scientists discover potential signs of life on Europa moon"],
        ],
        inputs=inp,
        label="Try an example"
    )

    btn.click(fn=classify_headline, inputs=inp, outputs=out)
    inp.submit(fn=classify_headline, inputs=inp, outputs=out)

# ── Launch ───────────────────────────────────────────────────
demo.launch(css=custom_css)