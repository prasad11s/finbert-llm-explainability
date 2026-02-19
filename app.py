import gradio as gr
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

finbert = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

label_to_idx = {'positive': 0, 'negative': 1, 'neutral': 2}
class_names = ['positive', 'negative', 'neutral']

def predict(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1).detach().numpy()

def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1).detach().numpy()

def analyze(headline):
    # Sentiment prediction
    scores = finbert(headline[:512])[0]
    top = max(scores, key=lambda x: x['score'])
    sentiment = top['label'].upper()
    confidence = f"{top['score']*100:.1f}%"

    # SHAP
    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(predict, masker)
    sv = explainer([headline])
    pred_idx = label_to_idx[top['label']]
    words = sv[0].data
    shap_vals = sv[0].values[:, pred_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Sentiment: {sentiment} ({confidence})', fontsize=13, fontweight='bold')

    # SHAP plot
    colors_shap = ['green' if v > 0 else 'red' for v in shap_vals]
    axes[0].bar(range(len(words)), shap_vals, color=colors_shap)
    axes[0].set_xticks(range(len(words)))
    axes[0].set_xticklabels(words, rotation=30, ha='right', fontsize=9)
    axes[0].set_title('SHAP: Word Contributions')
    axes[0].set_ylabel('SHAP Value')
    axes[0].axhline(0, color='black', linewidth=0.8)

    # LIME
    lime_explainer = LimeTextExplainer(class_names=class_names)
    exp = lime_explainer.explain_instance(headline, predict_proba,
                                          num_features=len(headline.split()),
                                          num_samples=300, labels=[0, 1, 2])
    lime_words = [w for w, _ in exp.as_list(label=pred_idx)]
    lime_scores = [s for _, s in exp.as_list(label=pred_idx)]

    colors_lime = ['green' if s > 0 else 'red' for s in lime_scores]
    axes[1].bar(range(len(lime_words)), lime_scores, color=colors_lime)
    axes[1].set_xticks(range(len(lime_words)))
    axes[1].set_xticklabels(lime_words, rotation=30, ha='right', fontsize=9)
    axes[1].set_title('LIME: Word Contributions')
    axes[1].set_ylabel('LIME Score')
    axes[1].axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    return fig

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Financial Headline", placeholder="e.g. JPMorgan reports record profits in Q3 earnings"),
    outputs=gr.Plot(label="SHAP vs LIME Explainability"),
    title="FinBERT LLM Explainability: SHAP vs LIME",
    description="Enter a financial headline to see FinBERT's sentiment prediction and word-level explanations from both SHAP and LIME.",
    examples=[
        ["JPMorgan reports record profits in Q3 earnings"],
        ["Federal Reserve raises interest rates amid inflation fears"],
        ["Goldman Sachs cuts 3000 jobs in major restructuring"],
        ["Apple stock surges after strong iPhone sales report"]
    ]
)

demo.launch()