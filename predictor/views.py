from pathlib import Path
import pickle

import torch
import torch.nn as nn
from django.shortcuts import render

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
WEIGHTS_PATH = BASE_DIR / "best_rnn_weights.pth"


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def _load_vectorizer():
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Missing vectorizer: {VECTORIZER_PATH}")
    with VECTORIZER_PATH.open("rb") as f:
        return pickle.load(f)


def _load_model(input_size):
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}")
    model = RNN(input_size)
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


_VECTORIZER = _load_vectorizer()
_INPUT_SIZE = len(_VECTORIZER.get_feature_names_out())
_MODEL = _load_model(_INPUT_SIZE)


def _predict_sentiment(text):
    vec = _VECTORIZER.transform([text])
    x = torch.from_numpy(vec.toarray()).float().unsqueeze(1)
    with torch.no_grad():
        logits = _MODEL(x)
        prob = torch.sigmoid(logits.squeeze()).item()
    label = 1 if prob >= 0.5 else 0
    return label, prob


def index(request):
    result = None
    prob = None
    text = ""

    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        if text:
            label, prob = _predict_sentiment(text)
            result = "Positive" if label == 1 else "Negative"

    context = {
        "result": result,
        "prob": prob,
        "text": text,
    }
    return render(request, "predictor/index.html", context)
