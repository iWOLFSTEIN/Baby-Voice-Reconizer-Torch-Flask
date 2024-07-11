from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import librosa
import numpy as np
import os

from keys import AUTH_KEY

app = Flask(__name__)


class CNN_Baby(nn.Module):
    def __init__(self):
        super(CNN_Baby, self).__init__()
        self.fc1 = nn.Linear(1025 * 315, 4)
        self.fc3 = nn.Linear(500, 4)

    def forward(self, x):
        out = self.fc1(x)
        return out


model = CNN_Baby()
model_path = "neural_network_20.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()


def readFile(filepath):
    y, sr = librosa.load(filepath)
    D = librosa.stft(y)
    D_real, D_imag = np.real(D), np.imag(D)
    D_energy = np.sqrt(D_real**2 + D_imag**2)
    norm = librosa.util.normalize(D_energy)
    if norm.shape[1] < 315:
        norm = np.pad(norm, ((0, 0), (0, 315 - norm.shape[1])), "constant")
    else:
        norm = norm[:, :315]
    return norm


def get_label_index(tensor_pred):
    npray = tensor_pred.detach().numpy()
    index = np.argmax(npray)
    return index


@app.route("/predict", methods=["POST"])
def predict():
    if "auth" not in request.headers or request.headers["auth"] != AUTH_KEY:
        return jsonify({"error": "Invalid auth key"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join("./uploads", file.filename)
        file.save(filepath)

        f = readFile(filepath)
        f_tensor = torch.tensor(f).contiguous().view(-1, 1025 * 315).float()

        with torch.no_grad():
            p = model(f_tensor)

        index = get_label_index(p)
        labels = ["Silence", "Noise", "Baby laugh", "Crying baby"]
        prediction = labels[index]

        os.remove(filepath)

        is_baby_sound = prediction in ["Baby laugh", "Crying baby"]

        return jsonify({"prediction": is_baby_sound})


if __name__ == "__main__":
    if not os.path.exists("./uploads"):
        os.makedirs("./uploads")
    # app.run(host="0.0.0.0", port=5000, debug=True)
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
