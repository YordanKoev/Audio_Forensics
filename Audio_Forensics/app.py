import base64
import io
import os
import time

import numpy as np
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, url_for

from analysis.ai_agent import AIDetectionAgent
from analysis.audio_io import load_wav
from analysis.labels import display_value, label
from analysis.manipulation import analyze_manipulation
from analysis.plots import ai_plots, manipulation_plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "diploma-dev-key")
app.jinja_env.filters["label"] = label
app.jinja_env.filters["display_value"] = display_value

agent = AIDetectionAgent()


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_analysis(data: bytes, filename: str) -> tuple:
    t0 = time.time()
    audio = load_wav(io.BytesIO(data))
    manip = to_jsonable(analyze_manipulation(audio))
    ai = to_jsonable(agent.run(audio))
    report = {
        "filename": filename,
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "info": to_jsonable(audio.info),
        "manipulation": {k: v for k, v in manip.items() if k != "series"},
        "ai_detection": {k: v for k, v in ai.items() if k not in ("series", "deep")},
    }
    report["elapsed_s"] = round(time.time() - t0, 2)
    plots = {"manipulation": manipulation_plots(audio, manip), "ai": ai_plots(audio, ai)}
    return report, plots


def _list_samples():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".wav"))


@app.route("/")
def index():
    return render_template("index.html", samples=_list_samples(), model_loaded=agent.model is not None,
                           deep_loaded=agent.deep.available)


def _read_upload():
    file = request.files.get("audio")
    if file is None or file.filename == "":
        raise ValueError("Не е избран файл.")
    if not file.filename.lower().endswith(".wav"):
        raise ValueError("Поддържат се само файлове във формат .wav.")
    return file.read(), file.filename


def _render_result(data: bytes, report: dict, plots: dict):
    audio_src = "data:audio/wav;base64," + base64.b64encode(data).decode("ascii")
    return render_template("result.html", r=report, plots=plots, audio_src=audio_src)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data, name = _read_upload()
        report, plots = run_analysis(data, name)
    except Exception as exc:
        flash(f"Грешка при анализа: {exc}")
        return redirect(url_for("index"))
    return _render_result(data, report, plots)


@app.route("/sample/<name>")
def analyze_sample(name):
    if name not in _list_samples():
        abort(404)
    with open(os.path.join(SAMPLES_DIR, name), "rb") as fh:
        data = fh.read()
    report, plots = run_analysis(data, name)
    return _render_result(data, report, plots)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        data, name = _read_upload()
        report, _ = run_analysis(data, name)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(report)


@app.errorhandler(413)
def too_large(_):
    flash("Файлът е твърде голям (максимум 100 MB).")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)