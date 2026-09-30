"""Serve the FlowGuard form and predictions from trusted scikit-learn pickles."""

import json
import math
import os
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


ROOT = Path(__file__).resolve().parent
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_REQUEST_BYTES = 16_384


class PredictionEngine:
    def __init__(self, bundle):
        if not isinstance(bundle, dict) or bundle.get("artifact_version") != 1:
            raise ValueError("Unsupported model.pkl format")
        features = bundle.get("features")
        models = bundle.get("models")
        if (not isinstance(features, list) or not features or
                len(features) != len(set(features)) or
                not isinstance(models, dict) or
                set(models) != {"random_forest", "logistic_regression"}):
            raise ValueError("model.pkl has incomplete model or feature metadata")
        self.features = tuple(features)
        self.models = models
        self.evaluation = bundle.get("evaluation", {})

    def status(self):
        return {
            "ready": True,
            "features": list(self.features),
            "models": list(self.models),
            "evaluation": self.evaluation,
        }

    def predict(self, values):
        if not isinstance(values, dict) or set(values) != set(self.features):
            raise ValueError(f"Provide exactly the {len(self.features)} named flow features")
        ordered = []
        for name in self.features:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be a finite number")
            ordered.append(number)
        frame = pd.DataFrame([ordered], columns=self.features)
        result = {}
        for name, model in self.models.items():
            predicted = int(model.predict(frame)[0])
            if predicted not in (0, 1):
                raise ValueError("Model returned an unknown class")
            result[name] = "DDoS" if predicted == 1 else "BENIGN"
        return result


class FlowGuardHandler(BaseHTTPRequestHandler):
    engine = None
    static_dir = ROOT

    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, data):
        self.send_bytes(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            self.send_json(200, self.engine.status())
            return
        if path not in STATIC_FILES:
            self.send_json(404, {"error": "Page not found"})
            return
        filename, content_type = STATIC_FILES[path]
        file = self.static_dir / filename
        if not file.is_file():
            self.send_json(404, {"error": "Page not found"})
            return
        self.send_bytes(200, file.read_bytes(), content_type)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/predict":
            self.send_json(404, {"error": "Endpoint not found"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "Send JSON input"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request size is invalid")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or set(data) != {"features"}:
                raise ValueError("Expected a features object")
            prediction = self.engine.predict(data["features"])
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        self.send_json(200, prediction)


def main():
    with (ROOT / "model.pkl").open("rb") as handle:
        engine = PredictionEngine(pickle.load(handle))
    handler = type("ConfiguredHandler", (FlowGuardHandler,), {"engine": engine})
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    with ThreadingHTTPServer((host, port), handler) as server:
        print(f"FlowGuard ready at http://{host}:{port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
