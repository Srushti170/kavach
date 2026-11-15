"""Kavach web API and demo page.

Run:
    .venv/Scripts/python -m uvicorn api.app:app --port 8000
Then open http://localhost:8000

Endpoints:
    POST /v1/scan       {"text": "..."} -> Kavach's verdict
    POST /demo/compare  {"text": "..."} -> Kavach and ProtectAI side by side (used by the demo page)
    GET  /health
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kavach import load_detector

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).parent / "static"
KAVACH_MODEL = ROOT / "models" / "kavach-small"
PROTECTAI_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
THRESHOLD = 0.5

detectors = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models once at startup, not on every request.
    detectors["kavach"] = load_detector(str(KAVACH_MODEL), max_length=256)
    detectors["protectai"] = load_detector(PROTECTAI_MODEL)
    yield
    detectors.clear()


app = FastAPI(title="Kavach", lifespan=lifespan)


class ScanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    threshold: float = Field(THRESHOLD, ge=0, le=1)


def run_detector(name: str, text: str, threshold: float) -> dict:
    start = time.perf_counter()
    score = detectors[name].score([text])[0]
    return {
        "score": round(score, 4),
        "is_injection": score >= threshold,
        "threshold": threshold,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
    }


@app.post("/v1/scan")
def scan(request: ScanRequest) -> dict:
    return run_detector("kavach", request.text, request.threshold)


@app.post("/demo/compare")
def compare(request: ScanRequest) -> dict:
    return {name: run_detector(name, request.text, request.threshold) for name in ("kavach", "protectai")}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": list(detectors)}


@app.api_route("/", methods=["GET", "HEAD"])
def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
