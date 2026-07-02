"""
FastAPI Endpoint Tests
Validates API behavior without requiring trained model weights.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import app

client = TestClient(app)


class TestRootEndpoint:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_project_info(self):
        response = client.get("/")
        data = response.json()
        assert "project" in data
        assert data["project"] == "AstroVerse"


class TestTargetsEndpoint:
    def test_returns_200(self):
        response = client.get("/targets")
        assert response.status_code == 200

    def test_returns_list(self):
        response = client.get("/targets")
        data = response.json()
        assert isinstance(data, list)


class TestLightcurveEndpoint:
    def test_returns_200(self):
        response = client.get("/lightcurve/12345")
        assert response.status_code == 200

    def test_returns_phase_and_flux(self):
        response = client.get("/lightcurve/12345")
        data = response.json()
        assert "phase" in data
        assert "flux" in data
        assert len(data["phase"]) == len(data["flux"])


class TestPredictEndpoint:
    def test_rejects_missing_file(self):
        response = client.post("/predict")
        assert response.status_code == 422

    def test_rejects_invalid_csv(self):
        content = b"not,a,valid,csv\n1,2,3,4"
        response = client.post(
            "/predict",
            files={"file": ("test.csv", io.BytesIO(content), "text/csv")},
        )
        assert response.status_code == 400

    def test_rejects_csv_without_flux(self):
        content = b"time,brightness\n1.0,0.99\n2.0,1.01"
        response = client.post(
            "/predict",
            files={"file": ("test.csv", io.BytesIO(content), "text/csv")},
        )
        assert response.status_code == 400


class TestReportEndpoint:
    def test_returns_200(self):
        response = client.post(
            "/report",
            data={"tic_id": "12345", "probability": "85.5"},
        )
        assert response.status_code == 200

    def test_returns_markdown(self):
        response = client.post(
            "/report",
            data={"tic_id": "12345", "probability": "85.5"},
        )
        data = response.json()
        assert "markdown_content" in data
        assert "12345" in data["markdown_content"]
