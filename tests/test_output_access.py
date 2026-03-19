from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class OutputAccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        root = Path(cls.tempdir.name)
        workflows_dir = root / "workflows"
        runs_dir = root / "runs"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        workflow_name = "test_txt2img.json"
        (workflows_dir / workflow_name).write_text(
            json.dumps(
                {
                    "prompt": {
                        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sample"}},
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        env = {
            "API_TOKEN": "secret-token",
            "COMFYUI_BASE_URL": "http://127.0.0.1:8188",
            "COMFYUI_STARTUP_CHECK": "0",
            "DEFAULT_TXT2IMG_WORKFLOW": workflow_name,
            "DEFAULT_IMG2IMG_WORKFLOW": workflow_name,
            "DEFAULT_IMG2VIDEO_WORKFLOW": workflow_name,
            "ENABLE_WORKFLOW_WATCH": "0",
            "RUNS_DIR": str(runs_dir),
            "WORKER_CONCURRENCY": "1",
            "WORKFLOWS_DIR": str(workflows_dir),
        }
        cls.env_patcher = patch.dict(os.environ, env, clear=False)
        cls.env_patcher.start()

        import comfyui2api.app as app_module

        cls.app_module = importlib.reload(app_module)
        cls.app = cls.app_module.app
        cls.workflow_name = workflow_name
        cls.runs_dir = runs_dir

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env_patcher.stop()
        cls.tempdir.cleanup()

    def setUp(self) -> None:
        self.client_cm = TestClient(self.app)
        self.client = self.client_cm.__enter__()

    def tearDown(self) -> None:
        self.client_cm.__exit__(None, None, None)

    def test_job_detail_rewrites_output_urls_with_signed_query(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        job = Job(
            job_id="job-output-urls",
            created_at_utc="2026-03-19T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2img",
            workflow=self.workflow_name,
            url="/runs/job-output-urls/preview.png",
            outputs=[
                JobOutput(
                    filename="preview.png",
                    url="/runs/job-output-urls/preview.png",
                    media_type="image/png",
                    node_id="2",
                    output_key="images",
                )
            ],
        )

        with patch("comfyui2api.signed_urls.utc_now_unix", return_value=1_700_000_000):
            with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
                response = self.client.get(
                    "/v1/jobs/job-output-urls",
                    headers={"Authorization": "Bearer secret-token"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()["job"]
        for raw_url in (payload["url"], payload["outputs"][0]["url"]):
            parsed = urlparse(raw_url)
            params = parse_qs(parsed.query)
            self.assertEqual(parsed.scheme, "http")
            self.assertEqual(parsed.netloc, "testserver")
            self.assertEqual(parsed.path, "/runs/job-output-urls/preview.png")
            self.assertEqual(params["exp"], ["1700003600"])
            self.assertEqual(len(params["sig"][0]), 43)
            self.assertNotIn("api_key", params)

    def test_runs_output_requires_auth_and_accepts_signed_query(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        job_id = "job-runs-download"
        out_dir = self.runs_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "preview.png"
        out_path.write_bytes(b"fake-image")

        job = Job(
            job_id=job_id,
            created_at_utc="2026-03-19T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2img",
            workflow=self.workflow_name,
            outputs=[
                JobOutput(
                    filename="preview.png",
                    url=f"/runs/{job_id}/preview.png",
                    media_type="image/png",
                    node_id="2",
                    output_key="images",
                )
            ],
        )

        with patch("comfyui2api.signed_urls.utc_now_unix", return_value=1_700_000_000):
            with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
                signed_response = self.client.get(
                    "/v1/jobs/job-runs-download",
                    headers={"Authorization": "Bearer secret-token"},
                )

        signed_url = signed_response.json()["job"]["url"]
        signed_path = urlparse(signed_url).path
        signed_query = urlparse(signed_url).query

        with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
            unauthorized = self.client.get(f"/runs/{job_id}/preview.png")
            with patch("comfyui2api.signed_urls.utc_now_unix", return_value=1_700_000_000):
                response = self.client.get(
                    f"{signed_path}?{signed_query}",
                )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-image")
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_video_status_returns_signed_content_url_and_expires_at(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        job = Job(
            job_id="video-job",
            created_at_utc="2026-03-19T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2video",
            workflow=self.workflow_name,
            requested_model="video-model",
            outputs=[
                JobOutput(
                    filename="clip.mp4",
                    url="/runs/video-job/clip.mp4",
                    media_type="video/mp4",
                    node_id="2",
                    output_key="videos",
                )
            ],
        )

        with patch("comfyui2api.signed_urls.utc_now_unix", return_value=1_700_000_000):
            with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
                response = self.client.get(
                    "/v1/videos/video_video-job",
                    headers={"Authorization": "Bearer secret-token"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        parsed = urlparse(payload["url"])
        params = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/v1/videos/video_video-job/content")
        self.assertEqual(params["exp"], ["1700003600"])
        self.assertEqual(payload["expires_at"], 1700003600)

    def test_video_content_accepts_signed_query(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        job_id = "video-content-job"
        out_dir = self.runs_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "clip.mp4"
        out_path.write_bytes(b"fake-video")

        job = Job(
            job_id=job_id,
            created_at_utc="2026-03-19T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2video",
            workflow=self.workflow_name,
            outputs=[
                JobOutput(
                    filename="clip.mp4",
                    url=f"/runs/{job_id}/clip.mp4",
                    media_type="video/mp4",
                    node_id="2",
                    output_key="videos",
                )
            ],
        )

        with patch("comfyui2api.signed_urls.utc_now_unix", return_value=1_700_000_000):
            with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
                status_response = self.client.get(
                    "/v1/videos/video_video-content-job",
                    headers={"Authorization": "Bearer secret-token"},
                )

        signed_url = status_response.json()["url"]
        parsed = urlparse(signed_url)

        with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
            unauthorized = self.client.get("/v1/videos/video_video-content-job/content")
            with patch("comfyui2api.signed_urls.utc_now_unix", return_value=1_700_000_000):
                authorized = self.client.get(f"{parsed.path}?{parsed.query}")

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertEqual(authorized.content, b"fake-video")
        self.assertEqual(authorized.headers["content-type"], "video/mp4")

    def test_newapi_video_status_returns_signed_url_and_expires_at(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        job = Job(
            job_id="newapi-video-job",
            created_at_utc="2026-03-19T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2video",
            workflow=self.workflow_name,
            outputs=[
                JobOutput(
                    filename="clip.mp4",
                    url="/runs/newapi-video-job/clip.mp4",
                    media_type="video/mp4",
                    node_id="2",
                    output_key="videos",
                )
            ],
        )

        with patch("comfyui2api.signed_urls.utc_now_unix", return_value=1_700_000_000):
            with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
                response = self.client.get(
                    "/v1/video/generations/newapi-video-job",
                    headers={"Authorization": "Bearer secret-token"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        parsed = urlparse(payload["url"])
        params = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/v1/videos/video_newapi-video-job/content")
        self.assertEqual(params["exp"], ["1700003600"])
        self.assertEqual(payload["expires_at"], 1700003600)


if __name__ == "__main__":
    unittest.main()
