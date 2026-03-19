from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

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

    def test_job_detail_rewrites_output_urls_with_api_key_query(self) -> None:
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

        with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
            response = self.client.get(
                "/v1/jobs/job-output-urls",
                headers={"Authorization": "Bearer secret-token"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()["job"]
        self.assertEqual(
            payload["url"],
            "http://testserver/runs/job-output-urls/preview.png?api_key=secret-token",
        )
        self.assertEqual(
            payload["outputs"][0]["url"],
            "http://testserver/runs/job-output-urls/preview.png?api_key=secret-token",
        )

    def test_runs_output_requires_auth_and_accepts_query_api_key(self) -> None:
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

        with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
            unauthorized = self.client.get(f"/runs/{job_id}/preview.png")
            authorized = self.client.get(f"/runs/{job_id}/preview.png?api_key=secret-token")

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertEqual(authorized.content, b"fake-image")
        self.assertEqual(authorized.headers["content-type"], "image/png")


if __name__ == "__main__":
    unittest.main()
