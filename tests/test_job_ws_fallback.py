from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class JobWebsocketFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_job_completes_when_ws_monitor_fails_but_history_succeeds(self) -> None:
        import comfyui2api.jobs as jobs_module
        from comfyui2api.comfy_client import QueuedPrompt

        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            manager = jobs_module.JobManager(
                cfg=SimpleNamespace(
                    worker_concurrency=1,
                    runs_dir=runs_dir,
                    timeout_s=30,
                    poll_interval_s=0.01,
                ),
                registry=SimpleNamespace(),
                comfy=SimpleNamespace(
                    object_info=AsyncMock(return_value={}),
                    queue_prompt=AsyncMock(return_value=QueuedPrompt(prompt_id="prompt-1", client_id="client-1", number=1)),
                    wait_for_history_complete=AsyncMock(
                        return_value={
                            "outputs": {
                                "save-node": {
                                    "images": [
                                        {
                                            "filename": "result.png",
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            }
                        }
                    ),
                    view_bytes=AsyncMock(return_value=b"png-bytes"),
                ),
            )

            workflow = SimpleNamespace(
                name="test.json",
                parameter_spec=None,
                clone_obj=lambda: {
                    "prompt": {
                        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sample"}},
                    }
                },
            )

            job = await manager.create_job(kind="txt2img", workflow="test.json", prompt="hello")

            with patch.object(manager, "_resolve_workflow", AsyncMock(return_value=workflow)), patch.object(
                jobs_module, "prepare_prompt", return_value=({"1": {}}, None, [], {"positive": [], "negative": []})
            ), patch.object(
                manager, "_monitor_ws", AsyncMock(side_effect=ConnectionError("ws closed"))
            ), patch.object(
                jobs_module.logger, "warning"
            ) as mock_warning:
                await manager._run_job(job.job_id)

        updated = await manager.get_job(job.job_id)
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.status, "completed")
        self.assertTrue(updated.done.is_set())
        self.assertEqual(len(updated.outputs), 1)
        self.assertEqual(updated.outputs[0].filename, "save-node__result.png")
        self.assertEqual(updated.url, "/runs/" + job.job_id + "/save-node__result.png")
        mock_warning.assert_called_once()
        self.assertIn("job websocket monitor failed, continuing with history polling", mock_warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
