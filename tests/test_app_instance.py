from __future__ import annotations

import os
import subprocess
import sys
import time

from lerobot_studio.app import acquire_instance_lock


def test_lock_can_be_checked_without_replacing_current_process(tmp_path):
    lock_path = tmp_path / "instance.lock"
    first = acquire_instance_lock(lock_path)
    assert first is not None
    try:
        assert acquire_instance_lock(lock_path) is None
    finally:
        first.unlock()

    replacement = acquire_instance_lock(lock_path)
    assert replacement is not None
    replacement.unlock()


def test_new_launch_replaces_running_application(tmp_path):
    config = tmp_path / "config"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "LEROBOT_STUDIO_CONFIG": str(config),
            "LEROBOT_STUDIO_WORKSPACE": str(workspace),
        }
    )
    command = [sys.executable, "-m", "lerobot_studio.app"]
    first = subprocess.Popen(command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    second = None
    try:
        lock_path = config / "instance.lock"
        deadline = time.monotonic() + 8
        while not lock_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert lock_path.exists()
        time.sleep(0.3)

        second = subprocess.Popen(
            command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        assert first.wait(timeout=10) == 0
        time.sleep(0.5)
        assert second.poll() is None
    finally:
        if first.poll() is None:
            first.terminate()
            first.wait(timeout=5)
        if second is not None and second.poll() is None:
            second.terminate()
            second.wait(timeout=5)
