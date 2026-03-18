import os
import subprocess
import threading
import time
from typing import Optional

from logger import SYSTEM_LOGGER

LOGGER = SYSTEM_LOGGER


class GDriveUploader:
    """Non-blocking uploader that shells out to rclone.

    Usage:
        uploader = GDriveUploader(remote="gdrive:", root_dir="voice2chat")
        done_event = uploader.start_upload(
            local_path="/tmp/audio.wav",
            remote_rel_path="123/conv_123_1.wav",
        )
        # done_event.is_set() indicates completion
    """

    def __init__(self, remote: str, root_dir: str, max_retries: int = 3, backoff_seconds: float = 1.5):
        self.remote = remote.rstrip(":/") + ":"
        self.root_dir = root_dir.strip("/")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def _ensure_remote_dir(self, remote_dir: str) -> bool:
        try:
            # rclone mkdir remote:path
            cmd = [
                "rclone",
                "mkdir",
                f"{self.remote}{remote_dir}"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            LOGGER.error(f"rclone mkdir failed for {remote_dir}: {e}")
            return False

    def _copy_file(self, local_path: str, remote_path: str) -> bool:
        try:
            # rclone copyto local remote
            cmd = [
                "rclone",
                "copyto",
                local_path,
                f"{self.remote}{remote_path}",
                "--transfers", "1",
                "--checkers", "2",
                "--retries", str(self.max_retries),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            LOGGER.error(f"rclone copyto failed for {local_path} -> {remote_path}: {e}")
            return False

    def start_upload(self, local_path: str, conversation_id: str, dest_filename: str) -> Optional[threading.Event]:
        """Start upload in a thread. Returns an event that is set upon completion.

        remote structure: <root_dir>/<conversation_id>/<dest_filename>
        """
        if not os.path.exists(local_path):
            LOGGER.warning(f"Local path missing, skipping upload: {local_path}")
            return None

        done_event = threading.Event()

        def _worker():
            try:
                remote_dir = f"{self.root_dir}/{conversation_id}"
                if not self._ensure_remote_dir(remote_dir):
                    done_event.set()
                    return

                remote_path = f"{remote_dir}/{dest_filename}"

                attempt = 0
                while attempt < self.max_retries:
                    ok = self._copy_file(local_path, remote_path)
                    if ok:
                        LOGGER.info(f"Uploaded {local_path} to {self.remote}{remote_path}")
                        break
                    attempt += 1
                    sleep_for = self.backoff_seconds * (2 ** (attempt - 1))
                    time.sleep(sleep_for)
            finally:
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return done_event


