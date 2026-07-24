import json
import os
from pathlib import Path

from audio_downloader import AudioHelper
from audio_processor import DiarizationUnavailableError
from transcription_service import TranscriptionService
from function_services import JSONUtils, CallAnalyzer
from logger import SYSTEM_LOGGER

CHECKPOINTS_DIR = Path(__file__).parent / "health_checkpoints"

# Fields that legitimately differ between two otherwise-identical runs and
# must be excluded from the comparison - not a signal of anything being
# wrong with the pipeline. "id" is utterances[].id, a fresh uuid4() per
# utterance from SentimentProcessor.combine_data() (function_services.py),
# unrelated to content.
IGNORED_FIELDS = {"request_id", "id"}


class HealthCheckService:
    """
    Runs a fixed reference audio through the real transcription pipeline
    (the same steps /url_transcribe uses) and compares the result against a
    saved baseline JSON - a functional end-to-end canary check, not just a
    liveness ping. One instance per client, so adding a new client is just
    instantiating another HealthCheckService with its own name/audio_url.
    """

    LOGGER = SYSTEM_LOGGER

    def __init__(self, client_name, audio_url, get_asr_url, get_diarization_url, results_dir):
        self.client_name = client_name
        self.audio_url = audio_url
        self.get_asr_url = get_asr_url
        self.get_diarization_url = get_diarization_url
        self.results_dir = results_dir
        self.checkpoint_dir = CHECKPOINTS_DIR / client_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_path = self.checkpoint_dir / "baseline.json"

    def _run_pipeline(self):
        """Executes the same steps as /url_transcribe against self.audio_url."""
        helper = AudioHelper(False)
        resolved_url, resolved_filename = helper.resolve_url_and_filename(self.audio_url, None)
        audio_data = helper.download_audio(resolved_url, filename=resolved_filename)
        if audio_data is None:
            raise RuntimeError("Failed to download reference audio")

        temp_audio_path = os.path.join(
            self.results_dir, f"healthcheck_{self.client_name}_{resolved_filename}.wav"
        )
        with open(temp_audio_path, "wb") as fout:
            fout.write(audio_data.getvalue())

        try:
            audio_duration = CallAnalyzer.get_audio_duration_from_file(temp_audio_path)
            asr_url = self.get_asr_url()
            diar_url = self.get_diarization_url()
            transcription_service = TranscriptionService(asr_url, diar_url, None)
            transcription_result = transcription_service.process_audio(
                audio_data, resolved_filename + ".wav"
            )
            response = JSONUtils.generate_rich_response(
                f"healthcheck_{self.client_name}", transcription_result, audio_duration, self.results_dir
            )
            response["URL"] = self.audio_url
            return response
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def save_baseline(self):
        """Runs the pipeline now and overwrites the stored baseline with this result."""
        response = self._run_pipeline()
        self.baseline_path.write_text(json.dumps(response, indent=2))
        self.LOGGER.info(f"[HealthCheck:{self.client_name}] Baseline saved to {self.baseline_path}")
        return response

    def check(self):
        """
        Returns {"status": "PASS"|"FAIL", "client": ..., "triggers": [...], "response": {...}}.
        FAIL reasons, in the order they're checked: API_ERROR, EMPTY_RESPONSE, NO_BASELINE, MISMATCH.
        """
        try:
            response = self._run_pipeline()
        except DiarizationUnavailableError as e:
            return self._fail([f"TRIGGER: API_ERROR - {e}"])
        except Exception as e:
            return self._fail([f"TRIGGER: API_ERROR - {e}"])

        transcriptions = response.get("transcriptions", [])
        if not transcriptions:
            return self._fail(["TRIGGER: EMPTY_RESPONSE - transcriptions list is empty"], response)

        if not self.baseline_path.exists():
            return self._fail(
                ["TRIGGER: NO_BASELINE - no saved baseline yet; call save_baseline() first"], response
            )

        baseline = json.loads(self.baseline_path.read_text())
        diffs = self._diff(baseline, response, path="")
        if diffs:
            return self._fail([f"TRIGGER: MISMATCH - {d}" for d in diffs], response)

        return {
            "status": "PASS",
            "client": self.client_name,
            "triggers": ["OK - response matches baseline exactly"],
        }

    def _fail(self, triggers, response=None):
        result = {"status": "FAIL", "client": self.client_name, "triggers": triggers}
        if response is not None:
            result["response"] = response
        return result

    def _diff(self, expected, actual, path):
        """Exact-match structural diff (no fuzzy/tolerance matching), skipping IGNORED_FIELDS."""
        diffs = []
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in sorted(set(expected) | set(actual)):
                if key in IGNORED_FIELDS:
                    continue
                p = f"{path}.{key}" if path else key
                if key not in actual:
                    diffs.append(f"{p}: missing from response")
                elif key not in expected:
                    diffs.append(f"{p}: unexpected field in response")
                else:
                    diffs.extend(self._diff(expected[key], actual[key], p))
        elif isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                diffs.append(f"{path}: expected {len(expected)} items, got {len(actual)}")
            else:
                for i, (e, a) in enumerate(zip(expected, actual)):
                    diffs.extend(self._diff(e, a, f"{path}[{i}]"))
        else:
            if expected != actual:
                diffs.append(f"{path}: expected {expected!r} got {actual!r}")
        return diffs
