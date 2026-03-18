import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Dict

import requests

from audio_processor import AudioProcessor
from speaker_identifier import SpeakerIdentifier
from logger import SYSTEM_LOGGER, timing_decorator

MIN_SAMPLE_SIZE = 800

LOGGER = SYSTEM_LOGGER


class TranscriptionService:
    def __init__(self, asr_url, diarization_url, vad_url):
        self.ASR_URL = asr_url
        self.DIARIZATION_URL = diarization_url
        self.VAD_URL = vad_url
        self.session = requests.Session()

    def transcribe_audio(self, waveform) -> str:
        payload = json.dumps({"waveform": waveform.numpy().tolist()})
        headers = {"Content-Type": "application/json"}
        try:
            response = self.session.post(self.ASR_URL, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get("text", "")
        except requests.exceptions.Timeout:
            LOGGER.info(f'Timeout - size {waveform.size()[-1]}')
            return ""
        except requests.exceptions.ConnectionError:
            LOGGER.info(f'ConnectionError - size {waveform.size()[-1]}')
            return ""
        except requests.exceptions.RequestException as e:
            LOGGER.info(f'RequestException - size {waveform.size()[-1]}')
            return ""

    def process_segment(self, segment, processor):
        """Process a single segment and return its transcription."""
        start_time, end_time, speaker_id = segment['start'], segment['end'], segment['speaker']
        speaker_audio_chunk = processor.extract_speaker_chunk(start_time, end_time).squeeze(0)
        if speaker_audio_chunk.size()[-1] <= MIN_SAMPLE_SIZE:
            return None

        transcription = self.transcribe_audio(speaker_audio_chunk)
        return {
            "start_time": start_time,
            "end_time": end_time,
            "speaker_id": speaker_id,
            # "act_id": "",
            "transcription": transcription,
        }

    def process_audio(self, audio_data: BytesIO, filename) -> List[Dict[str, str]]:
        processor = AudioProcessor(audio_data, filename, self.DIARIZATION_URL, None)
        segments = processor.get_diarization_segments()
        result, speaker_transcriptions = self._process_segments(segments, processor)
        speaker_0_text = ' '.join(speaker_transcriptions["SPEAKER_00"])
        speaker_1_text = ' '.join(speaker_transcriptions["SPEAKER_01"])
        speaker_0_score = SpeakerIdentifier.score_speaker_transcription(speaker_0_text)
        speaker_1_score = SpeakerIdentifier.score_speaker_transcription(speaker_1_text)
        agent = "SPEAKER_00" if speaker_0_score > speaker_1_score else "SPEAKER_01"
        for item in result:
            item["speaker_id"] = "agent" if item["speaker_id"] == agent else "customer"
        return result

    def process_audio_VAD(self, audio_data: BytesIO, filename) -> List[Dict[str, str]]:
        processor = AudioProcessor(audio_data, filename, None, self.VAD_URL)
        segments = processor.get_VAD_segments()
        result, speaker_transcriptions = self._process_segments(segments, processor)
        return result


    @timing_decorator("Transcription")
    def _process_segments(self, segments, processor):
        result = []
        speaker_transcriptions = {"SPEAKER_00": [], "SPEAKER_01": []}

        with ThreadPoolExecutor() as executor:
            segment_results = list(executor.map(lambda seg: self.process_segment(seg, processor), segments))

        for segment_result in segment_results:
            if segment_result:
                speaker_id = segment_result["speaker_id"]
                transcription = segment_result["transcription"]

                if speaker_id not in speaker_transcriptions:
                    speaker_transcriptions[speaker_id] = []
                speaker_transcriptions[speaker_id].append(transcription)

                result.append(segment_result)

        return result, speaker_transcriptions
