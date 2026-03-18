import logging
import os
from io import BytesIO

import requests
import torchaudio
import torchaudio.functional as F

from logger import timing_decorator, SYSTEM_LOGGER

TARGET_SAMPLE_RATE = 16000

# from flask_app.logger import get_logger


class AudioProcessor:
    LOGGER = SYSTEM_LOGGER

    @timing_decorator("AudioProcessor")
    def __init__(self, audio_data: BytesIO, filename, diarization_url=None,vad_url=None):
        self.DIARIZATION_URL = diarization_url
        self.VAD_URL = vad_url
        self.audio_buffer = audio_data
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in [".wav", ".mp3"]:
            self.LOGGER.error(f"Unsupported file format: {ext}. Only .wav and .mp3 are allowed")
            raise ValueError(f"Unsupported file format: {ext}. Only .wav and .mp3 are allowed.")
        
        self.audio_buffer.seek(0) # extra line addeed suriya
        waveform, sample_rate = torchaudio.load(self.audio_buffer)
        # waveform, sample_rate = torchaudio.load(self.audio_buffer, format="wav")
        self.waveform = waveform
        self.sample_rate = sample_rate
        if self.sample_rate != TARGET_SAMPLE_RATE:
            self.LOGGER.warning(f"Sample rate {sample_rate} Hz is not 16000 Hz. Resampling to {TARGET_SAMPLE_RATE} Hz.")
            self.waveform = F.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
            self.sample_rate = TARGET_SAMPLE_RATE
        self.LOGGER.info(f"{filename} Audio loaded")

    @timing_decorator("Diarization")
    def get_diarization_segments(self):
        files = {"audio": ("audio.wav", self.audio_buffer.getvalue(), "audio/wav")}
        response = requests.post(self.DIARIZATION_URL, files=files)
        if response.status_code != 200:
            logging.error(f"Diarization Error: {response.text}")
            return []
        return response.json().get("segments", [])

    def extract_speaker_chunk(self, start_time, end_time):
        start_frame = int(start_time * self.sample_rate)
        end_frame = int(end_time * self.sample_rate)
        return self.waveform[:, start_frame:end_frame]

    @timing_decorator("VAD")
    def get_VAD_segments(self):
        """
        Get VAD segments using HTTP POST to VAD service.
        Returns segments in the same format as diarization service.
        """
        try:
            self.LOGGER.info("Running VAD using HTTP POST to VAD service...")
            
            # Save audio buffer to temporary file for VAD service
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                self.audio_buffer.seek(0)
                temp_file.write(self.audio_buffer.getvalue())
                temp_file_path = temp_file.name
            
            try:
                # Call VAD service via HTTP POST
                with open(temp_file_path, 'rb') as f:
                    files = {'audio_file': (os.path.basename(temp_file_path), f, 'audio/wav')}
                    response = requests.post(self.VAD_URL, files=files, timeout=60)
                
                if response.status_code != 200:
                    self.LOGGER.error(f"VAD service error: {response.status_code} - {response.text}")
                    return []
                
                result = response.json()
                if not result.get("success"):
                    self.LOGGER.error(f"VAD processing failed: {result.get('error_message', 'Unknown error')}")
                    return []
                
                segments = result.get("segments", [])
                if not segments:
                    self.LOGGER.warning("No VAD segments found")
                    return []
                
            
                # Single-channel: force all segments to speaker_1 for consistency
                vad_segments = []
                for idx, segment in enumerate(segments, 1):
                    speaker_id = "speaker_1"
                    vad_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "segment_id": segment["segment_id"],
                        "speaker": speaker_id,
                        "transcription": ""
                    })
                
                self.LOGGER.info(f"VAD found {len(vad_segments)} segments")
                return vad_segments
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
        except Exception as e:
            self.LOGGER.error(f"VAD processing failed: {e}")
            return []

      