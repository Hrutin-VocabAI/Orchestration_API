import hashlib
import io
import os
import requests
from typing import Optional, Tuple
from pydub import AudioSegment
from logger import SYSTEM_LOGGER

AUDIO_PATH = os.getenv("AUDIO_LOCATION", ".")


class InvalidAudioError(Exception):
    """Custom exception for invalid audio files."""
    pass


# CHANGED: Expanded MIME types to support ASF/WMA and added application/octet-stream for auto-detection
# ORIGINAL:
# MIME_TO_EXT = {
#     "audio/wav": ".wav",
#     "audio/wave": ".wav",
#     "audio/x-wav": ".wav",
#     "audio/mpeg": ".mp3",
#     "audio/mp3": ".mp3",
#     "audio/ogg": ".ogg",
#     "audio/flac": ".flac",
#     "audio/x-flac": ".flac",
#     "audio/aac": ".aac",
#     "audio/x-aac": ".aac",
#     "audio/mp4": ".m4a",
#     "audio/x-m4a": ".m4a",
# } 
MIME_TO_EXT = {
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/vnd.wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/aac": ".aac",
    "audio/x-aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/x-ms-wma": ".wma",
    "video/x-ms-asf": ".asf",
    "application/octet-stream": None,
} 


class AudioHelper:
    def __init__(self, store_audio=False):
        self.logger = SYSTEM_LOGGER
        self.store_audio = store_audio

    def _generate_filename(self, url: str) -> str:
        """Generate a unique filename based on the URL hash."""
        return hashlib.md5(url.encode()).hexdigest()

    def _is_valid_audio_url(self, url: str) -> bool:
        """Check if the URL is a valid audio file by inspecting headers."""
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            content_type = response.headers.get('Content-Type', '').lower()
            return (
                content_type.startswith('audio/') or
                content_type.startswith('video/') or
                content_type == 'application/octet-stream' or
                any(mime in content_type for mime in MIME_TO_EXT)
            )
        except requests.RequestException as e:
            self.logger.error(f"Failed to validate URL: {e}")
            return False

    def resolve_url_and_filename(self, url: str, filename: Optional[str] = None) -> Tuple[str, str]:
        """
        Check if the URL points to an HTML page (UI player) or a direct audio file.
        If it's an HTML page, parse it to extract the real audio URL and filename.
        """
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        
        try:
            response = requests.head(url, headers=headers, allow_redirects=True, timeout=10)
            content_type = response.headers.get('Content-Type', '').lower()
        except requests.RequestException as e:
            self.logger.warning(f"HEAD request failed to validate URL {url}: {e}. Falling back to GET.")
            content_type = ""

        if "text/html" in content_type or not content_type:
            try:
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code == 200:
                    from bs4 import BeautifulSoup
                    from urllib.parse import urljoin
                    soup = BeautifulSoup(r.text, "html.parser")
                    source = soup.find("source")
                    if source:
                        media_path = source.get("src")
                        if media_path:
                            media_url = urljoin(url, media_path)
                            extracted_filename = os.path.basename(media_path)
                            self.logger.info(f"Resolved UI player URL {url} to audio URL {media_url} with filename {extracted_filename}")
                            if "?" in extracted_filename:
                                extracted_filename = extracted_filename.split("?")[0]
                            if not filename or filename == "sample_audio":
                                filename = extracted_filename
                            else:
                                ext = os.path.splitext(extracted_filename)[1]
                                if not os.path.splitext(filename)[1] and ext:
                                    filename = filename + ext
                            return media_url, filename
            except Exception as e:
                self.logger.error(f"Failed to resolve UI player URL: {e}")

        if not filename or filename == "sample_audio":
            try:
                path_filename = os.path.basename(url.split("?")[0])
                if path_filename and os.path.splitext(path_filename)[1]:
                    filename = path_filename
            except Exception:
                pass
        
        if not filename or filename == "sample_audio":
            filename = self._generate_filename(url)
        
        return url, filename

    def download_audio(self, url: str, filename=None) -> io.BytesIO:
        """Download the audio file and return it as a WAV buffer."""
        if not self._is_valid_audio_url(url):
            self.logger.error("Invalid audio URL")
            raise InvalidAudioError(f"Invalid audio content type")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            audio_buffer = io.BytesIO(response.content)
            content_type = response.headers.get('Content-Type')

            audio_buffer = self.convert_if_needed(audio_buffer, content_type)
            if self.store_audio:
                if filename is None:
                    filename = self._generate_filename(url)
                full_path = os.path.join(AUDIO_PATH, filename)
                with open(f"{full_path}.wav", 'wb') as f:
                    f.write(audio_buffer.getvalue())
                self.logger.info(f"Audio saved as {filename}.wav")
            return audio_buffer
        except requests.RequestException as e:
            self.logger.error(f"Failed to download audio: {e}")
            return None

    def _convert_to_wav(self, audio_buffer: io.BytesIO, ext: str) -> io.BytesIO:
        """Convert audio to WAV format."""
        try:
            audio_buffer.seek(0)
            # CHANGED: Added logic to try auto-detect if the provided extension hint fails (Crucial for files like your ASF labeled as MP3)
            # ORIGINAL:
            # audio = AudioSegment.from_file(audio_buffer, format=ext.strip('.'))
            try:
                format_hint = ext.strip('.') if ext else None
                audio = AudioSegment.from_file(audio_buffer, format=format_hint)
            except Exception as e:
                self.logger.warning(f"Failed to load with hint {ext}, trying auto-detect: {e}")
                audio_buffer.seek(0)
                audio = AudioSegment.from_file(audio_buffer)
                
            audio = audio.set_frame_rate(16000).set_channels(1)
            wav_buffer = io.BytesIO()
            # CHANGED: Explicitly setting pcm_s16le codec to ensure maximum compatibility with the torchaudio backend on the server
            # ORIGINAL:
            # audio.export(wav_buffer, format='wav')
            audio.export(wav_buffer, format='wav', parameters=["-acodec", "pcm_s16le"])
            wav_buffer.seek(0)
            return wav_buffer
        except Exception as e:
            self.logger.error(f"Failed to convert audio to WAV: {e}")
            return None

    def convert_if_needed(self, audio_buffer, content_type) -> io.BytesIO:
        if audio_buffer.getbuffer().nbytes == 0:
            self.logger.error("Downloaded file is empty")
            return None
        
        # CHANGED: Removed the 'assert' which would crash the server if an unknown MIME type was sent.
        # Now it defaults to None (Auto-detect) if the content_type is unknown.
        # ORIGINAL:
        # ext = MIME_TO_EXT.get(content_type)
        # assert ext is not None, f"content type: {content_type} not supported"
        ext = MIME_TO_EXT.get(content_type, None)
        
        if ext not in ['.wav', '.wave']:
            return self._convert_to_wav(audio_buffer, ext)

        # CHANGED: Even for WAV files, we now try to catch exceptions during loading and use the standardized pcm_s16le export.
        # ORIGINAL:
        # audio = AudioSegment.from_file(audio_buffer, format=ext.strip('.'))
        # if audio.frame_rate != 16000:
        #     audio = audio.set_frame_rate(16000).set_channels(1)
        #     wav_buffer = io.BytesIO()
        #     audio.export(wav_buffer, format='wav')
        #     wav_buffer.seek(0)
        #     return wav_buffer
        try:
            audio_buffer.seek(0)
            audio = AudioSegment.from_file(audio_buffer, format='wav')
            if audio.frame_rate != 16000 or audio.channels != 1:
                audio = audio.set_frame_rate(16000).set_channels(1)
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format='wav', parameters=["-acodec", "pcm_s16le"])
                wav_buffer.seek(0)
                return wav_buffer
            return audio_buffer
        except Exception as e:
            self.logger.warning(f"Failed to load WAV with hint, trying auto-detect: {e}")
            return self._convert_to_wav(audio_buffer, None)

    def check_channels_and_convert(self, audio_buffer: io.BytesIO, content_type: Optional[str]) -> Tuple[Optional[io.BytesIO], int]:
        """
        Convert/resample audio to WAV@16kHz while preserving channel count.

        This is intended for stereo/dual-channel workflows where downmixing to mono
        would lose channel separation. It does NOT change existing behavior of
        convert_if_needed(), which intentionally downmixes to 1 channel.
        """
        if audio_buffer.getbuffer().nbytes == 0:
            self.logger.error("Audio buffer is empty")
            return None, 0

        ext = MIME_TO_EXT.get(content_type, None)
        audio_buffer.seek(0)
        try:
            try:
                format_hint = ext.strip(".") if ext else None
                audio = AudioSegment.from_file(audio_buffer, format=format_hint)
            except Exception as e:
                self.logger.warning(f"Failed to load with hint {ext}, trying auto-detect: {e}")
                audio_buffer.seek(0)
                audio = AudioSegment.from_file(audio_buffer)
        except Exception as e:
            self.logger.error(f"Failed to decode audio: {e}")
            return None, 0

        channels = int(getattr(audio, "channels", 0) or 0)
        if channels <= 0:
            return None, 0

        try:
            audio = audio.set_frame_rate(16000)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav", parameters=["-acodec", "pcm_s16le"])
            wav_buffer.seek(0)
            return wav_buffer, channels
        except Exception as e:
            self.logger.error(f"Failed to convert audio to WAV (preserving channels): {e}")
            return None, 0
 



#import hashlib
# import io
# import logging
# import os

# import requests
# from pydub import AudioSegment

# AUDIO_PATH = os.getenv("AUDIO_LOCATION", ".")


# class InvalidAudioError(Exception):
#     """Custom exception for invalid audio files."""
#     pass


# MIME_TO_EXT = {
#     "audio/wav": ".wav",
#     "audio/wave": ".wav",
#     "audio/x-wav": ".wav",
#     "audio/mpeg": ".mp3",
#     "audio/mp3": ".mp3",
#     "audio/ogg": ".ogg",
#     "audio/flac": ".flac",
#     "audio/x-flac": ".flac",
#     "audio/aac": ".aac",
#     "audio/x-aac": ".aac",
#     "audio/mp4": ".m4a",
#     "audio/x-m4a": ".m4a",
# } 


# class AudioHelper:
#     def __init__(self, store_audio=False):
#         self.logger = logging.getLogger(__name__)
#         self.store_audio = store_audio
#         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     def _generate_filename(self, url: str) -> str:
#         """Generate a unique filename based on the URL hash."""
#         return hashlib.md5(url.encode()).hexdigest()

#     def _is_valid_audio_url(self, url: str) -> bool:
#         """Check if the URL is a valid audio file by inspecting headers."""
#         try:
#             response = requests.head(url, allow_redirects=True, timeout=5)
#             content_type = response.headers.get('Content-Type', '')
#             return content_type.startswith('audio/')
#         except requests.RequestException as e:
#             self.logger.error(f"Failed to validate URL: {e}")
#             return False

#     def download_audio(self, url: str, filename=None) -> io.BytesIO:
#         """Download the audio file and return it as a WAV buffer."""
#         if not self._is_valid_audio_url(url):
#             self.logger.error("Invalid audio URL")
#             raise InvalidAudioError(f"Invalid audio content type")

#         try:
#             response = requests.get(url, stream=True, timeout=30)
#             response.raise_for_status()
#             audio_buffer = io.BytesIO(response.content)
#             content_type = response.headers.get('Content-Type')

#             audio_buffer = self.convert_if_needed(audio_buffer, content_type)
#             if self.store_audio:
#                 if filename is None:
#                     filename = self._generate_filename(url)
#                 full_path = os.path.join(AUDIO_PATH, filename)
#                 with open(f"{full_path}.wav", 'wb') as f:
#                     f.write(audio_buffer.getvalue())
#                 self.logger.info(f"Audio saved as {filename}.wav")
#             return audio_buffer
#         except requests.RequestException as e:
#             self.logger.error(f"Failed to download audio: {e}")
#             return None

#     def _convert_to_wav(self, audio_buffer: io.BytesIO, ext: str) -> io.BytesIO:
#         """Convert audio to WAV format."""
#         try:
#             audio_buffer.seek(0)
#             audio = AudioSegment.from_file(audio_buffer, format=ext.strip('.'))
#             audio = audio.set_frame_rate(16000).set_channels(1)
#             wav_buffer = io.BytesIO()
#             audio.export(wav_buffer, format='wav')
#             wav_buffer.seek(0)
#             return wav_buffer
#         except Exception as e:
#             self.logger.error(f"Failed to convert audio to WAV: {e}")
#             return None

#     def convert_if_needed(self, audio_buffer, content_type) -> io.BytesIO:
#         if audio_buffer.getbuffer().nbytes == 0:
#             self.logger.error("Downloaded file is empty")
#             return None
#         ext = MIME_TO_EXT.get(content_type)
#         assert ext is not None, f"content type: {content_type} not supported"
#         if ext not in ['.wav', '.wave']:
#             return self._convert_to_wav(audio_buffer, ext)

#         audio = AudioSegment.from_file(audio_buffer, format=ext.strip('.'))
#         if audio.frame_rate != 16000:
#             audio = audio.set_frame_rate(16000).set_channels(1)
#             wav_buffer = io.BytesIO()
#             audio.export(wav_buffer, format='wav')
#             wav_buffer.seek(0)
#             return wav_buffer
#         return audio_buffer
