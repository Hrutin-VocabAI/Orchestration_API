import os
import json
import uuid
import hashlib
import subprocess
import requests
import audioread
from pydub import AudioSegment


# ---------------- Audio Conversion Utilities ----------------

class AudioConverter:
    """Utility class for converting audio files to WAV format."""

    @staticmethod
    def run_ffmpeg_conversion(input_path, output_path):
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", input_path, output_path]
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0

    @staticmethod
    def run_sox_conversion(input_path, output_path):
        sox_cmd = ["sox", input_path, output_path]
        result = subprocess.run(sox_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0


# ---------------- Audio Utilities ----------------

class AudioUtils:
    """Utility class for audio file processing."""

    @staticmethod
    def get_audio_duration(file_path):
        """Calculate duration of the audio file in seconds."""
        audio = AudioSegment.from_file(file_path)
        return int(len(audio) / 1000)

    @staticmethod
    def download_and_convert_to_wav(audio_url, output_directory):
        """Download audio file from URL and convert to WAV."""
        try:
            os.makedirs(output_directory, exist_ok=True)

            url_hash = hashlib.md5(audio_url.encode('utf-8')).hexdigest()
            ext = os.path.splitext(audio_url.split("?")[0])[-1] or ".unknown"
            file_name = f"audio_{url_hash}{ext}"
            original_file_path = os.path.join(output_directory, file_name)

            response = requests.get(audio_url, stream=True)
            if response.status_code != 200:
                return None
            with open(original_file_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)

            if not os.path.exists(original_file_path) or os.path.getsize(original_file_path) == 0:
                return None

            if original_file_path.lower().endswith('.wav'):
                return original_file_path

            wav_file_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}.wav")

            try:
                with audioread.audio_open(original_file_path) as f:
                    detected_format = f.format
            except Exception:
                detected_format = None

            try:
                audio = AudioSegment.from_file(original_file_path, format=detected_format)
                audio.export(wav_file_path, format="wav")
            except Exception:
                if not AudioConverter.run_ffmpeg_conversion(original_file_path, wav_file_path):
                    if not AudioConverter.run_sox_conversion(original_file_path, wav_file_path):
                        raise RuntimeError("All conversion methods failed.")

            return wav_file_path

        except Exception as e:
            raise RuntimeError(f"Error in download/convert: {e}")


# ---------------- JSON Utilities ----------------

class JSONUtils:
    """Utility class for JSON file handling."""

    @staticmethod
    def read_json(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []  # return empty list if fail

    @staticmethod
    def generate_rich_response(conversation_id, transcriptions, audio_duration, results_dir):
        """Generates the rich response format requested by the user."""
        from file_service import FileService
        from flask import g
        
        # Save to temp file for TranscriptHandler
        temp_json_path = os.path.join(results_dir, f"{conversation_id}_temp.json")
        FileService.save_transcription_json(transcriptions, temp_json_path)
        
        try:
            agent_transcript = TranscriptHandler.extract_agent_transcripts(temp_json_path)
            customer_transcript = TranscriptHandler.extract_customer_transcripts(temp_json_path)
            transcript_conf_score = TranscriptHandler.get_transcripts_with_confidence(temp_json_path)
            utterances, alternatives = SentimentProcessor.combine_data(transcript_conf_score)
            call_analysis = CallAnalyzer.analyze_call_timing(transcriptions, audio_duration)

            return {
                "request_id": getattr(g, "request_id", None),
                "conversation_id": conversation_id,
                "transcriptions": transcriptions,
                "agent_transcript": agent_transcript,
                "customer_transcript": customer_transcript,
                "alternatives": alternatives,
                "utterances": utterances,
                "call_duration": call_analysis["call_duration"],
                "silence/hold_duration": call_analysis["silence_hold_duration"],
                "silence/hold_percentage": call_analysis["silence_hold_percentage"],
                "silence_before_start": call_analysis["silence_before_start"],
                "silence_after_end": call_analysis["silence_after_end"],
                "cross_talk_duration": call_analysis["cross_talk_duration"],
                "total_speaking_time": call_analysis["total_speaking_time"],
                "speaking_percentage": call_analysis["speaking_percentage"],
            }
        finally:
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)


# ---------------- Transcript Utilities ----------------

class TranscriptProcessor:
    """Handles transcript processing and analysis."""

    STOP_WORDS = {"i", "me", "my", "myself", "we", "our", "ours",
                  "ourselves", "you", "you're", "you've", "you'll", "you'd"}

    @staticmethod
    def extract_transcripts(json_data, speaker_type):
        """Extract transcripts for the specified speaker type."""
        if not isinstance(json_data, list):
            return ""
        return " ".join(
            row.get("transcription", "").strip()
            for row in json_data
            if row.get("speaker_id", "").lower() == speaker_type.lower()
        )

    @staticmethod
    def calculate_word_frequencies(text):
        """Calculates top-5 word frequencies, excluding stop words."""
        freq = {}
        for word in text.lower().split():
            if word not in TranscriptProcessor.STOP_WORDS:
                freq[word] = freq.get(word, 0) + 1
        return sorted(
            [{"word": w, "frequency": f} for w, f in freq.items()],
            key=lambda x: x["frequency"],
            reverse=True
        )[:5]

    @staticmethod
    def check_keywords(transcript, tags):
        return {tag: "yes" if tag in transcript else "no" for tag in tags}


# ---------------- Sentiment Processing ----------------

class SentimentProcessor:
    """Handles sentiment + confidence data."""

    @staticmethod
    def process_transcript(transcript, confidence_score, speaker_id,
                           start_time, end_time, sentiment="neutral", sentiment_score=0.0):
        """Split transcript into words and assign timings + sentiment."""
        words = transcript.split()
        word_duration = (end_time - start_time) / max(len(words), 1)
        return [
            {
                "word": word,
                "start": start_time + i * word_duration,
                "end": start_time + (i + 1) * word_duration,
                "confidence": confidence_score,
                "speaker": speaker_id,
                "speaker_confidence": confidence_score,
                "punctuated_word": word.capitalize() if word.isalpha() else word,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
            }
            for i, word in enumerate(words)
        ]

    @staticmethod
    def combine_data(confidence_data, sentiment_data=None):
        """Combine confidence and sentiment per word for full transcript."""
        combined_data, all_words = [], []
        for conf in confidence_data:
            start, end = float(conf.get("start_time", 0.0)), float(conf.get("end_time", 0.0))
            sentiment, score = "neutral", 0.0
            if sentiment_data:
                matched = [
                    s for s in sentiment_data
                    if start >= float(s.get("start_time", 0.0)) and end <= float(s.get("end_time", 0.0))
                ]
                if matched:
                    sentiment = matched[0].get("emotion_label", "neutral")
                    score = matched[0].get("sentiment_confidence_score", 0.0)

            word_data = SentimentProcessor.process_transcript(
                conf.get("transcription", ""),
                conf.get("transcription_confidence_score", 0.0),
                conf.get("speaker_id", ""),
                start, end,
                sentiment, score
            )
            all_words.extend(word_data)
            combined_data.append({
                "id": str(uuid.uuid4()),
                "start": start,
                "end": end,
                "speaker": conf.get("speaker_id", ""),
                "confidence": conf.get("transcription_confidence_score", 0.0),
                "sentiment": sentiment,
                "sentiment_score": score,
                "transcript": conf.get("transcription", ""),
                "words": word_data,
            })
        return combined_data, all_words


# ---------------- Transcript Handler ----------------

class TranscriptHandler:
    """Handles transcripts from JSON files."""

    @staticmethod
    def get_transcripts_with_confidence(json_file_path):
        data = JSONUtils.read_json(json_file_path)
        return [
            {
                "transcription": t.get("transcription", "").strip(),
                "transcription_confidence_score": 0.0,  # default
                "start_time": t.get("start_time", 0.0),
                "end_time": t.get("end_time", 0.0),
                "speaker_id": t.get("speaker_id", ""),
            }
            for t in data
        ]

    @staticmethod
    def get_all_transcriptions(json_file_path):
        data = JSONUtils.read_json(json_file_path)
        return " ".join(t.get("transcription", "").strip() for t in data).strip()

    @staticmethod
    def extract_agent_transcripts(json_file_path):
        data = JSONUtils.read_json(json_file_path)
        # Try "speaker_1" first (VAD pipeline), fall back to "agent" (diarization)
        result = TranscriptProcessor.extract_transcripts(data, "speaker_1")
        if not result.strip():
            result = TranscriptProcessor.extract_transcripts(data, "agent")
        return result

    @staticmethod
    def extract_customer_transcripts(json_file_path):
        data = JSONUtils.read_json(json_file_path)
        # Try "speaker_0" first (VAD pipeline), fall back to "customer" (diarization)
        result = TranscriptProcessor.extract_transcripts(data, "speaker_0")
        if not result.strip():
            result = TranscriptProcessor.extract_transcripts(data, "customer")
        return result


# ---------------- Call Analysis ----------------

class CallAnalyzer:
    """Analyzes call timing metrics from merged transcript data."""
    
    @staticmethod
    def analyze_call_timing(json_data, audio_duration=None):
        """
        Analyze call timing metrics from merged transcript JSON data.
        
        Args:
            json_data: List of transcript segments with start_time, end_time, speaker_id
            audio_duration: Total audio duration in seconds (optional)
            
        Returns:
            dict: Dictionary containing all timing metrics
        """
        if not json_data or not isinstance(json_data, list):
            return {
                "call_duration": 0,
                "silence_hold_duration": 0,
                "silence_hold_percentage": 0,
                "silence_before_start": 0,
                "silence_after_end": 0,
                "cross_talk_duration": 0,
                "total_speaking_time": 0,
                "speaking_percentage": 0
            }
        
        # Sort segments by start_time to ensure proper order
        sorted_segments = sorted(json_data, key=lambda x: x.get('start_time', 0))
        
        # Prefer segments that have ASR text; if none, fall back to timing-only VAD segments
        segments_with_text = [seg for seg in sorted_segments if seg.get('transcription', '').strip()]
        valid_segments = segments_with_text if segments_with_text else sorted_segments
        
        if not valid_segments:
            return {
                "call_duration": audio_duration or 0,
                "silence_hold_duration": audio_duration or 0,
                "silence_hold_percentage": 100.0,
                "silence_before_start": audio_duration or 0,
                "silence_after_end": 0,
                "cross_talk_duration": 0,
                "total_speaking_time": 0,
                "speaking_percentage": 0
            }
        
        # Calculate basic timing
        first_segment_start = valid_segments[0]['start_time']
        last_segment_end = valid_segments[-1]['end_time']
        
        # Use audio_duration if provided, otherwise use last segment end time
        call_duration = audio_duration or last_segment_end
        
        # Calculate silence before first speech
        silence_before_start = first_segment_start
        
        # Calculate silence after last speech
        silence_after_end = max(0, call_duration - last_segment_end)
        
        # Calculate total speaking time by merging overlapping segments
        total_speaking_time = CallAnalyzer._calculate_total_speaking_time(valid_segments)
        
        silence_hold_duration = call_duration - total_speaking_time
        silence_hold_percentage = (silence_hold_duration / call_duration * 100) if call_duration > 0 else 0
        speaking_percentage = (total_speaking_time / call_duration * 100) if call_duration > 0 else 0
        
        # Calculate cross-talk duration (overlapping speech)
        cross_talk_duration = CallAnalyzer._calculate_crosstalk_duration(valid_segments)
        
        return {
            "call_duration": round(call_duration, 2),
            "silence_hold_duration": round(silence_hold_duration, 2),
            "silence_hold_percentage": round(silence_hold_percentage, 2),
            "silence_before_start": round(silence_before_start, 2),
            "silence_after_end": round(silence_after_end, 2),
            "cross_talk_duration": round(cross_talk_duration, 2),
            "total_speaking_time": round(total_speaking_time, 2),
            "speaking_percentage": round(speaking_percentage, 2)
        }
    
    @staticmethod
    def _calculate_crosstalk_duration(segments):
        """
        Calculate total cross-talk duration where speakers overlap.
        
        Args:
            segments: List of sorted transcript segments
            
        Returns:
            float: Total cross-talk duration in seconds
        """
        if len(segments) < 2:
            return 0.0
        
        crosstalk_duration = 0.0
        
        for i in range(len(segments)):
            current_seg = segments[i]
            current_start = current_seg['start_time']
            current_end = current_seg['end_time']
            current_speaker = current_seg.get('speaker_id', '')
            
            # Check for overlaps with subsequent segments
            for j in range(i + 1, len(segments)):
                next_seg = segments[j]
                next_start = next_seg['start_time']
                next_end = next_seg['end_time']
                next_speaker = next_seg.get('speaker_id', '')
                
                # Skip if same speaker
                if current_speaker == next_speaker:
                    continue
                
                # Check for overlap
                if next_start < current_end:
                    # Calculate overlap duration
                    overlap_start = max(current_start, next_start)
                    overlap_end = min(current_end, next_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    crosstalk_duration += overlap_duration
                else:
                    # No more overlaps possible since segments are sorted
                    break
        
        return crosstalk_duration
    
    @staticmethod
    def _calculate_total_speaking_time(segments):
        """
        Calculate total speaking time by merging overlapping segments.
        
        Args:
            segments: List of sorted transcript segments
            
        Returns:
            float: Total speaking time in seconds (overlaps counted only once)
        """
        if not segments:
            return 0.0
        
        # Sort segments by start_time
        sorted_segments = sorted(segments, key=lambda x: x.get('start_time', 0))
        
        # Merge overlapping segments
        merged_segments = []
        current_start = sorted_segments[0]['start_time']
        current_end = sorted_segments[0]['end_time']
        
        for i in range(1, len(sorted_segments)):
            next_start = sorted_segments[i]['start_time']
            next_end = sorted_segments[i]['end_time']
            
            # If segments overlap or are adjacent, extend current segment
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                # No overlap, save current segment and start new one
                merged_segments.append((current_start, current_end))
                current_start = next_start
                current_end = next_end
        
        # Add the last segment
        merged_segments.append((current_start, current_end))
        
        # Calculate total speaking time
        total_time = sum(end - start for start, end in merged_segments)
        return total_time
    
    @staticmethod
    def get_audio_duration_from_file(file_path):
        """
        Get audio duration from file using AudioUtils.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            float: Audio duration in seconds, or None if error
        """
        try:
            return AudioUtils.get_audio_duration(file_path)
        except Exception:
            return None
    
    @staticmethod
    def analyze_from_json_file(json_file_path, audio_file_path=None):
        """
        Analyze call timing from JSON file and optional audio file.
        
        Args:
            json_file_path: Path to merged transcript JSON file
            audio_file_path: Optional path to audio file for duration
            
        Returns:
            dict: Complete timing analysis results
        """
        # Load JSON data
        json_data = JSONUtils.read_json(json_file_path)
        
        # Get audio duration if file provided
        audio_duration = None
        if audio_file_path and os.path.exists(audio_file_path):
            audio_duration = CallAnalyzer.get_audio_duration_from_file(audio_file_path)
        
        # Perform analysis
        return CallAnalyzer.analyze_call_timing(json_data, audio_duration)


# ---------------- File Handling ----------------

class FileHandler:
    """Handles file validation, saving, and format conversion."""

    @staticmethod
    def validate_file_size(uploaded_file, max_size):
        uploaded_file.seek(0, os.SEEK_END)
        size = uploaded_file.tell()
        uploaded_file.seek(0)
        return size <= max_size

    @staticmethod
    def save_uploaded_file(uploaded_file):
        file_path = os.path.join("uploads", uploaded_file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        uploaded_file.save(file_path)
        return file_path

    @staticmethod
    def validate_audio_format(file_path):
        return os.path.splitext(file_path)[1].lower() in {".mp3", ".wav"}

    @staticmethod
    def convert_to_wav_if_needed(file_path):
        if file_path.lower().endswith(".mp3"):
            sound = AudioSegment.from_mp3(file_path)
            new_path = file_path.replace(".mp3", ".wav")
            sound.export(new_path, format="wav")
            return new_path
        return file_path
