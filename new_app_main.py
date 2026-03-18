import os
import random
import time
# import time
import uuid
# from flask import request, jsonify

from distutils.util import strtobool
from io import BytesIO
from itertools import cycle


from flask import Flask, request, jsonify, g

from audio_downloader import AudioHelper
from request_validator import RequestValidator
from logger import SYSTEM_LOGGER, log_request
from transcription_service import TranscriptionService
from function_services import (
    TranscriptHandler,
    SentimentProcessor,
    CallAnalyzer,
    JSONUtils,
)

from file_service import FileService

LOGGER = SYSTEM_LOGGER
HOSTNAME = "http://localhost"  # Default hostname, can be overridden by env variable

try:
    STORE_AUDIO = bool(strtobool(os.getenv("STORE_AUDIO", "False")))
except ValueError:
    STORE_AUDIO = False

# GPU counts from env (default 4)
GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))

# Build backend URL lists
DIARIZATION_PORTS = [6000 + i for i in range(GPU_COUNT)]
ASR_PORTS = [7000 + i for i in range(GPU_COUNT)]
VAD_PORTS = [8000 + i for i in range(GPU_COUNT)]  # Assuming 1 VAD instance for now

DIARIZATION_URLS = [f"{HOSTNAME}:{port}/process" for port in DIARIZATION_PORTS]
ASR_URLS = [f"{HOSTNAME}:{port}/process" for port in ASR_PORTS]
VAD_URLS = [f"{HOSTNAME}:{port}/vad/process" for port in VAD_PORTS]

# Round-robin iterators
diar_iter = cycle(DIARIZATION_URLS)
asr_iter = cycle(ASR_URLS)
vad_iter = cycle(VAD_URLS)  # Using ASR URLs for VAD as placeholder

app = Flask(__name__)

RESULTS_DIR = "transcription_results"
os.makedirs(RESULTS_DIR, exist_ok=True)  # Create the directory if it doesn't exist


file_service = FileService()

def get_next_diarization_url():
    return next(diar_iter)


def get_next_asr_url():
    return next(asr_iter)

def get_next_vad_url():
    return next(vad_iter)





@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vocab-ai.com Transcription Service</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f0f0f0;
            }
            .container {
                text-align: center;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                color: #333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vocab-ai.com Transcription Service</h1>
            <p>Convert Voice calls to Chats.</p>
        </div>
    </body>
    </html>
    """


@app.route("/transcribe", methods=["POST"])
@log_request("upload")
def transcribe():
    conversation_id = request.form.get("conversation_id", str(uuid.uuid4())[:8])

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file or file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    audio_data = file.read()
    if not audio_data:
        return jsonify({"error": "Uploaded file is empty"}), 400

    # Save temp audio to get duration and for processing
    temp_audio_path = os.path.join(RESULTS_DIR, f"{conversation_id}_{file.filename}")
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_data)
    
    try:
        audio_duration = CallAnalyzer.get_audio_duration_from_file(temp_audio_path)
        buffer = AudioHelper().convert_if_needed(BytesIO(audio_data), file.content_type)
        
        if buffer is None:
            return jsonify({"error": "Unsupported or corrupted audio format"}), 400

        asr_url = get_next_asr_url()
        diar_url = get_next_diarization_url()

        transcription_service = TranscriptionService(asr_url, diar_url, None)
        transcription_result = transcription_service.process_audio(buffer, file.filename)

        response = JSONUtils.generate_rich_response(conversation_id, transcription_result, audio_duration, RESULTS_DIR)
        return jsonify(response), 200
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.route("/url_transcribe", methods=["POST"])
@log_request("url")
def transcribe_url():
    conversation_id = request.form.get("conversation_id", str(uuid.uuid4())[:8])
    if "audio_url" not in request.form:
        return jsonify({"error": "No audio_url provided"}), 400

    filename = request.form.get("file_name", "sample_audio")
    audio_url = request.form["audio_url"]

    if audio_url == "":
        return jsonify({"error": "Empty audio_url"}), 400

    audio_data = AudioHelper(STORE_AUDIO).download_audio(audio_url, filename=filename)
    if audio_data is None:
        return jsonify({"error": "Failed to download audio"}), 400

    # Save temp audio to get duration
    temp_audio_path = os.path.join(RESULTS_DIR, f"{conversation_id}_{filename}.wav")
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_data.getvalue())

    try:
        audio_duration = CallAnalyzer.get_audio_duration_from_file(temp_audio_path)
        
        # Pick backend dynamically
        asr_url = get_next_asr_url()
        diar_url = get_next_diarization_url()

        transcription_service = TranscriptionService(asr_url, diar_url, None)
        transcription_result = transcription_service.process_audio(audio_data, filename + ".wav")

        response = JSONUtils.generate_rich_response(conversation_id, transcription_result, audio_duration, RESULTS_DIR)
        return jsonify(response), 200
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.route("/transcriptions/formatted", methods=["POST"])
@log_request("dual_channel")
def transcribe_formatted():
    """
    RESTful API Endpoint: Transcribe one or two audio URLs and return merged/formatted transcription.
    """
    validator = RequestValidator(request.form)
    conversation_id = validator.get_required("conversation_id")
    audio_url_1 = validator.get_required("audio_url_1")
    audio_url_0 = validator.get_optional("audio_url_0")

    if not validator.is_valid():
        return validator.error_response()
    
    temp_files = []

    try:
        # -------- Process audio_url_1 --------
        audio_data1 = AudioHelper().download_audio(audio_url_1)
        if audio_data1 is None:
            return jsonify({"error": "Failed to download audio_url_1"}), 400

        # Save audio data to temporary file to get duration
        temp_audio_path = os.path.join(RESULTS_DIR, f"{conversation_id}_temp_audio1.wav")
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_data1.getvalue())
        temp_files.append(temp_audio_path)
        
        ASR_URL =  get_next_asr_url()
        VAD_URL =  get_next_vad_url()

        transcription_service = TranscriptionService(ASR_URL , None, VAD_URL)

        audio_duration = CallAnalyzer.get_audio_duration_from_file(temp_audio_path)
        transcription1 = transcription_service.process_audio_VAD(audio_data1, "audio1.wav")

        file1_path = os.path.join(RESULTS_DIR, f"{conversation_id}_audio1.json")
        temp_files.append(file1_path)
        file_service.save_transcription_json(transcription1, file1_path)
        final_json_path = file1_path

        # -------- Process audio_url_0 (if available) --------
        file2_path, transcription2 = None, []
        if audio_url_0:
            audio_data2 = AudioHelper().download_audio(audio_url_0)
            if audio_data2 is None:
                return jsonify({"error": "Failed to download audio_url_0"}), 400

            transcription2 = transcription_service.process_audio_VAD(audio_data2, "audio2.wav")
            file2_path = os.path.join(RESULTS_DIR, f"{conversation_id}_audio2.json")
            temp_files.append(file2_path)
            file_service.save_transcription_json(transcription2, file2_path)

        # -------- Merge transcripts --------
        if file2_path:
            merged_path = os.path.join(RESULTS_DIR, f"{conversation_id}_combined.json")
            transcriptions = file_service.merge_transcripts_json(file1_path, file2_path, merged_path)
            temp_files.append(merged_path)
        else:
            transcriptions = transcription1

        response = JSONUtils.generate_rich_response(conversation_id, transcriptions, audio_duration, RESULTS_DIR)
        return jsonify(response), 200

    except Exception as e:
        print("ERROR: Exception occurred ->", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Optional: Clean up temp files if needed
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try: 
                    os.remove(temp_file)
                    print("DEBUG: Removed temp file ->", temp_file)
                except Exception as cleanup_err:
                    print("WARNING: Failed to remove temp file ->", temp_file, cleanup_err)




if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
