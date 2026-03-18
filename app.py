import os
import random
import time
import uuid

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

# ----------------------------------------------------------------
# Configuration (no dotenv - relies on os.getenv / hardcoded defaults)
# ----------------------------------------------------------------
HOSTNAME = os.getenv("SERVER_HOSTNAME", "http://27.111.72.61")

try:
    STORE_AUDIO = bool(strtobool(os.getenv("STORE_AUDIO", "False")))
except ValueError:
    STORE_AUDIO = False

# GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))
GPU_COUNT = 1

# DIARIZATION_PORTS = [6000 + i for i in range(GPU_COUNT)]
# ASR_PORTS         = [7000 + i for i in range(GPU_COUNT)]
# VAD_PORTS         = [8000 + i for i in range(GPU_COUNT)]

DIARIZATION_PORTS = [6001]
ASR_PORTS         = [7001]
VAD_PORTS         = [8001]

DIARIZATION_URLS = [f"{HOSTNAME}:{port}/process"     for port in DIARIZATION_PORTS]
ASR_URLS         = [f"{HOSTNAME}:{port}/process"     for port in ASR_PORTS]
VAD_URLS         = [f"{HOSTNAME}:{port}/vad/process" for port in VAD_PORTS]

# Round-robin iterators
diar_iter = cycle(DIARIZATION_URLS)
asr_iter  = cycle(ASR_URLS)
vad_iter  = cycle(VAD_URLS)

app = Flask(__name__)

RESULTS_DIR = "transcription_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

file_service = FileService()


def get_next_diarization_url():
    return next(diar_iter)

def get_next_asr_url():
    return next(asr_iter)

def get_next_vad_url():
    return next(vad_iter)


# ----------------------------------------------------------------
# Index
# ----------------------------------------------------------------
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
            body { font-family: Arial, sans-serif; margin: 0; padding: 0;
                   display: flex; justify-content: center; align-items: center;
                   height: 100vh; background-color: #f0f0f0; }
            .container { text-align: center; background: white; padding: 20px;
                         border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
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


# ----------------------------------------------------------------
# /transcribe  (file upload)
# ----------------------------------------------------------------
@app.route("/transcribe", methods=["POST"])
@log_request("upload")
def transcribe():
    t0 = time.time()
    rid = getattr(g, "request_id", str(uuid.uuid4())[:8])

    LOGGER.info(f"[{rid}] ===== /transcribe START =====")
    LOGGER.info(f"[{rid}] [REQ] content_length={request.content_length} mimetype={request.mimetype}")

    conversation_id = request.form.get("conversation_id", str(uuid.uuid4())[:8])

    # ------------------------------------------------------------------
    # STEP A: Parse request / access uploaded file
    # ------------------------------------------------------------------
    tA0 = time.time()
    LOGGER.info(f"[{rid}] [STEP A] Accessing request.files ...")

    if "file" not in request.files:
        LOGGER.error(f"[{rid}] No file provided")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    tA1 = time.time()
    LOGGER.info(f"[{rid}] [STEP A DONE] request.files parsed in {tA1 - tA0:.2f}s")

    if not file or file.filename == "":
        LOGGER.error(f"[{rid}] Empty file")
        return jsonify({"error": "Empty file"}), 400

    LOGGER.info(f"[{rid}] [FILE] filename={file.filename} content_type={file.content_type}")

    # ------------------------------------------------------------------
    # STEP B: Read uploaded bytes
    # ------------------------------------------------------------------
    tB0 = time.time()
    LOGGER.info(f"[{rid}] [STEP B] Starting file.read() ...")

    audio_data = file.read()

    tB1 = time.time()
    LOGGER.info(f"[{rid}] [STEP B DONE] file.read() took {tB1 - tB0:.2f}s  size={len(audio_data)/1024/1024:.2f}MB")

    if not audio_data:
        LOGGER.error(f"[{rid}] Uploaded file has 0 bytes after read()")
        return jsonify({"error": "Uploaded file is empty"}), 400

    # Save temp audio to get duration
    temp_audio_path = os.path.join(RESULTS_DIR, f"{conversation_id}_{file.filename}")
    with open(temp_audio_path, "wb") as fout:
        fout.write(audio_data)

    try:
        # ------------------------------------------------------------------
        # STEP C: Get audio duration
        # ------------------------------------------------------------------
        tC0 = time.time()
        LOGGER.info(f"[{rid}] [STEP C] Getting audio duration ...")

        audio_duration = CallAnalyzer.get_audio_duration_from_file(temp_audio_path)

        tC1 = time.time()
        LOGGER.info(f"[{rid}] [STEP C DONE] audio_duration={audio_duration:.2f}s  took {tC1 - tC0:.2f}s")

        # ------------------------------------------------------------------
        # STEP D: Convert audio (if needed)
        # ------------------------------------------------------------------
        tD0 = time.time()
        LOGGER.info(f"[{rid}] [STEP D] Converting audio if needed ...")

        buffer = AudioHelper().convert_if_needed(BytesIO(audio_data), file.content_type)

        tD1 = time.time()
        LOGGER.info(f"[{rid}] [STEP D DONE] convert_if_needed took {tD1 - tD0:.2f}s")

        if buffer is None:
            LOGGER.error(f"[{rid}] Audio processing failed for file: {file.filename}")
            return jsonify({"error": "Unsupported or corrupted audio format"}), 400

        # ------------------------------------------------------------------
        # STEP E: Pick backend URLs
        # ------------------------------------------------------------------
        asr_url  = get_next_asr_url()
        diar_url = get_next_diarization_url()
        LOGGER.info(f"[{rid}] [STEP E] Selected ASR URL: {asr_url}")
        LOGGER.info(f"[{rid}] [STEP E] Selected DIAR URL: {diar_url}")

        # ------------------------------------------------------------------
        # STEP F: Call Transcription Service
        # ------------------------------------------------------------------
        tF0 = time.time()
        LOGGER.info(f"[{rid}] [STEP F] Calling TranscriptionService.process_audio() ...")

        transcription_service  = TranscriptionService(asr_url, diar_url, None)
        safe_filename = os.path.splitext(file.filename)[0] + ".wav"
        transcription_result   = transcription_service.process_audio(buffer, safe_filename)

        tF1 = time.time()
        LOGGER.info(f"[{rid}] [STEP F DONE] process_audio took {tF1 - tF0:.2f}s  entries={len(transcription_result) if transcription_result else 0}")

        # ------------------------------------------------------------------
        # STEP G: Build rich response
        # ------------------------------------------------------------------
        tG0 = time.time()
        LOGGER.info(f"[{rid}] [STEP G] Building rich response ...")

        response = JSONUtils.generate_rich_response(conversation_id, transcription_result, audio_duration, RESULTS_DIR)

        tG1 = time.time()
        total = time.time() - t0
        LOGGER.info(f"[{rid}] [STEP G DONE] rich response built in {tG1 - tG0:.2f}s")
        LOGGER.info(f"[{rid}] ===== /transcribe COMPLETE in {total:.2f}s =====")

        return jsonify(response), 200

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# ----------------------------------------------------------------
# /url_transcribe  (audio URL)
# ----------------------------------------------------------------
@app.route("/url_transcribe", methods=["POST"])
@log_request("url")
def transcribe_url():
    t0  = time.time()
    rid = getattr(g, "request_id", str(uuid.uuid4())[:8])

    LOGGER.info(f"[{rid}] ===== /url_transcribe START =====")

    conversation_id = request.form.get("conversation_id", str(uuid.uuid4())[:8])

    if "audio_url" not in request.form:
        LOGGER.error(f"[{rid}] No audio_url provided")
        return jsonify({"error": "No audio_url provided"}), 400

    filename  = request.form.get("file_name", "sample_audio")
    audio_url = request.form["audio_url"]

    if audio_url == "":
        LOGGER.error(f"[{rid}] Empty audio_url")
        return jsonify({"error": "Empty audio_url"}), 400

    LOGGER.info(f"[{rid}] [INFO] conversation_id={conversation_id}  filename={filename}")

    # ------------------------------------------------------------------
    # STEP A: Download audio
    # ------------------------------------------------------------------
    tA0 = time.time()
    LOGGER.info(f"[{rid}] [STEP A] Downloading audio from URL ...")

    audio_data = AudioHelper(STORE_AUDIO).download_audio(audio_url, filename=filename)

    tA1 = time.time()
    if audio_data is None:
        LOGGER.error(f"[{rid}] [STEP A FAIL] Failed to download audio after {tA1 - tA0:.2f}s")
        return jsonify({"error": "Failed to download audio"}), 400
    LOGGER.info(f"[{rid}] [STEP A DONE] Download took {tA1 - tA0:.2f}s")

    # Save temp file for duration
    temp_audio_path = os.path.join(RESULTS_DIR, f"{conversation_id}_{filename}.wav")
    with open(temp_audio_path, "wb") as fout:
        fout.write(audio_data.getvalue())

    try:
        # ------------------------------------------------------------------
        # STEP B: Get audio duration
        # ------------------------------------------------------------------
        tB0 = time.time()
        LOGGER.info(f"[{rid}] [STEP B] Getting audio duration ...")

        audio_duration = CallAnalyzer.get_audio_duration_from_file(temp_audio_path)

        tB1 = time.time()
        LOGGER.info(f"[{rid}] [STEP B DONE] audio_duration={audio_duration:.2f}s  took {tB1 - tB0:.2f}s")

        # ------------------------------------------------------------------
        # STEP C: Pick backend URLs
        # ------------------------------------------------------------------
        asr_url  = get_next_asr_url()
        diar_url = get_next_diarization_url()
        LOGGER.info(f"[{rid}] [STEP C] Selected ASR URL: {asr_url}")
        LOGGER.info(f"[{rid}] [STEP C] Selected DIAR URL: {diar_url}")

        # ------------------------------------------------------------------
        # STEP D: Call Transcription Service
        # ------------------------------------------------------------------
        tD0 = time.time()
        LOGGER.info(f"[{rid}] [STEP D] Calling TranscriptionService.process_audio() ...")

        transcription_service = TranscriptionService(asr_url, diar_url, None)
        transcription_result  = transcription_service.process_audio(audio_data, filename + ".wav")

        tD1 = time.time()
        LOGGER.info(f"[{rid}] [STEP D DONE] process_audio took {tD1 - tD0:.2f}s  entries={len(transcription_result) if transcription_result else 0}")

        # ------------------------------------------------------------------
        # STEP E: Build rich response
        # ------------------------------------------------------------------
        tE0 = time.time()
        LOGGER.info(f"[{rid}] [STEP E] Building rich response ...")

        response = JSONUtils.generate_rich_response(conversation_id, transcription_result, audio_duration, RESULTS_DIR)

        tE1 = time.time()
        total = time.time() - t0
        LOGGER.info(f"[{rid}] [STEP E DONE] rich response built in {tE1 - tE0:.2f}s")
        LOGGER.info(f"[{rid}] ===== /url_transcribe COMPLETE in {total:.2f}s =====")

        return jsonify(response), 200

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# ----------------------------------------------------------------
# /transcriptions/formatted  (dual-channel VAD)
# ----------------------------------------------------------------
@app.route("/transcriptions/formatted", methods=["POST"])
@log_request("dual_channel")
def transcribe_formatted():
    """
    Transcribe one or two audio URLs via VAD pipeline and return
    merged/formatted rich response.
    """
    t0  = time.time()
    rid = getattr(g, "request_id", str(uuid.uuid4())[:8])

    LOGGER.info(f"[{rid}] ===== /transcriptions/formatted START =====")
    LOGGER.info(f"[{rid}] [REQ] form={dict(request.form)}")

    validator       = RequestValidator(request.form)
    conversation_id = validator.get_required("conversation_id")
    audio_url_1     = validator.get_required("audio_url_1")
    audio_url_0     = validator.get_optional("audio_url_0")

    if not validator.is_valid():
        LOGGER.error(f"[{rid}] Validation failed: {validator.errors}")
        return validator.error_response()

    LOGGER.info(f"[{rid}] conversation_id={conversation_id}  url_1={audio_url_1}  url_0={audio_url_0}")

    temp_files = []

    try:
        # ------------------------------------------------------------------
        # STEP A: Download audio_url_1
        # ------------------------------------------------------------------
        tA0 = time.time()
        LOGGER.info(f"[{rid}] [STEP A] Downloading audio_url_1 ...")

        audio_data1 = AudioHelper().download_audio(audio_url_1)

        tA1 = time.time()
        if audio_data1 is None:
            LOGGER.error(f"[{rid}] [STEP A FAIL] Failed to download audio_url_1 after {tA1 - tA0:.2f}s")
            return jsonify({"error": "Failed to download audio_url_1"}), 400
        LOGGER.info(f"[{rid}] [STEP A DONE] Download took {tA1 - tA0:.2f}s")

        # Save temp file for duration measurement
        temp_audio_path = os.path.join(RESULTS_DIR, f"{conversation_id}_temp_audio1.wav")
        with open(temp_audio_path, "wb") as fout:
            fout.write(audio_data1.getvalue())
        temp_files.append(temp_audio_path)

        # ------------------------------------------------------------------
        # STEP B: Get audio duration + pick backend URLs
        # ------------------------------------------------------------------
        tB0 = time.time()
        LOGGER.info(f"[{rid}] [STEP B] Getting audio duration ...")

        audio_duration = CallAnalyzer.get_audio_duration_from_file(temp_audio_path)

        tB1 = time.time()
        LOGGER.info(f"[{rid}] [STEP B DONE] audio_duration={audio_duration:.2f}s  took {tB1 - tB0:.2f}s")

        ASR_URL = get_next_asr_url()
        VAD_URL = get_next_vad_url()
        LOGGER.info(f"[{rid}] [STEP B] Selected ASR URL: {ASR_URL}")
        LOGGER.info(f"[{rid}] [STEP B] Selected VAD URL: {VAD_URL}")

        transcription_service = TranscriptionService(ASR_URL, None, VAD_URL)

        # ------------------------------------------------------------------
        # STEP C: Transcribe audio_url_1 via VAD
        # ------------------------------------------------------------------
        tC0 = time.time()
        LOGGER.info(f"[{rid}] [STEP C] Processing audio1.wav via VAD ...")

        transcription1 = transcription_service.process_audio_VAD(audio_data1, "audio1.wav")

        tC1 = time.time()
        LOGGER.info(f"[{rid}] [STEP C DONE] process_audio_VAD took {tC1 - tC0:.2f}s  entries={len(transcription1) if transcription1 else 0}")

        file1_path = os.path.join(RESULTS_DIR, f"{conversation_id}_audio1.json")
        temp_files.append(file1_path)
        file_service.save_transcription_json(transcription1, file1_path)
        LOGGER.info(f"[{rid}] Saved transcription1 -> {file1_path}")
        final_json_path = file1_path

        # ------------------------------------------------------------------
        # STEP D: Download + transcribe audio_url_0 (optional)
        # ------------------------------------------------------------------
        file2_path, transcription2 = None, []
        if audio_url_0:
            tD0 = time.time()
            LOGGER.info(f"[{rid}] [STEP D] Downloading audio_url_0 ...")

            audio_data2 = AudioHelper().download_audio(audio_url_0)

            tD1 = time.time()
            if audio_data2 is None:
                LOGGER.error(f"[{rid}] [STEP D FAIL] Failed to download audio_url_0 after {tD1 - tD0:.2f}s")
                return jsonify({"error": "Failed to download audio_url_0"}), 400
            LOGGER.info(f"[{rid}] [STEP D] Download took {tD1 - tD0:.2f}s")

            tD2 = time.time()
            LOGGER.info(f"[{rid}] [STEP D] Processing audio2.wav via VAD ...")

            transcription2 = transcription_service.process_audio_VAD(audio_data2, "audio2.wav")

            tD3 = time.time()
            LOGGER.info(f"[{rid}] [STEP D DONE] process_audio_VAD took {tD3 - tD2:.2f}s  entries={len(transcription2) if transcription2 else 0}")

            file2_path = os.path.join(RESULTS_DIR, f"{conversation_id}_audio2.json")
            temp_files.append(file2_path)
            file_service.save_transcription_json(transcription2, file2_path)
            LOGGER.info(f"[{rid}] Saved transcription2 -> {file2_path}")

        # ------------------------------------------------------------------
        # STEP E: Merge transcripts (if both channels available)
        # ------------------------------------------------------------------
        if file2_path:
            tE0 = time.time()
            LOGGER.info(f"[{rid}] [STEP E] Merging transcripts ...")

            merged_path    = os.path.join(RESULTS_DIR, f"{conversation_id}_combined.json")
            transcriptions = file_service.merge_transcripts_json(file1_path, file2_path, merged_path)
            temp_files.append(merged_path)
            final_json_path = merged_path

            tE1 = time.time()
            LOGGER.info(f"[{rid}] [STEP E DONE] Merge complete. Total entries={len(transcriptions)}  took {tE1 - tE0:.2f}s")
        else:
            transcriptions = transcription1
            LOGGER.info(f"[{rid}] [STEP E] Single channel only. Total entries={len(transcriptions)}")

        # ------------------------------------------------------------------
        # STEP F: Build rich response
        # ------------------------------------------------------------------
        tF0 = time.time()
        LOGGER.info(f"[{rid}] [STEP F] Building rich response ...")

        response = JSONUtils.generate_rich_response(conversation_id, transcriptions, audio_duration, RESULTS_DIR)

        tF1 = time.time()
        total = time.time() - t0
        LOGGER.info(f"[{rid}] [STEP F DONE] rich response built in {tF1 - tF0:.2f}s")
        LOGGER.info(f"[{rid}] ===== /transcriptions/formatted COMPLETE in {total:.2f}s =====")

        return jsonify(response), 200

    except Exception as e:
        LOGGER.error(f"[{rid}] Exception occurred -> {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    LOGGER.info(f"[{rid}] Removed temp file -> {temp_file}")
                except Exception as cleanup_err:
                    LOGGER.warning(f"[{rid}] Failed to remove temp file {temp_file}: {cleanup_err}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
