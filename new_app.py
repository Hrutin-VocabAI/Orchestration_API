import os
import random
from distutils.util import strtobool
from io import BytesIO
from itertools import cycle

from flask import Flask, request, jsonify

from audio_downloader import AudioHelper
from request_validator import RequestValidator
from logger import get_logger
from transcription_service import TranscriptionService

from function_services import (
    TranscriptHandler,
    SentimentProcessor,
    CallAnalyzer,
)

from file_service import FileService
from file_service import FileService
from gdrive_uploader import GDriveUploader

LOGGER = get_logger(__name__)
HOSTNAME = os.getenv("SERVER_HOSTNAME", "http://localhost")

try:
    STORE_AUDIO = bool(strtobool(os.getenv("STORE_AUDIO", "False")))
except ValueError:
    STORE_AUDIO = False

# GDrive upload configuration
ENABLE_GDRIVE_UPLOAD = bool(strtobool(os.getenv("ENABLE_GDRIVE_UPLOAD", "False")))
GDRIVE_REMOTE = os.getenv("GDRIVE_REMOTE", "gdrive:")
GDRIVE_ROOT_DIR = os.getenv("GDRIVE_ROOT_DIR", "voice2chat")
GDRIVE_UPLOAD_WAIT_SECS = int(os.getenv("GDRIVE_UPLOAD_WAIT_SECS", "300"))

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
def transcribe():
    if "file" not in request.files:
        LOGGER.error("No file provided")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        LOGGER.error("Empty file")
        return jsonify({"error": "Empty file"}), 400

    audio_data = file.read()
    buffer = AudioHelper().convert_if_needed(BytesIO(audio_data), file.content_type)

    # Pick backend dynamically
    asr_url = get_next_asr_url()
    diar_url = get_next_diarization_url()

    transcription_service = TranscriptionService(asr_url, diar_url,None)
    transcription_result = transcription_service.process_audio(buffer, file.filename)

    return jsonify({"transcription": transcription_result})


@app.route("/url_transcribe", methods=["POST"])
def transcribe_url():
    if "audio_url" not in request.form:
        LOGGER.error("No audio_url provided")
        return jsonify({"error": "No audio_url provided"}), 400

    filename = request.form.get("file_name", None)
    audio_url = request.form["audio_url"]

    if audio_url == "":
        LOGGER.error("Empty audio_url")
        return jsonify({"error": "Empty audio_url"}), 400

    audio_data = AudioHelper(STORE_AUDIO).download_audio(audio_url, filename=filename)
    if audio_data is None:
        LOGGER.error("Failed to download audio")
        return jsonify({"error": "Failed to download audio"}), 400

    if filename is None:
        filename = "sample_audio"

    # Pick backend dynamically
    asr_url = get_next_asr_url()
    diar_url = get_next_diarization_url()

    transcription_service = TranscriptionService(asr_url, diar_url, "Empty")
    transcription_result = transcription_service.process_audio(audio_data, filename + ".wav")

    return jsonify({"transcription": transcription_result})


# ---------------- VAD EndpointS ----------------


@app.route("/url_channel_transcribe", methods=["POST"])
def url_channel_transcribe():
    """
    API Endpoint: Transcribe audio from one or two URLs.

    Required form-data:
        - conversation_id (str): unique conversation identifier
        - audio_url_1 (str): URL to the first audio file

    Optional form-data:
        - audio_url_0 (str): URL to the second audio file

    Returns:
        JSON response with:
        {
            "conversation_id": <conversation_id>,
            "transcriptions": [ { act_id, start_time, end_time, speaker_id, transcription }, ... ]
        }
    """



    validator = RequestValidator(request.form)
    # Validate fields
    conversation_id = validator.get_required("conversation_id")
    audio_url_1 = validator.get_required("audio_url_0")
    audio_url_0 = validator.get_optional("audio_url_1")

    if not validator.is_valid():
        return validator.error_response()

    # -------- Process audio_url_1 (mandatory) --------
    audio_data1 = AudioHelper().download_audio(audio_url_1)
    if audio_data1 is None:
        return jsonify({"error": "Failed to download audio_url_1"}), 400

    ASR_URL =  get_next_asr_url()
    VAD_URL =  get_next_vad_url()  # get_next_vad_url()

    transcription_service = TranscriptionService(ASR_URL , None, VAD_URL) # add code

    transcription1 = transcription_service.process_audio(audio_data1, "audio1.wav")
    file1_path = os.path.join(RESULTS_DIR, f"{conversation_id}_audio1.json")
    file_service.save_transcription_json(transcription1, file1_path)

    # -------- Process audio_url_0 (optional) --------
    file2_path = None
    transcription2 = []
    if audio_url_0:
        audio_data2 = AudioHelper().download_audio(audio_url_0)
        if audio_data2 is None:
            return jsonify({"error": "Failed to download audio_url_0"}), 400

        transcription2 = transcription_service.process_audio(audio_data2, "audio2.wav")
        file2_path = os.path.join(RESULTS_DIR, f"{conversation_id}_audio2.json")
        file_service.save_transcription_json(transcription2, file2_path)

    # -------- Merge transcripts --------
    if file2_path:  # Merge if both exist
        output_path = os.path.join(RESULTS_DIR, f"{conversation_id}_combined.json")
        merged_transcriptions = file_service.merge_transcripts_json(file1_path, file2_path, output_path)
        transcriptions = merged_transcriptions
    else:
        transcriptions = transcription1

    # -------- Final Response --------
    results = {
        "conversation_id": conversation_id,
        "transcriptions": transcriptions
    }

    return jsonify(results)

@app.route("/transcriptions/formatted", methods=["POST"])
def transcribe_formatted():
    """
    RESTful API Endpoint: Transcribe one or two audio URLs and return merged/formatted transcription.
    - POST /transcriptions/formatted
    - Request JSON: {
          "conversation_id": "...",
          "audio_url_1": "...",
          "audio_url_0": "..."  # optional
      }
    - Response: Combined transcription JSON with agent/customer splits, utterances, alternatives, etc.
    - Cleans up all audio and intermediate JSON files after processing.
    """

    print("DEBUG: Incoming request ->", request.form)  # 👈 check raw input
    validator = RequestValidator(request.form) # validating the input form data
    # Validate required/optional fields
    conversation_id = validator.get_required("conversation_id")
    audio_url_1 = validator.get_required("audio_url_1")
    audio_url_0 = validator.get_optional("audio_url_0")

    print("DEBUG: conversation_id =", conversation_id)
    print("DEBUG: audio_url_1 =", audio_url_1)
    print("DEBUG: audio_url_0 =", audio_url_0)

    if not validator.is_valid():
        print("DEBUG: Validation failed ->", validator.errors)
        return validator.error_response()
    
    temp_files = []
    upload_events = []

    try:
        # -------- Process audio_url_1 --------
        print("DEBUG: Downloading audio_url_1...")
        audio_data1 = AudioHelper().download_audio(audio_url_1)
        if audio_data1 is None:
            print("ERROR: Failed to download audio_url_1")
            return jsonify({"error": "Failed to download audio_url_1"}), 400

        # Save audio data to temporary file to get duration and for GDrive upload
        temp_audio_path = os.path.join(RESULTS_DIR, f"{conversation_id}_temp_audio1.wav")
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_data1.getvalue())  # Get bytes from BytesIO object
        temp_files.append(temp_audio_path)

        # Kick off background upload for agent audio (audio_url_1)
        if ENABLE_GDRIVE_UPLOAD:
            try:
                uploader = GDriveUploader(remote=GDRIVE_REMOTE, root_dir=GDRIVE_ROOT_DIR)
                event = uploader.start_upload(
                    local_path=temp_audio_path,
                    conversation_id=conversation_id,
                    dest_filename=f"{conversation_id}_1.wav",
                )
                if event:
                    upload_events.append(event)
            except Exception as up_err:
                LOGGER.error(f"Failed to start GDrive upload for agent audio: {up_err}")
        
        ASR_URL =  get_next_asr_url()
        VAD_URL =  get_next_vad_url()  # get_next_vad_url()

        transcription_service = TranscriptionService(ASR_URL , None, VAD_URL) # add code

        # Get audio duration from the first audio file
        audio_duration = CallAnalyzer.get_audio_duration_from_file(temp_audio_path)
        print("DEBUG: Audio duration =", audio_duration)

        print("DEBUG: Processing audio1.wav...")
        transcription1 = transcription_service.process_audio_VAD(audio_data1, "audio1.wav")
        print("DEBUG: Transcription1 length =", len(transcription1) if transcription1 else 0)

        file1_path = os.path.join(RESULTS_DIR, f"{conversation_id}_audio1.json")
        temp_files.append(file1_path)
        file_service.save_transcription_json(transcription1, file1_path)
        print("DEBUG: Saved transcription1 ->", file1_path)
        final_json_path = file1_path

        # -------- Process audio_url_0 (if available) --------
        file2_path, transcription2 = None, []
        if audio_url_0:
            print("DEBUG: Downloading audio_url_0...")
            audio_data2 = AudioHelper().download_audio(audio_url_0)
            if audio_data2 is None:
                print("ERROR: Failed to download audio_url_0")
                return jsonify({"error": "Failed to download audio_url_0"}), 400

            # Save customer audio to temporary file for GDrive upload
            temp_audio_path_2 = os.path.join(RESULTS_DIR, f"{conversation_id}_temp_audio0.wav")
            with open(temp_audio_path_2, 'wb') as f:
                f.write(audio_data2.getvalue())
            temp_files.append(temp_audio_path_2)


            if ENABLE_GDRIVE_UPLOAD:
                print(f"DEBUG: Starting upload for {conversation_id}")
                print(f"DEBUG: GDRIVE_REMOTE={GDRIVE_REMOTE}")
                print(f"DEBUG: GDRIVE_ROOT_DIR={GDRIVE_ROOT_DIR}")
            # Kick off background upload for customer audio (audio_url_0)
            if ENABLE_GDRIVE_UPLOAD:
                try:
                    uploader = GDriveUploader(remote=GDRIVE_REMOTE, root_dir=GDRIVE_ROOT_DIR)
                    event = uploader.start_upload(
                        local_path=temp_audio_path_2,
                        conversation_id=conversation_id,
                        dest_filename=f"{conversation_id}_0.wav",
                    )
                    if event:
                        upload_events.append(event)
                except Exception as up_err:
                    LOGGER.error(f"Failed to start GDrive upload for customer audio: {up_err}")

            print("DEBUG: Processing audio2.wav...")
            transcription2 = transcription_service.process_audio_VAD(audio_data2, "audio2.wav")
            print("DEBUG: Transcription2 length =", len(transcription2) if transcription2 else 0)

            file2_path = os.path.join(RESULTS_DIR, f"{conversation_id}_audio2.json")
            temp_files.append(file2_path)
            file_service.save_transcription_json(transcription2, file2_path)
            print("DEBUG: Saved transcription2 ->", file2_path)

        # -------- Merge transcripts --------
        if file2_path:
            print("DEBUG: Merging transcripts...")
            merged_path = os.path.join(RESULTS_DIR, f"{conversation_id}_combined.json")
            merged_transcriptions = file_service.merge_transcripts_json(file1_path, file2_path, merged_path)
            temp_files.append(merged_path)
            transcriptions = merged_transcriptions
            final_json_path = merged_path
            print("DEBUG: Merge complete. Total =", len(transcriptions))
        else:
            transcriptions = transcription1
            print("DEBUG: Using only transcription1. Total =", len(transcriptions))

        # -------- Post-processing with TranscriptHandler --------
        print("DEBUG: Running TranscriptHandler + SentimentProcessor...")
        transcript_conf_score = TranscriptHandler.get_transcripts_with_confidence(final_json_path)
        all_transcriptions = TranscriptHandler.get_all_transcriptions(final_json_path)
        agent_transcript = TranscriptHandler.extract_agent_transcripts(final_json_path)
        customer_transcript = TranscriptHandler.extract_customer_transcripts(final_json_path)
        utterances, alternatives = SentimentProcessor.combine_data(transcript_conf_score)

        # -------- Call Analysis --------
        print("DEBUG: Running CallAnalyzer...")
        call_analysis = CallAnalyzer.analyze_call_timing(transcriptions, audio_duration)
        print("DEBUG: Call analysis results =", call_analysis)

        # -------- Build final response --------
        response = {
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
            # "audio_file_name": "audio2.wav" if file2_path else "audio1.wav",
            # "id": conversation_id,
        }

        print("DEBUG: Final response prepared")
        return jsonify(response), 200

    except Exception as e:
        print("ERROR: Exception occurred ->", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # If uploads were started, wait for them to complete before deleting temp files
        if upload_events:
            try:
                for ev in upload_events:
                    ev.wait(timeout=GDRIVE_UPLOAD_WAIT_SECS)
            except Exception as wait_err:
                LOGGER.error(f"Error while waiting for GDrive uploads: {wait_err}")

        # Clean up temp files after uploads complete
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print("DEBUG: Removed temp file ->", temp_file)
                except Exception as cleanup_err:
                    print("WARNING: Failed to remove temp file ->", temp_file, cleanup_err)




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
