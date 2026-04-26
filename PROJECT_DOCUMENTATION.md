# Call Center Quality Auditing Pipeline

## Overview
This project performs automated quality auditing for call-center conversations. It ingests call audio, identifies who spoke when, transcribes each segment using fine-tuned ASR models, and computes call-level metrics and structured transcripts. The output is designed for downstream quality evaluation (met/not-met rules with evidence) and customer reporting.

Repos covered in this document:
- Model services and deployment: `/media/vocab/DATA5/Vocab_Services_/DockerImage_CRED_ASR_/SURIYA/DUAL-CHANNEL-ASR-SERVICE`
- Orchestrator API and pipeline integration: `/media/vocab/DATA5/Vocab_Services_/DockerImage_CRED_ASR_/SURIYA/ORCHESTRATOR_API/Final_API`

## Goals
- Accurate speaker segmentation for mono and dual-channel calls.
- High-quality transcription using fine-tuned ASR models trained on internal call-center data.
- Rich, structured outputs with timestamps, speaker labels, and call-level timing metrics.
- GPU-backed services with a scalable, service-based architecture.

## Architecture
The system is split into two layers:

1. Model Services Layer
- ASR service
- Diarization service
- VAD service

2. Orchestrator API Layer
- Accepts audio uploads or URLs
- Selects backend services
- Runs diarization or VAD segmentation
- Sends segments to ASR
- Merges outputs and returns rich JSON

## Audio Handling
Supported input sources:
- File upload
- Remote audio URL

Supported formats (auto-converted to WAV 16kHz mono):
- WAV, MP3, OGG, FLAC, AAC, M4A, WMA, ASF
- `application/octet-stream` with auto-detect

Mono vs dual-channel:
- Mono calls use diarization or VAD.
- Dual-channel calls are handled by submitting two audio URLs (one per channel) and merging results.

## Pipeline Flow
1. Input: audio upload or URL(s) with `conversation_id`.
2. Audio normalization: convert to 16kHz mono WAV.
3. Segmentation:
- Diarization for speaker separation
- VAD for time-based segments
4. Transcription: each segment sent to ASR.
5. Speaker labeling:
- Diarization path uses keyword heuristics to assign `agent` vs `customer`.
- Dual-channel VAD path labels `speaker_1` for channel 1 and `speaker_0` for channel 0.
6. Output: structured JSON with transcripts and call metrics.

## Orchestrator API Endpoints

`POST /transcribe`
- Upload a file via `multipart/form-data` with `file` and optional `conversation_id`.
- Uses diarization + ASR.

`POST /url_transcribe`
- Form fields: `audio_url`, optional `file_name`, `conversation_id`.
- Uses diarization + ASR.

`POST /transcriptions/formatted`
- Form fields: `conversation_id`, `audio_url_1`, optional `audio_url_0`.
- Uses VAD + ASR per channel and merges output.
- Intended for dual-channel or per-speaker audio.

## Output Schema (High Level)
The API returns a rich response including:
- `transcriptions`: list of segments with `start_time`, `end_time`, `speaker_id`, `transcription`
- `agent_transcript`, `customer_transcript`
- `utterances` and `alternatives` with word-level metadata
- Call metrics:
- `call_duration`
- `silence/hold_duration` and percentage
- `silence_before_start`, `silence_after_end`
- `cross_talk_duration`
- `total_speaking_time` and percentage

These outputs feed downstream quality auditing logic for met/not-met parameters and evidence extraction.

## Model Services Deployment
Deployment is handled via Docker Compose with GPU support.

Primary compose file:
- `docker-compose-all-v1.4.yml`

Service ports:
- ASR: host `7001` -> container `7001`
- VAD: host `8001` -> container `8001`
- Diarization: host `6001` -> container `6001`

Models:
- ASR v1.0: `./ASR_v1.0/en`
- ASR v1.4: `./ASR_v1.4/en`
- VAD v2.0 model: `VAD_v2.0/vad_v2/models/vad_model.joblib`

## Fine-Tuning Summary
ASR, diarization, and VAD models are fine-tuned on internal real call-center audio to improve:
- Domain vocabulary recognition
- Speaker separation quality
- Robustness to real-world call acoustics

## Operational Notes
- The orchestrator uses round-robin selection for ASR, diarization, and VAD service URLs.
- Audio duration is computed from a temporary WAV for timing metrics.
- Logs include step-level timings for performance monitoring.

Key env variables in the orchestrator:
- `SERVER_HOSTNAME` (defaults to `http://27.111.72.61`)
- `STORE_AUDIO` (boolean)
- `AUDIO_LOCATION` (where to store downloaded audio)

## Quality Auditing Integration
After transcription, the outputs are consumed by the quality auditing layer:
- Parameter evaluation: met / not-met
- Evidence extraction from transcripts
- Aggregated call scoring for customer reporting

## Current Limitations
- Dual-channel expects two separate audio URLs rather than a single stereo file.
- Speaker role assignment in diarization mode relies on keyword heuristics.
- Quality rules engine is downstream and not part of these repos.
