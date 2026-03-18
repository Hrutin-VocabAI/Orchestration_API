import json
import os


class FileService:
    """
    Service class for handling file-based operations such as
    saving and merging transcription JSON files.
    """

    @staticmethod
    def save_transcription_json(data: dict, file_path: str):
        """
        Save transcription data to a JSON file.

        Args:
            data (dict): Transcription data to save.
            file_path (str): Path where the JSON file will be saved.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def merge_transcripts_json(file1_path: str, file2_path: str, output_path: str) -> list:
        """
        Merge two transcription JSON files into one, sorted by start_time.
        - file1: speaker_id stays as is (speaker_1)
        - file2: all speaker_id set to speaker_0

        Args:
            file1_path (str): Path to the first transcription file (speaker IDs preserved).
            file2_path (str): Path to the second transcription file (speaker IDs overridden).
            output_path (str): Path where the merged file will be saved.

        Returns:
            list: Combined list of transcript entries.
        """
        all_transcripts = []

        def load_file(file_path, speaker_override=None):
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        if speaker_override:
                            for item in data:
                                item['speaker_id'] = speaker_override
                        return data
                    elif isinstance(data, dict) and 'transcription' in data:
                        items = data['transcription']
                        if speaker_override:
                            for item in items:
                                item['speaker_id'] = speaker_override
                        return items
            return []

        # Collect transcripts
        all_transcripts.extend(load_file(file1_path))                # keep original speaker_ids
        all_transcripts.extend(load_file(file2_path, "speaker_0"))   # override to speaker_0

        # Sort by start_time if possible
        try:
            all_transcripts.sort(key=lambda x: x['start_time'])
        except (KeyError, TypeError):
            pass

        # Save merged file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(all_transcripts, outfile, indent=4, ensure_ascii=False)

        return all_transcripts
