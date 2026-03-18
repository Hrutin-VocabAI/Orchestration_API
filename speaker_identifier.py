import re


class SpeakerIdentifier:
    @staticmethod
    def score_speaker_transcription(speaker_text: str) -> int:
        agent_keywords = [
            "name", "calling", "apologize", "sorry", "check", "give", "queries", "feedback",
            "wait", "take", "working", "ensure", "payment", "thank", "understand", "assist"
        ]
        agent_pattern = re.compile(r"\b(" + "|".join(agent_keywords) + r")\b", re.IGNORECASE)
        if not isinstance(speaker_text, str):
            speaker_text = str(speaker_text) if speaker_text else ""
        matches = agent_pattern.findall(speaker_text)
        return len(matches)
