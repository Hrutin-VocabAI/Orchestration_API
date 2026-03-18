from flask import jsonify


class RequestValidator:
    """
    Helper class to validate incoming form-data requests.
    Ensures required fields are present and non-empty.
    """

    def __init__(self, form):
        self.form = form
        self.errors = []

    def get_required(self, key: str) -> str:
        """
        Get a required field from form-data.
        Adds an error if missing or empty.
        """
        value = self.form.get(key, "").strip()
        if not value:
            self.errors.append(f"{key} is required and cannot be empty")
            return None
        return value

    def get_optional(self, key: str) -> str:
        """
        Get an optional field from form-data.
        Returns an empty string if not provided.
        """
        return self.form.get(key, "").strip()

    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self.errors) == 0

    def error_response(self):
        """Return errors as JSON response."""
        return jsonify({"errors": self.errors}), 400

