class MemoryService:

    def __init__(self, max_messages=6):
        self.max_messages = max_messages
        self.history = []

    def add_user(self, message):
        self.history.append({
            "role": "user",
            "content": message
        })
        self._trim()

    def add_assistant(self, message):
        self.history.append({
            "role": "assistant",
            "content": message
        })
        self._trim()

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []

    def _trim(self):
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]    