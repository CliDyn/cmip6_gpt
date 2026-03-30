"""
In-memory session manager replacing st.session_state.
Each session stores messages, opendap state, and pending expanders.
"""
from typing import Any, Dict, List
import uuid


class Session:
    """A single chat session."""
    __slots__ = ("messages", "opendap_states", "pending_expanders", "expanded_models")

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.opendap_states: Dict[str, Any] = {}
        self.pending_expanders: List[Dict[str, Any]] = []
        self.expanded_models: set = set()

    def clear(self):
        self.messages.clear()
        self.opendap_states.clear()
        self.pending_expanders.clear()
        self.expanded_models.clear()


class SessionManager:
    """Thread-safe in-memory session store."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session()
        return self._sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def create_session(self) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = Session()
        return sid

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return self.get_session(session_id).messages

    def append_message(self, session_id: str, message: Dict[str, Any]):
        self.get_session(session_id).messages.append(message)


# Global singleton
session_manager = SessionManager()
