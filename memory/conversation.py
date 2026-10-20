"""Conversation Memory — persistată în PostgreSQL (tabelul chat_messages).

LLM-ul e stateless: noi ținem istoricul și-l retrimitem la fiecare request.
"""

from sqlalchemy import select

from db.database import transaction
from db.models import ChatMessage


class ConversationMemory:
    def load_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        """Istoricul unei sesiuni, în ordine cronologică: [{"role", "content"}, ...]."""
        with transaction() as db:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at, ChatMessage.id)
            )
            rows = db.execute(stmt).scalars().all()
            msgs = [{"role": m.role, "content": m.content} for m in rows]

        if limit is not None:
            msgs = msgs[-limit:]   # păstrăm ultimele N mesaje = fereastra short-term
        return msgs

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Salvează un tur complet: mesajul user-ului + răspunsul agentului."""
        with transaction() as db:
            db.add(ChatMessage(session_id=session_id, role="user", content=user_msg))
            db.add(ChatMessage(session_id=session_id, role="assistant", content=assistant_msg))