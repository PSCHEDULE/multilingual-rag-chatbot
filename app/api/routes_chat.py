"""Chat SSE streaming endpoint."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ChatRequest
from app.api.session_store import append_message, get_messages, set_messages
from app.graph.workflow import stream_tokens

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


def _sse_events(req: ChatRequest) -> Iterator[dict]:
    session_id = req.session_id or str(uuid.uuid4())
    history = append_message(session_id, "user", req.message)

    state = {
        "messages": history,
        "session_id": session_id,
        "language": req.language,
    }
    answer_acc: list[str] = []
    try:
        for evt in stream_tokens(state):
            name = evt.get("event") or "message"
            data = evt.get("data") or {}
            if name == "meta":
                data = {**data, "session_id": session_id}
            if name == "token":
                answer_acc.append(str(data.get("text") or ""))
            if name == "done":
                final = data.get("answer") or "".join(answer_acc)
                if final:
                    msgs = get_messages(session_id)
                    # user already appended; add assistant
                    if not msgs or msgs[-1].get("role") != "assistant":
                        append_message(session_id, "assistant", final)
                    else:
                        set_messages(
                            session_id,
                            msgs[:-1] + [{"role": "assistant", "content": final}],
                        )
                data = {**data, "session_id": session_id}
            yield {"event": name, "data": json.dumps(data, ensure_ascii=False)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("SSE chat failed: %s", exc)
        yield {
            "event": "error",
            "data": json.dumps({"message": str(exc)}, ensure_ascii=False),
        }
        yield {
            "event": "done",
            "data": json.dumps(
                {"finish_reason": "error", "session_id": session_id},
                ensure_ascii=False,
            ),
        }


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest) -> EventSourceResponse:
    """Stream assistant tokens and metadata over Server-Sent Events."""
    return EventSourceResponse(_sse_events(body))
