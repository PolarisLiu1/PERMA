import os
import sys

from datetime import datetime, timezone

INIT_RESULT = {
    "add_input_prompt": [],
    "add_output_prompt": [],
    "api_call_nums": 0
}

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ingest_session(session, date, user_id, session_id, frame, client):
    messages = []
    if "mem0" in frame:
        for _idx, msg in enumerate(session):
            messages.append({"role": msg["role"], "content": msg["content"][:8000]})
        client.add(messages, user_id, int(date.timestamp()), batch_size=2)
    elif frame == "memobase":
        for _idx, msg in enumerate(session):
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"][:8000],
                    "created_at": date.isoformat(),
                }
            )
        client.add(messages, user_id, batch_size=2)

    elif "memos-api" in frame:
        for msg in session:
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"][:8000],
                    "chat_time": date.isoformat(),
                }
            )
        if messages:
            client.add(messages=messages, user_id=user_id, conv_id=session_id, batch_size=2)

    elif frame == "supermemory":
        for _idx, msg in enumerate(session):
            messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"][:8000],
                    "chat_time": date.isoformat(),
                }
            )
        client.add(messages, user_id)

    elif frame == "lightmem":
        num_turns = len(session) // 2 
        for turn_idx in range(num_turns):
            turn_messages = session[turn_idx*2 : turn_idx*2 + 2]
            for msg in turn_messages:
                msg["time_stamp"] = date.strftime('%Y-%m-%d %H:%M:%S')
            messages.extend(turn_messages)
            store_result = client.add(turn_messages)

    print(
        f"[{frame}] ✅ Session {session_id}: Ingested {len(messages)} messages at {date.isoformat()}"
    )