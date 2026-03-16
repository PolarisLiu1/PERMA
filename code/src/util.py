import os
import re
import json
import numpy as np
from datetime import datetime, timezone
from datetime import time
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

PERIOD_MAP = {
    "morning": time(6, 0),
    "afternoon": time(15, 0),
    "evening": time(19, 0),
}

def iso_or_default(date_str: Optional[str]) -> str:
    if not date_str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return date_str

def get_client(frame: str, user_id: str, api_key: str = None):
    if frame == "mem0" or frame == "mem0_graph":
        from utils.client import Mem0Client

        client = Mem0Client(enable_graph="graph" in frame)
        # client.client.delete_all(user_id=user_id)
    elif frame == "memos-api":
        from utils.client import MemosApiClient

        client = MemosApiClient()
    elif frame == "memos-api-online":
        from utils.client import MemosApiOnlineClient

        client = MemosApiOnlineClient()
    elif frame == "memobase":
        from utils.client import MemobaseClient

        client = MemobaseClient()
        client.delete_user(user_id)

    elif frame == "supermemory":
        from utils.client import SupermemoryClient
        if api_key:
            client = SupermemoryClient(api_key=api_key)
        else:
            client = SupermemoryClient()
    
    elif frame == "lightmem":
        from utils.client import LightMemoryClient

        client = LightMemoryClient(user_id)

    return client



def _link_affinity_mentions(context: List[Any], topic: List[str], user_preference: List[Dict[str, Any]], relevant_affinity_types: List[str]) -> List[Dict[str, Any]]:
    """
    Link emotional mentions in the dialogue to user preferences.

    1. First, use the event-level preference links to locate the full event dialogue where the corresponding preference appears.
    2. If the context is not in a paired structure or no match is found, fall back to the original logic.
    """
    links: List[Dict[str, Any]] = []
    if not relevant_affinity_types:
        return links
    def _link_by_event_preferences() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        mentions: List[Dict[str, Any]] = []
        for idx, pair in enumerate(context or []):
            try:
                pref_text = str(pair[1])
            except Exception:
                pref_text = ""
            if not pref_text:
                continue
            detect_pref = []
            for affi in relevant_affinity_types:
                for t in topic:
                    if affi in user_preference[t]:
                        detect_pref.append(f"{t}-{affi}")
            if any([re.search(r"\b" + re.escape(str(a)) + r"\b", pref_text, flags=re.IGNORECASE) for a in detect_pref]): 
                mentions.append({
                    "index": idx,
                    "value": pair[0],
                    "noise": pair[2],
                    "date": pair[3],
                    "task_type": pair[4],
                    "pref": pref_text
                })
        return mentions

    has_pair_shape = bool(context) and all(isinstance(c, (list, tuple)) and len(c) >= 2 for c in context)
    return  _link_by_event_preferences() if has_pair_shape else []

def ensure_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def save_json(output_path: str, obj: Any):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_sessions_from_dialogue(tdata):
    sessions = []
    for ev in ensure_list(tdata):
        conv = ev.get("conversation")
        if isinstance(conv, list) and conv:
            cleaned = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in conv]
            sessions.append(cleaned)
    return sessions


def _encode_dialog(dialog: List[Dict[str, str]], emb_model: SentenceTransformer) -> np.ndarray:
    sentence = "\n".join([f"{m['role']}: {m['content']}" for m in dialog])
    return emb_model.encode([sentence], normalize_embeddings=True)[0]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def eval_jsons(data_root: str, args) -> List[str]:
    eval_dir = os.path.join(data_root, "eval")
    paths: List[str] = []
    if os.path.isdir(eval_dir):
        try:
            for name in os.listdir(eval_dir):
                if name.lower().endswith(".json"):
                    condition = ("multi" in name) if args.multi_domain else ("multi" not in name)
                    if condition:
                        paths.append(os.path.join(eval_dir, name))
        except Exception:
            pass
    return sorted(paths)

def parse_date_with_period(s: str) -> datetime:
    from datetime import datetime
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    date_part = datetime.strptime(m.group(1), "%Y-%m-%d").date()
    rest = s[m.end():].strip().lower()
    for k, v in PERIOD_MAP.items():
        if k in rest:
            return datetime.combine(date_part, v)
    return datetime.combine(date_part, time(0, 0))