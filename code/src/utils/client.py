import json
import os
import sys
import time
import uuid
from contextlib import suppress
import requests

from dotenv import load_dotenv


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()


class Mem0Client:
    def __init__(self, enable_graph=False):
        from mem0 import MemoryClient

        self.client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        self.enable_graph = enable_graph

    def add(self, messages, user_id, timestamp, batch_size=2):
        max_retries = 5
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    if self.enable_graph:
                        self.client.add(
                            messages=batch_messages,
                            timestamp=timestamp,
                            user_id=user_id,
                            enable_graph=True,
                        )
                    else:
                        self.client.add(
                            messages=batch_messages,
                            timestamp=timestamp,
                            user_id=user_id,
                        )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

    def search(self, query, user_id, top_k):
        res = self.client.search(
            query=query,
            top_k=top_k,
            user_id=user_id,
            enable_graph=self.enable_graph,
            filters={"AND": [{"user_id": f"{user_id}"}]},
        )
        return res


class MemobaseClient:
    def __init__(self):
        from .memobase import MemoBaseClient

        self.client = MemoBaseClient(
            project_url=os.getenv("MEMOBASE_PROJECT_URL"), api_key=os.getenv("MEMOBASE_API_KEY")
        )

    def add(self, messages, user_id, batch_size=2):
        """
        messages = [{"role": "assistant", "content": data, "created_at": iso_date}]
        """
        from .memobase import ChatBlob

        real_uid = self.string_to_uuid(user_id)
        user = self.client.get_or_create_user(real_uid)
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    _ = user.insert(ChatBlob(messages=batch_messages), sync=True)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

    def search(self, query, user_id, top_k):
        real_uid = self.string_to_uuid(user_id)
        user = self.client.get_user(real_uid, no_get=True)
        memories = user.context(
            max_token_size=top_k * 100,
            chats=[{"role": "user", "content": query}],
            event_similarity_threshold=0.2,
            fill_window_with_events=True,
        )
        return memories

    def delete_user(self, user_id):
        from .memobase.error import ServerError

        real_uid = self.string_to_uuid(user_id)
        with suppress(ServerError):
            self.client.delete_user(real_uid)

    def string_to_uuid(self, s: str, salt="memobase_client"):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s + salt))


class MemosApiClient:
    def __init__(self):
        self.memos_url = os.getenv("MEMOS_URL")
        self.headers = {"Content-Type": "application/json", "Authorization": os.getenv("MEMOS_KEY")}

    def add(self, messages, user_id, conv_id, batch_size: int = 9999):
        """
        messages = [{"role": "assistant", "content": data, "chat_time": date_str}]
        """
        url = f"{self.memos_url}/product/add"
        added_memories = []
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            payload = json.dumps(
                {
                    "messages": batch_messages,
                    "user_id": user_id,
                    "mem_cube_id": user_id,
                    "conversation_id": conv_id,
                }
            )
            response = requests.request("POST", url, data=payload, headers=self.headers)
            assert response.status_code == 200, response.text
            assert json.loads(response.text)["message"] == "Memory added successfully", (
                response.text
            )
            added_memories += json.loads(response.text)["data"]
        return added_memories

    def search(self, query, user_id, top_k):
        """Search memories."""
        url = f"{self.memos_url}/product/search"
        payload = json.dumps(
            {
                "query": query,
                "user_id": user_id,
                "mem_cube_id": user_id,
                "conversation_id": "",
                "top_k": top_k,
                "mode": os.getenv("SEARCH_MODE", "fast"),
                "include_preference": True,
                "pref_top_k": 6,
            },
            ensure_ascii=False,
        )
        response = requests.request("POST", url, data=payload, headers=self.headers)
        assert response.status_code == 200, response.text
        assert json.loads(response.text)["message"] == "Search completed successfully", (
            response.text
        )
        return json.loads(response.text)["data"]


class MemosApiOnlineClient:
    def __init__(self):
        self.memos_url = os.getenv("MEMOS_ONLINE_URL")
        self.headers = {"Content-Type": "application/json", "Authorization": os.getenv("MEMOS_KEY")}

    def add(self, messages, user_id, conv_id=None, batch_size: int = 9999):
        url = f"{self.memos_url}/add/message"
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            payload = json.dumps(
                {
                    "messages": batch_messages,
                    "user_id": user_id,
                    "conversation_id": conv_id,
                }
            )

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.request("POST", url, data=payload, headers=self.headers)
                    assert response.status_code == 200, response.text
                    assert json.loads(response.text)["message"] == "ok", response.text
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise e

    def search(self, query, user_id, top_k):
        """Search memories."""
        url = f"{self.memos_url}/search/memory"
        payload = json.dumps(
            {
                "query": query,
                "user_id": user_id,
                "memory_limit_number": top_k,
                "mode": os.getenv("SEARCH_MODE", "fast"),
                "include_preference": True,
                "pref_top_k": 6,
            }
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.request("POST", url, data=payload, headers=self.headers)
                assert response.status_code == 200, response.text
                assert json.loads(response.text)["message"] == "ok", response.text
                text_mem_res = json.loads(response.text)["data"]["memory_detail_list"]
                pref_mem_res = json.loads(response.text)["data"]["preference_detail_list"]
                preference_note = json.loads(response.text)["data"]["preference_note"]
                for i in text_mem_res:
                    i.update({"memory": i.pop("memory_value")})

                explicit_prefs = [
                    p["preference"]
                    for p in pref_mem_res
                    if p.get("preference_type", "") == "explicit_preference"
                ]
                implicit_prefs = [
                    p["preference"]
                    for p in pref_mem_res
                    if p.get("preference_type", "") == "implicit_preference"
                ]

                pref_parts = []
                if explicit_prefs:
                    pref_parts.append(
                        "Explicit Preference:\n"
                        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(explicit_prefs))
                    )
                if implicit_prefs:
                    pref_parts.append(
                        "Implicit Preference:\n"
                        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(implicit_prefs))
                    )

                pref_string = "\n".join(pref_parts) + preference_note

                return {"text_mem": [{"memories": text_mem_res}], "pref_string": pref_string}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e


class SupermemoryClient:
    def __init__(self, api_key=os.getenv("SUPERMEMORY_API_KEY")):
        from supermemory import Supermemory

        self.client = Supermemory(api_key=api_key)

    def add(self, messages, user_id):
        content = "\n".join(
            [f"{msg['chat_time']} {msg['role']}: {msg['content']}" for msg in messages]
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.client.memories.add(content=content, container_tag=user_id)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e

    def search(self, query, user_id, top_k):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                results = self.client.search.memories(
                    q=query,
                    container_tag=user_id,
                    threshold=0,
                    rerank=True,
                    rewrite_query=True,
                    limit=top_k,
                )
                context = "\n\n".join([r.memory for r in results.results])
                return context
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise e


class LightMemoryClient:
    def __init__(self, user_id):
        from .lightmem.memory.lightmem import LightMemory
        API_KEY = os.getenv("CHAT_MODEL_API_KEY")
        API_BASE_URL = os.getenv("CHAT_MODEL_BASE_URL")
        LLM_MODEL='gpt-4o-mini'
        EMBEDDING_MODEL_PATH='sentence-transformers/all-MiniLM-L6-v2'
        LLMLINGUA_MODEL_PATH='microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank'

        config_dict = {
            "pre_compress": True,
            "pre_compressor": {
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": LLMLINGUA_MODEL_PATH,
                        "device_map": "cuda",
                        "use_llmlingua2": True,
                    },
                }
            },
            "topic_segment": True,
            "precomp_topic_shared": True,
            "topic_segmenter": {
                "model_name": "llmlingua-2",
            },
            "messages_use": "user_only",
            "metadata_generate": True,
            "text_summary": True,
            "memory_manager": {
                "model_name": 'openai', # such as 'openai' or 'ollama' ...
                "configs": {
                    "model": LLM_MODEL,
                    "api_key": API_KEY,
                    "max_tokens": 16000,
                    "openai_base_url": API_BASE_URL # API model specific, such as 'openai_base_url' or 'deepseek_base_url' ...
                }
            },
            "extract_threshold": 0.1,
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": "huggingface",
                "configs": {
                    "model": EMBEDDING_MODEL_PATH,
                    "embedding_dims": 384,
                    "model_kwargs": {"device": "cuda"},
                },
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": user_id,
                    "embedding_model_dims": 384,
                    "path": f"./tmp/{user_id}", 
                }
            },
            "update": "offline"
        }
        self.lightmem = LightMemory.from_config(config_dict)

    def add(self, messages):
        store_result = self.lightmem.add_memory(
            messages=messages,
            force_segment=True,
            force_extract=True
        )

        return store_result

    def search(self, query, top_k):
        related_memories = self.lightmem.retrieve(query, limit=top_k)
        return related_memories