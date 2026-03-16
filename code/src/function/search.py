import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from time import time
from utils.prompts import (
    MEM0_CONTEXT_TEMPLATE,
    MEM0_GRAPH_CONTEXT_TEMPLATE,
    MEMOS_CONTEXT_TEMPLATE,
)


def mem0_search(client, query, user_id, top_k):
    start = time()
    results = client.search(query, user_id, top_k)
    memory = [f"{memory['created_at']}: {memory['memory']}" for memory in results["results"]]
    if client.enable_graph:
        graph = "\n".join(
            [
                f"  - 'source': {item.get('source', '?')} -> 'target': {item.get('target', '?')} "
                f"(relationship: {item.get('relationship', '?')})"
                for item in results.get("relations", [])
            ]
        )
        context = MEM0_GRAPH_CONTEXT_TEMPLATE.format(
            user_id=user_id, memories=memory, relations=graph
        )
    else:
        context = MEM0_CONTEXT_TEMPLATE.format(user_id=user_id, memories=memory)
    duration_ms = (time() - start) * 1000
    return context, duration_ms

def lightmem_search(client, query, top_k):
    start = time()
    results = client.search(query, top_k)
    memory = [f"{memory['created_at']}: {memory['memory']}" for memory in results] # TODO
    # context = MEM0_CONTEXT_TEMPLATE.format(user_id=user_id, memories=memory)
    duration_ms = (time() - start) * 1000
    return memory, duration_ms


def memos_search(client, query, user_id, top_k):
    start = time()
    results = client.search(query=query, user_id=user_id, top_k=top_k)
    print(results)
    context = (
        "\n".join([i["memory"] for i in results["text_mem"][0]["memories"]])
        + f"\n{results.get('pref_string', '')}"
    )
    context = MEMOS_CONTEXT_TEMPLATE.format(user_id=user_id, memories=context)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def memobase_search(client, query, user_id, top_k):
    start = time()
    try:
        context = client.search(query=query, user_id=user_id, top_k=top_k)
        duration_ms = (time() - start) * 1000
    except:
        context = "ERROR"
        duration_ms = 0.0
    return context, duration_ms

def supermemory_search(client, query, user_id, top_k):
    start = time()
    context = client.search(query, user_id, top_k)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def process_user(question, frame, user_id, top_k=20, key_map_super=None):

    print(f"❓ Question: {question}")
    print("-" * 80)
    search_results = defaultdict(list)


    if "mem0" in frame:
        from utils.client import Mem0Client

        client = Mem0Client(enable_graph="graph" in frame)
        context, duration_ms = mem0_search(client, question, user_id, top_k)
    elif frame == "memobase":
        from utils.client import MemobaseClient

        client = MemobaseClient()
        context, duration_ms = memobase_search(client, question, user_id, top_k)
    elif frame == "memos-api":
        from utils.client import MemosApiClient

        client = MemosApiClient()
        context, duration_ms = memos_search(client, question, user_id, top_k)
    elif frame == "memos-api-online":
        from utils.client import MemosApiOnlineClient

        client = MemosApiOnlineClient()
        context, duration_ms = memos_search(client, question, user_id, top_k)
    elif frame == "memu":
        from utils.client import MemuClient

        client = MemuClient()
        context, duration_ms = memu_search(client, question, user_id, top_k)
    elif frame == "supermemory":
        from utils.client import SupermemoryClient

        client = SupermemoryClient(key_map_super)
        context, duration_ms = supermemory_search(client, question, user_id, top_k)
    
    elif frame == "lightmem":
        from utils.client import LightMemoryClient

        client = LightMemoryClient(user_id)
        context, duration_ms = lightmem_search(client, question, top_k)

    search_results[user_id].append(
        {
            "question": question,
            "search_context": context,
            "search_duration_ms": duration_ms,
        }
    )

    return search_results