import os
import json
import numpy as np
import argparse
import random
import re
import logging
import tiktoken
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from function.ingestion import ingest_session
from function.search import process_user
from function.client import *
from openai import OpenAI
from prompt import (
    USER_AGENT_QUERY_PROMPT,
    USER_AGENT_FEEDBACK_PROMPT_TASK1,
    USER_AGENT_FEEDBACK_PROMPT_TASK2,
    ANSWER_PROMPT,
    EVAL_DIALOGUE_TASK_COMPLETION,
    EVAL_DIALOGUE_MEMORY,
    ANSWER_OPTIONAL_PROMPT
)
from util import get_client, save_json, _encode_dialog, ensure_list, _cosine_similarity, parse_date_with_period, iso_or_default
from bert_score import score as _bert_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COUNTRY = {
    "Canada": [334],
    "Mexico": [354],
    "Finland": [123],
    "United States": [1377],
    "Australia": [507],
    "United Kingdom": [914],
    "Switzerland": [112],
    "Israel": [419],
    "Russian Federation": [108],
    "Belgium": [109]
}

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
token_encoding = tiktoken.get_encoding("cl100k_base")
random.seed(42)
count_country = COUNTRY
USER_IDS = []
for country, indices in count_country.items():
    USER_IDS.extend(indices)


def _get_task_limit(args) -> Optional[int]:
    limit = getattr(args, "max_tasks", None)
    if limit is None and getattr(args, "smoke_test", False):
        limit = 5
    if limit is None:
        return None
    return max(0, int(limit))


def _get_eval_user_ids(args) -> List[int]:
    max_users = getattr(args, "max_users", None)
    if max_users is None and getattr(args, "smoke_test", False):
        max_users = 1
    if max_users is not None:
        selected = USER_IDS[: max(0, int(max_users))]
    return selected

try:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    try:
        load_dotenv()
    except Exception:
        pass

oai_client = OpenAI(api_key=os.getenv("CHAT_MODEL_API_KEY"), base_url=os.getenv("CHAT_MODEL_BASE_URL"))

def _load_profile(uid: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ppath = os.path.join(DATA_ROOT, "profile", f"user{uid}", "profile.json")
    try:
        with open(ppath, "r") as f:
            prof = json.load(f)
        return prof.get("demographics", {}), prof.get("affinities", {}), prof.get("interactions", {})
    except Exception:
        return {}, {}, {}

def _get_embedding_model() -> SentenceTransformer:
    global _EMB_MODEL
    try:
        if _EMB_MODEL is not None:
            return _EMB_MODEL
    except NameError:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _EMB_MODEL = SentenceTransformer("BAAI/bge-m3", device=device)
    return _EMB_MODEL

import tiktoken

def _metric_bert_f1(ref: str, hyp: str) -> float:
    if _bert_score is None:
        return 0.0
    try:
        _, _, f1 = _bert_score([ref], [hyp], lang="en", rescale_with_baseline=False, verbose=False)
        return f1.item() if f1 is not None else 0.0
    except Exception:
        return 0.0
# --------------------------Interaction----------------------------------
class UserLLMStub:
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.chat_model = "gpt-4o"

    def _complete(self, user: str, temperature: float = 0.5) -> str:
        for _ in range(5):
            resp = oai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
            result =  resp.choices[0].message.content
            if result:
                return result
            else:
                time.sleep(1)
                continue
        return "TERMINATE"

    def feedback_task(
        self,
        history: str,
        demographic_profile: str,
        preference_str: str,
        task_goal: str,
        task_description: str,
        user_use_topic_dialog: str,
        type_task, # 1 / 2
        task_question: str,
    ) -> str:
        if type_task == 1:
            user_prompt = USER_AGENT_FEEDBACK_PROMPT_TASK1.format(
                demographic_profile=demographic_profile,
                preference_str=preference_str,
                task_goal=task_goal,
                task_description=task_description,
                message_history=history,
                task_question=task_question,
            )
        else:
            user_prompt = USER_AGENT_FEEDBACK_PROMPT_TASK2.format(
                demographic_profile=demographic_profile,
                preference_str=preference_str,
                task_goal=task_goal,
                user_use_topic_dialog=user_use_topic_dialog,
                task_description=task_description,
                message_history=history,
                task_question=task_question,
            )
        content = self._complete(user_prompt, temperature=0.3)
        return content

def interact(
    user_llm,
    question: str,
    max_turns: int = 1,
    search_context: Optional[str] = None,
    type_task: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Interact using the existing memory interfaces:
    - Retrieval: process_user(question, frame, version, top_k)
    - Response: lme_response(openai_client, context, question, question_date)
    - Turns: controlled by max_turns (max_turns=1 indicates single-turn interaction).

    For multi-turn settings, a lightweight follow-up question is generated based on the previous answer. The first-turn response can also be used for alignment with single-turn evaluation.
    """
    turns = 0
    history_assistant = ""
    history_user = ""
    last = {"question": question, "history": ""}
    current_q = question
    context = search_context if type_task != 1 else ""

    for _ in range(max(1, max_turns)):
        turns += 1
        history_assistant += "User: " + current_q
        history_user += "You: " + current_q
        answer = response(oai_client, context, current_q, history_assistant)
        history_assistant += "\nYou: " + answer + "\n"
        history_user += "\nAssistant: " + answer + "\n"
        feedback_task = user_llm.feedback_task(
            history=history_user,
            demographic_profile=kwargs["demographic_profile"],
            preference_str=kwargs["preference_str"],
            task_description=kwargs["task_description"],
            task_goal=kwargs["task_goal"],
            task_question=question,
            user_use_topic_dialog=kwargs["user_use_topic_dialog"],
            type_task=type_task
        )
        current_q = feedback_task
        if "TERMINATE" in feedback_task:
            break
    last["history"] = history_assistant.replace("You: ", "Assistant: ")
    last["turns"] = turns
    return last


def response(llm_client, context: str, question: str, history: str) -> str:
    prompt = ANSWER_PROMPT.format(
        question=question,
        history=history,
        context=context,
    )
    for _ in range(5):
        response = llm_client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        result = response.choices[0].message.content
        if result:
            return result
        else:
            time.sleep(1)
            continue
    return "TERMINATE"


def _single_eval(prompt_text: str, model: str) -> str:
    for _ in range(10):
        try:
            resp = oai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.1,
                # extra_body={"search_disable": False, "reasoning_effort": "low"}, # TODO for reasoning model
            )
            result = resp.choices[0].message.content
            if result:
                return result
            else:
                time.sleep(1)
                continue
        except Exception as e:
            time.sleep(2)
            continue
    return ""

def evaluate(
    args,
    mode: str,
    max_turns: int = 10,
    stage: Optional[List[str]] = None,
):
    def _load_meta(out_root: str, scope: str, task_id: str) -> Dict[str, Any]:
        path = os.path.join(out_root, "meta", scope, f"{task_id}.json")
        return json.load(open(path, "r", encoding="utf-8"))

    def _flatten_context(dialogs) -> str:
        result = []
        for dialog in dialogs:
            result.extend(dialog[0])
        return result

    if "rag" in mode:
        emb_model = _get_embedding_model()

    task_limit = _get_task_limit(args)
    selected_user_ids = _get_eval_user_ids(args)
    logger.info(
        f"Evaluation users: {selected_user_ids}, task_limit_per_scope={task_limit}, smoke_test={getattr(args, 'smoke_test', False)}"
    )
    version = "_multi" if args.multi_domain else ""
    file_name = "_c" if args.no_noise else "_n"
    style_name = "_s" if args.style else ""
    top_k_name = f"_top{args.top_k}" if args.top_k == 20 else ""
    user_llm = UserLLMStub()
    model_name = "gpt-4o-mini" if args.mode == "longcontext" else "gpt-4o-mini" # TODO change model name (longcontext reasoning model)
    for st in stage:
        logger.info(f"Start stage: {st}")
        for uid in selected_user_ids:
            in_path = os.path.join(DATA_ROOT, "tasks", f"user{uid}", f"input_data{version}{file_name}{style_name}.json")
            input_data = json.load(open(in_path, "r", encoding="utf-8"))

            user_profile, _, _ = _load_profile(uid)
            out_root = os.path.join(args.output_dir, f"user{uid}")
            os.makedirs(out_root, exist_ok=True)

            def _tasks(scope_key: str) -> List[Dict[str, Any]]:
                items = ensure_list(input_data.get(scope_key))
                if args.mem_frame == "lightmem":
                    items = sorted(items, key=lambda ev: 1 if int(ev.get("type", "")) == 3 else 0)
                if task_limit is not None:
                    items = items[:task_limit]
                return items
            # Stage ADD
            if st == "add":
                for scope in (["overall"] if args.run_overall_eval else []):
                    flag = False
                    for idx, ev in tqdm(enumerate(_tasks(scope)), desc=f"Processing {scope} tasks"): # enumerate
                        task_id = ev.get("task_id", "")
                        dialogs = ev.get("context", [])
                        task_type = ev.get("type", "")
                        all_preference = ev.get("preferences", [])
                        query = _load_meta(out_root, scope, f"{task_id}_{task_type}")["question"]
                        qdate = _load_meta(out_root, scope, f"{task_id}_{task_type}")["question_date"]

                        if "baseline" in mode:
                            if int(task_type) == 3:
                                if flag:
                                    if args.mem_frame == "lightmem":
                                        start_time = time.time()
                                        search_results = client.search(query, top_k=args.top_k)
                                        end_time = time.time()
                                        duration_ms = (end_time - start_time) * 1000

                                        save_json(os.path.join(out_root, "baseline" + "_" + args.mem_frame, scope + version + file_name + style_name, "search" + top_k_name, f"{task_id}_{task_type}.json"), {
                                            "task_id": task_id,
                                            "question": query,
                                            "question_date": qdate,
                                            "search_context": search_results,
                                            "search_duration_ms": duration_ms,
                                        })
                                    continue
                                else:
                                    event_id = "ALL"
                                    flag = True
                            else:
                                event_id = idx
                            ing_user_id = f"user_{uid}_{event_id}_{scope}{version}{file_name}{style_name}"
                            client = get_client(args.mem_frame, ing_user_id)
                            for s_idx, session in enumerate(dialogs):
                                date_dt = parse_date_with_period(session[1])
                                session_id = ing_user_id + "_session_" + str(s_idx)
                                try:
                                    ingest_session(session[0], date_dt, ing_user_id, session_id, args.mem_frame, client)
                                except Exception as e:
                                    logger.warning(f"User {uid} task {task_id} fails: {e}")

                            if args.mem_frame == "lightmem":
                                start_time = time.time()
                                search_results = client.search(query, top_k=args.top_k)
                                end_time = time.time()
                                duration_ms = (end_time - start_time) * 1000
                                save_json(os.path.join(out_root, "baseline" + "_" + args.mem_frame, scope + version + file_name + style_name, "search" + top_k_name, f"{task_id}_{task_type}.json"), {
                                    "task_id": task_id,
                                    "question": query,
                                    "question_date": qdate,
                                    "search_context": search_results,
                                    "search_duration_ms": duration_ms,
                                })


                        elif "rag" in mode:
                            emb_dir = os.path.join(out_root, "embeddings", f"{scope}{version}{file_name}{style_name}")
                            os.makedirs(emb_dir, exist_ok=True)
                            emb_path = os.path.join(emb_dir, f"{task_id}_{task_type}.npy")
                            dialogs = _flatten_context(dialogs)
                            if not os.path.exists(emb_path):
                                try:
                                    embeddings = []
                                    for s_idx in range(0, len(dialogs), args.batch_size):
                                        session = dialogs[s_idx:s_idx+args.batch_size]
                                        embeddings.append(_encode_dialog(session, emb_model))
                                    embeddings = np.asarray(embeddings)
                                    np.save(emb_path, embeddings)
                                except Exception as e:
                                    logger.warning(f"User {uid} task {task_id} failed to generate embeddings: {e}")

                        elif "longcontext" in mode:
                            ctx = _flatten_context(dialogs)
                            ctx = "\n".join([f"{m['role']}: {m['content']}" for m in ctx])
                            save_json(os.path.join(out_root, "longcontext", scope + version + file_name + style_name, "context", f"{task_id}_{task_type}.json"), {
                                "task_id": task_id,
                                "search_context": ctx,
                            })

            # Stage SEARCH
            elif st == "search":
                for scope in (["overall"] if args.run_overall_eval else []):
                    for idx, ev in tqdm(enumerate(_tasks(scope)), desc=f"address {scope} task"): # enumerate
                        task_id = ev.get("task_id", "")
                        task_type = ev.get("type", "")
                        meta = _load_meta(out_root, scope, f"{task_id}_{task_type}")
                        question = meta.get("question", "")
                        question_date = meta.get("question_date", iso_or_default(None))

                        if "baseline" in mode:
                            event_id = "ALL" if int(task_type) == 3 else idx
                            user_key = f"user_{uid}_{event_id}_{scope}{version}{file_name}{style_name}"
                            search_results = process_user(question, args.mem_frame, user_key, args.top_k)
                            sr_list = search_results.get(user_key, [])
                            context = sr_list[0].get("search_context", "") if sr_list else ""
                            duration_ms = sr_list[0].get("search_duration_ms", 0.0) if sr_list else 0.0
                            save_json(os.path.join(out_root, "baseline" + "_" + args.mem_frame, scope + version + file_name + style_name, "search" + top_k_name, f"{task_id}_{task_type}.json"), {
                                "task_id": task_id,
                                "question": question,
                                "question_date": question_date,
                                "search_context": context,
                                "search_duration_ms": duration_ms,
                            })

                        elif "rag" in mode:
                            emb_path = os.path.join(out_root, "embeddings", f"{scope}{version}{file_name}{style_name}", f"{task_id}_{task_type}.npy")
                            try:
                                embeddings = np.load(emb_path)
                            except Exception:
                                logger.warning(f"User {uid} task {task_id} lack embeddings, skip")
                                continue
                            start_time = time.time()
                            q_vec = emb_model.encode([question], normalize_embeddings=True)[0]
                            sims = [(_cosine_similarity(q_vec, embeddings[i]), i) for i in range(len(embeddings))]
                            sims.sort(reverse=True)
                            selected = [idx for _, idx in sims[: max(1, args.top_k)]]
                            
                            dialogs = ev.get("context", [])
                            dialogs = _flatten_context(dialogs)
                            dialogs_batch = [dialogs[i:i+args.batch_size] for i in range(0, len(dialogs), args.batch_size)]
                            ctx_parts = []
                            for idx2 in selected:
                                if idx2 < len(dialogs_batch):
                                    d = dialogs_batch[idx2]
                                    ctx_parts.append("\n".join([f"{m['role']}: {m['content']}" for m in d]))
                            
                            context = "\n\n".join(ctx_parts)
                            duration_ms = (time.time() - start_time) * 1000
                            logger.info(f"User {uid} task {task_id} searches {args.top_k} chunks, time= {duration_ms:.2f} ms")
                            save_json(os.path.join(out_root, "rag", scope + version + file_name + style_name, "search" + top_k_name, f"{task_id}_{task_type}.json"), {
                                "task_id": task_id,
                                "question": question,
                                "question_date": question_date,
                                "top_k": args.top_k,
                                "selected_indices": selected,
                                "search_context": context,
                                "search_duration_ms": duration_ms,
                            })

                        elif "longcontext" in mode:
                            ctx_path = os.path.join(out_root, "longcontext", scope + version + file_name + style_name, "context", f"{task_id}_{task_type}.json")
                            try:
                                ctx_obj = json.load(open(ctx_path, "r", encoding="utf-8"))
                                context = ctx_obj.get("search_context", "")
                            except Exception:
                                context = _flatten_context(ev.get("context", []))
                                context = "\n".join([f"{m['role']}: {m['content']}" for m in context])
                            save_json(os.path.join(out_root, "longcontext", scope + version + file_name + style_name, "search", f"{task_id}_{task_type}.json"), {
                                "task_id": task_id,
                                "question": question,
                                "question_date": question_date,
                                "search_context": context,
                                "search_duration_ms": 0.0,
                            })

            # Stage ANSWER
            elif st == "answer":
                for scope in (["overall"] if args.run_overall_eval else []):
                    for ev in _tasks(scope):
                        logger.info(f"User {uid} task {ev.get('task_id', '')} type {ev.get('type', '')}")
                        task_id = ev.get("task_id", "")
                        task_type = ev.get("type", "")
                        meta = _load_meta(out_root, scope, f"{task_id}_{task_type}")
                        question = meta.get("question", "")
                        question_date = meta.get("question_date", iso_or_default(None))
                        type_task = ev.get("type", 1)

                        task_meta = ev.get("task", {})
                        task_desc = task_meta.get("description", "")
                        task_goal = task_meta.get("task_goal", "")
                        user_use_topic_dialog = ev.get("user_use_topic_dialog", "")
                        topic = ev.get("topic", [])
                        affinity_types = task_meta.get("Relevant Affinity Types", [])
                        
                        all_preference = ev.get("preferences", [])
                        user_preference_history = []
                        for pref in all_preference:
                            for t in affinity_types:
                                if t in pref:
                                    user_preference_history.append(pref)
                                    # ----------------------------------
                        task_preference = "\n".join(user_preference_history)
                        aff_links = ev.get("affinity_links", [])
                        aff_dialogs = []
                        for idx_link, link in enumerate(aff_links):
                            pref_dialog = link.get("value")
                            date_link = link.get("date")
                            aff_dialogs.append(f"Conversation {idx_link + 1} ({date_link}):\n - Conversation" + "\n".join([f"{item['role']}: {item['content']}" for item in pref_dialog[:-1]]))
                        
                        user_use_topic_dialog = "\n\n".join(aff_dialogs)
                        kwargs = {
                            "demographic_profile":str([f"{k}: {v}" for k, v in user_profile.items()]),
                            "preference_str": task_preference,
                            "task_description": task_desc,
                            "task_goal": task_goal,
                            "topic": topic,
                            "relevant_domains": ", ".join(topic if isinstance(topic, list) else [topic]),
                            "user_use_topic_dialog": user_use_topic_dialog,
                        }
                        
                        meta_info = _load_meta(out_root, scope, f"{task_id}_{task_type}")
                        option_resp = meta_info["options"]

                        frame = ("_" + args.mem_frame) if "baseline" in mode else ""
                        spath = os.path.join(out_root, mode + frame, scope + version + file_name + style_name, "search" + model_name.split("/")[-1] if mode == "longcontext" else "search" + top_k_name, f"{task_id}_{task_type}.json")

                        try:
                            sobj = json.load(open(spath, "r", encoding="utf-8"))
                            context = sobj.get("search_context", "")
                        except Exception:
                            logger.warning(f"User {uid} task {task_id} lack search output, skip answer stage")
                            continue
                        if type_task == 1:
                            context = "No memory found."

                        if args.interactive and mode != "longcontext":
                            try:
                                inter_res = interact(
                                    user_llm=user_llm,
                                    question=question,
                                    max_turns=max_turns,
                                    search_context=context,
                                    type_task=type_task,
                                    **kwargs,
                                )
                            except Exception:
                                logger.error(f"User {uid} task {task_id} evaluation failed, skip")
                                inter_res = {
                                    "history": "",
                                    "turns": 0,
                                }
                        else:
                            inter_res = {
                                "history": "",
                                "turns": 0,
                            }

                        answer_opention_prompt = ANSWER_OPTIONAL_PROMPT.format(
                            context=context,
                            question=question,
                            options=option_resp,
                        )
                        try:
                            answer_option = _single_eval(answer_opention_prompt, model_name)
                            if meta_info["gold_label"] == answer_option:
                                answer_option_score = 1
                            else:
                                answer_option_score = 0
                        except Exception:
                            logger.error(f"User {uid} task {task_id} evaluation failed, skip")
                            answer_option = ""
                            answer_option_score = 0
                        save_json(os.path.join(out_root, mode + frame, scope + version + file_name + style_name, "answer" + model_name.split("/")[-1] if mode == "longcontext" else "answer" + top_k_name, f"{task_id}_{task_type}.json"), {
                            "task_id": task_id,
                            "question": question,
                            "history": inter_res.get("history", ""),
                            "turns": inter_res.get("turns", 1),
                            "search_context": sobj.get("search_context", ""),
                            "search_duration_ms": sobj.get("search_duration_ms", 0.0),
                            "task_preference": task_preference,
                            "answer_option_score": answer_option_score,
                            "answer_option": answer_option,
                            "type_task": type_task,
                        })

            elif st == "eval":
                for scope in (["overall"] if args.run_overall_eval else []):
                    for ev in _tasks(scope):
                        task_id = ev.get("task_id", "")
                        task_meta = ev.get("task", {})
                        task_desc = task_meta.get("description", "")
                        task_goal = task_meta.get("task_goal", "")
                        task_type = ev.get("type", 1)

                        affinity_types = task_meta.get("Relevant Affinity Types", [])
                        topic = ev.get("topic", [])
                        user_use_topic_dialog = ev.get("user_use_topic_dialog", "")

                        frame = ("_" + args.mem_frame) if "baseline" in mode else ""
                        apath = os.path.join(out_root, mode + frame, scope + version + file_name + style_name, "answer" + model_name.split("/")[-1] if mode == "longcontext" else "answer" + top_k_name, f"{task_id}_{task_type}.json")
                        try:
                            aobj = json.load(open(apath, "r", encoding="utf-8"))
                        except Exception:
                            logger.warning(f"User {uid} task {task_id} answer output missing, skipping evaluation")
                            continue

                        meta_info = _load_meta(out_root, scope, f"{task_id}_{task_type}")


                        search_duration_ms = aobj.get("search_duration_ms", 0.0)
                        conversation = (aobj.get("history", "") or "").strip()
                        question = aobj.get("question", task_meta.get("question", ""))
                        situation_context = aobj.get('search_context', '')
                        type_task = aobj.get("type_task", "")

                        task_preference = aobj.get("task_preference", "")

                        task_completion_prompt = EVAL_DIALOGUE_TASK_COMPLETION.format(conversation=conversation, goal=task_goal, question=question)


                        def _extract_score(text: str, label: str) -> int:
                            m = re.search(rf"{label}\s*:\s*(\d+)", text)
                            return int(m.group(1)) if m else -1
                        def _extract_verdicts(text: str):
                            try:
                                verdict_pattern = r"VERDICT: (True|False)"
                                verdict_match = re.search(verdict_pattern, text, re.IGNORECASE)
                                verdict = verdict_match.group(1).lower() == "true" if verdict_match else None
                                return verdict
                            except Exception:
                                return None
                        def _extract_expl(text: str) -> str:
                            try:    
                                m = re.search(r"EXPLANATION\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
                                return (m.group(1).strip() if m else "").splitlines()[0]
                            except Exception:
                                return ""

                        if "longcontext" in mode or not args.interactive:
                            task_completion = ""
                        else:
                            task_completion = _single_eval(task_completion_prompt, "gpt-4o")
                            verdict = _extract_verdicts(task_completion)
                            explanation = _extract_expl(task_completion)

                        if type_task != 1 and type_task != 0 and "longcontext" not in mode:
                            aff_links = ev.get("affinity_links", [])
                            aff_dialogs = []
                            for idx_link, link in enumerate(aff_links):
                                pref_dialog = link.get("value")
                                date_link = link.get("date")
                                aff_dialogs.append(f"Evnet {idx_link + 1} ({date_link}):\n - Conversation" + "\n".join([f"{item['role']}: {item['content']}" for item in pref_dialog[:-1]]))
                            
                            user_use_topic_dialog = "\n\n".join(aff_dialogs)

                            memory_prompt = EVAL_DIALOGUE_MEMORY.format(
                                preferences_to_be_mastered=task_preference,
                                assistant_retrieved_memory=situation_context,
                                conversation=user_use_topic_dialog,
                                query=question
                            )
                            memory = _single_eval(memory_prompt, "gpt-4o")
                            memory_score = _extract_score(memory, "Memory Score")

                            bert_f1 = _metric_bert_f1(user_use_topic_dialog, situation_context)
                        else:
                            bert_f1 = -1
                            memory_score = -1
                        user_contents = re.findall(r"User:\s*(.*?)(?=\s*(?:Assistant:|User:|$))", conversation, re.DOTALL)
                        result = "".join(user_contents).strip()
                        len_user_token = len(token_encoding.encode(result))

                        pers_score = 0

                        tokens = len(token_encoding.encode(situation_context))
                        turns = aobj.get("turns", 0)
                        search_duration_ms = float(search_duration_ms)
                        
                        save_json(os.path.join(out_root, mode + frame, scope + version + file_name + style_name, "eval" + model_name.split("/")[-1] if mode == "longcontext" else "eval" + top_k_name, f"{task_id}_{type_task}.json"), {
                            "task_id": task_id,
                            "task_type": type_task,
                            "question": question,
                            "goal": task_goal,
                            "task_completion_raw": task_completion,
                            "answer_option_score": meta_info["gold_label"] == aobj.get("answer_option").strip().strip("\n"),
                            "task_completion_verdict": verdict,
                            "task_completion_explanation": explanation,
                            "memory_score": memory_score,
                            "context_tokens": tokens,
                            "turns": turns,
                            "user_use_topic_dialog": user_use_topic_dialog,
                            "search_duration_ms": search_duration_ms,
                            "len_user_token": len_user_token,
                            "bert_f1": bert_f1,
                        })
                        logger.info(f"User {uid} task {task_id} evaluation completed: P={pers_score}, TC={verdict}")

            else:
                logger.warning(f"{st}")
        if st == "eval":
            summarize_eval_metrics(args, scope="overall", model_name=model_name)

def summarize_eval_metrics(args, scope: str = "overall", model_name: str = "") -> Dict[str, Any]:
    """
    Summarizes evaluation metrics across all users and tasks.
    Aggregates metrics globally (for type 3) and per task type.
    """
    version = "_multi" if args.multi_domain else ""
    frame = ("_" + args.mem_frame) if "baseline" in args.mode else ""
    file_name = "_c" if args.no_noise else "_n"
    style_name = "_s" if args.style else ""
    top_k_name = f"_top{args.top_k}" if args.top_k == 20 else ""
    
    # Define metrics to track
    metrics_list = [
        "answer_option_score", "memory_score", "context_tokens", "turns",
        "task_completion_verdict", "search_duration_ms", "len_user_token", "bert_f1"
    ]
    
    # Helper to initialize [sum, count]
    def _metric_init(): return [0.0, 0]
    
    # Global aggregation (only for task_type == 3)
    agg_global = defaultdict(_metric_init)
    
    # Task-specific aggregation: task_id -> task_type (int) -> metric -> [sum, count]
    by_task_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(_metric_init)))
    
    # Helper to update metric stats
    def _update_stat(stats_dict, key, value):
        if value is None: return
        try:
            val_float = float(value)
            stats_dict[key][0] += val_float
            stats_dict[key][1] += 1
        except (ValueError, TypeError):
            pass

    # Helper to update turn buckets (cumulative logic from original)
    def _update_turns(stats_dict, value):
        try:
            v_int = int(value)
            # Counts (denominators)
            stats_dict['turn_other'][1] += 1
            stats_dict['turn_3'][1] += 1
            stats_dict['turn_2'][1] += 1
            stats_dict['turn_1'][1] += 1
            
            # Hits (numerators)
            if v_int == 1:
                stats_dict['turn_1'][0] += 1
            if 1 <= v_int <= 2:
                stats_dict['turn_2'][0] += 1
            if 1 <= v_int <= 3:
                stats_dict['turn_3'][0] += 1
            else:
                stats_dict['turn_other'][0] += 1
        except (ValueError, TypeError):
            pass

    for uid in USER_IDS:
        # Construct path
        eval_subpath = "eval" + model_name.replace("/", "-") if args.mode == "longcontext" else "eval" + top_k_name
        eval_dir = os.path.join(args.output_dir, f"user{uid}", args.mode + frame, scope + version + file_name + style_name, eval_subpath)
        
        if not os.path.isdir(eval_dir):
            continue
            
        for fname in os.listdir(eval_dir):
            if not fname.endswith(".json"): continue
            
            fpath = os.path.join(eval_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
            except Exception:
                continue

            task_id = obj.get("task_id", "")
            try:
                t_type = int(obj.get("task_type", -1))
            except:
                t_type = -1

            # Validate filename matches content
            if fname.strip(".json") != f"{task_id}_{t_type}":
                continue

            # --- Global Aggregation (Type 3) ---
            if t_type == 3:
                for m in metrics_list:
                    val = obj.get(m)
                    if m == "memory_score" and val is not None and float(val) < 0:
                        continue
                    
                    if m == "turns" and args.mode != "longcontext":
                        _update_turns(agg_global, val)
                    
                    _update_stat(agg_global, m, val)
                
                # Verdict (bool -> float)
                verdict = obj.get("task_completion_verdict")
                if isinstance(verdict, bool):
                    _update_stat(agg_global, "task_completion_verdict_rate", 1.0 if verdict else 0.0)

            # --- Per-Task Aggregation ---
            if task_id:
                task_stats = by_task_agg[task_id][t_type]
                for m in metrics_list:
                    val = obj.get(m)
                    if m == "memory_score" and val is not None and float(val) < 0:
                        continue
                    
                    if m == "turns" and args.mode != "longcontext":
                        _update_turns(task_stats, val)
                    
                    _update_stat(task_stats, m, val)
                
                verdict = obj.get("task_completion_verdict")
                if isinstance(verdict, bool):
                     _update_stat(task_stats, "verdict", 1.0 if verdict else 0.0)

    # --- Compute Means ---
    def _compute_means(agg_dict):
        means = {}
        for k, (s, c) in agg_dict.items():
            means[k] = (s / c) if c > 0 else 0.0
        return means

    global_means = _compute_means(agg_global)
    
    # Task-level means
    by_task_means = {}
    for tid, t_types in by_task_agg.items():
        by_task_means[tid] = {}
        for t_type, metrics in t_types.items():
            by_task_means[tid][str(t_type)] = _compute_means(metrics)

    # Category means (Common IDs present in 1, 2, and 3)
    common_ids = [tid for tid, types in by_task_means.items() if all(k in types for k in ["1", "2", "3"])]
    
    by_type_category_means = {"1": {}, "2": {}, "3": {}}
    for t_str in ["1", "2", "3"]:
        acc = defaultdict(float)
        cnt = defaultdict(int)
        for tid in common_ids:
            metrics = by_task_means[tid].get(t_str, {})
            for m, v in metrics.items():
                acc[m] += v
                cnt[m] += 1
        
        for m, s in acc.items():
            c = cnt[m]
            by_type_category_means[t_str][m] = (s / c) if c > 0 else 0.0

    summary = {
        "task3_means": global_means,
        "by_type_category_means": by_type_category_means,
    }
    logger.info(f"Summary: {summary}")
    return summary


def run_incremental_eval(args):
    mode = args.mode
    user_llm = UserLLMStub()
    version = "_multi" if args.multi_domain else ""
    file_name = "_c" if args.no_noise else "_n"
    style_name = "_s" if args.style else ""

    def _load_meta(out_root: str, scope: str, task_id: str) -> Dict[str, Any]:
        path = os.path.join(out_root, "meta", scope, f"{task_id}.json")
        return json.load(open(path, "r", encoding="utf-8"))

    if "rag" in mode:
        emb_model = _get_embedding_model()

    logger.info("Starting INCREMENTAL evaluation...")
    task_limit = _get_task_limit(args)
    selected_user_ids = _get_eval_user_ids(args)
    logger.info(
        f"Incremental users: {selected_user_ids}, task_limit={task_limit}, smoke_test={getattr(args, 'smoke_test', False)}"
    )
    
    def process_incremental_user_unified(uid, dataset_type="standard", eval_stage="search", eval_model=""):
        # Common setup
        out_root = os.path.join(args.output_dir, "incremental", f"user{uid}")
        user_profile, _, _ = _load_profile(uid)
        
        # 1. Data Loading Strategy
        all_sessions = []
        all_eval_items = []
        scope = "overall"
        start_percent = 10
        
        if dataset_type == "standard":
            # Standard incremental data loading
            in_path = os.path.join(DATA_ROOT, f"user{uid}", f"input_data2{version}{file_name}{style_name}.json")
            if not os.path.exists(in_path): return
            input_data = json.load(open(in_path, "r", encoding="utf-8"))
            
            # Load dialogue and timeline
            if args.style:
                dialogue_path = f"{DATA_ROOT}/tasks/user{uid}/raw_dialogues_s.json"
            else:
                dialogue_path = f"{DATA_ROOT}/tasks/user{uid}/raw_dialogues_c.json" if args.no_noise else f"{DATA_ROOT}/tasks/user{uid}/raw_dialogues_n.json"
            if not dialogue_path or not os.path.exists(dialogue_path):
                logger.warning(f"Dialogue file missing for user {uid}, skip")
                return
                
            with open(dialogue_path, 'r', encoding='utf-8') as f:
                single_checkpoint_dataset = json.load(f)
                all_dialogues = {item['topic']: {"dialogs": item['dialogs']} for item in single_checkpoint_dataset}
                
            task_id_to_dialogue = {}
            for topic, item in all_dialogues.items():
                for dialog in item['dialogs']:
                    task_id_to_dialogue[dialog['task_id']] = (dialog['conversation'], dialog['preferences'], dialog['selected_noise'], dialog['date'], dialog['event_type'])
                    
            timeline_path = os.path.join(DATA_ROOT, "tasks", f"user{uid}", "interleaved_timeline.json")
            with open(timeline_path, 'r', encoding='utf-8') as f:
                interleaved_timeline = json.load(f)
                
            for item in interleaved_timeline:
                task_id_tmp = item['task_id']
                conversation, preferences, selected_noise, _, event_type = task_id_to_dialogue[task_id_tmp]
                all_sessions.append({
                    "task_id": task_id_tmp,
                    "conversation": conversation,
                    "preferences": preferences,
                    "selected_noise": selected_noise,
                    "date": item['date'],
                    "event_type": event_type,
                })
                
            all_eval_items.extend(ensure_list(input_data.get("overall", [])))
            scope = "overall"
            start_percent = 10

        elif dataset_type in ["long", "long_multi"]:
            # Long context data loading
            is_multi = (dataset_type == "long_multi")
            in_path = os.path.join(DATA_ROOT, "tasks", f"user{uid}", f"input_data_s_long.json")
            if not os.path.exists(in_path): return
            input_data = json.load(open(in_path, "r", encoding="utf-8"))
            
            all_eval_items.extend(ensure_list(input_data.get("overall", [])))
            type3_indices = [i for i, item in enumerate(input_data.get("overall")) 
                            if isinstance(item, dict) and item.get("type") == 3 and isinstance(item.get("context"), list)]
            
            if type3_indices:
                target_idx = type3_indices[0]
                all_sessions = all_eval_items[target_idx].get("context")
            
            if is_multi:
                in_path_multi = os.path.join(DATA_ROOT, "tasks", f"user{uid}", f"input_data_multi_s.json")
                input_data = json.load(open(in_path_multi, "r", encoding="utf-8"))
                all_eval_items = []
                all_eval_items.extend(ensure_list(input_data.get("overall", [])))
                scope = "overall_s_long_multi"
            else:
                scope = "overall_s_long"
            start_percent = 100
        
        # 2. Task Filtering Logic
        task_groups = {}
        for ev in all_eval_items:
            tid = ev.get('task_id')
            if not tid: continue
            if tid not in task_groups: task_groups[tid] = set()
            task_groups[tid].add(int(ev.get('type', 0)))

        valid_task_ids = {tid for tid, types in task_groups.items() if {1, 2, 3}.issubset(types)}
        
        selected_events = []
        final_test_event = []
        seen_tasks = set()
        
        for ev in all_eval_items:
            tid = ev.get('task_id')
            t_type = int(ev.get('type', 0))
            if t_type == 3:
                if dataset_type == "long_multi":
                     final_test_event.append(ev)
                else:
                    if tid in valid_task_ids:
                        if tid not in seen_tasks:
                            selected_events.append(ev)
                            seen_tasks.add(tid)
                    else:
                        if dataset_type != "standard":
                            final_test_event.append(ev)

        # Log findings
        logger.info(f"User {uid}: Found {len(selected_events)} valid tasks, {len(final_test_event)} final test tasks.")
        if task_limit is not None:
            selected_events = selected_events[:task_limit]
            final_test_event = final_test_event[:task_limit]
        
        # 3. Processing Loop
        mem_user_key = f"user_{uid}_{dataset_type}_{version}" # Simplified key generation
        if dataset_type == "standard":
             mem_user_key = f"user_{uid}_incremental{version}{file_name}{style_name}"
        elif dataset_type in ["long", "long_multi"]:
             mem_user_key = f"user_{uid}_incremental_s_long" if is_multi else f"user_{uid}_incremental_s_long_multi"
        
        client = None
        if mode == "baseline":
            client = get_client(args.mem_frame, mem_user_key)

        total_sessions = len(all_sessions)
        current_idx = 0
        
        # Helper for single eval
        def _single_eval_helper(prompt_text, model):
            for _ in range(5):
                try:
                    resp = oai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.1,
                        # extra_body={"search_disable": False, "reasoning_effort": "low"}, # TODO for reasoning model
                    )
                    result = resp.choices[0].message.content
                    if result:
                        return str(result).strip()
                except Exception:
                    pass
                time.sleep(1)
            return ""

        for percent in tqdm(range(start_percent, 101, 10), desc=f"User {uid} Incremental {eval_stage}"):
            target_idx = int(total_sessions * (percent / 100.0))
            if target_idx <= current_idx: continue
            
            new_batch = all_sessions[current_idx:target_idx]
            
            frame = ("_" + args.mem_frame) if "baseline" in mode else ""
            save_dir_name = f"search_{percent}"
            save_dir = os.path.join(out_root, mode + frame, scope + version + file_name + style_name, save_dir_name)
            # Adjust save_dir for long types
            if dataset_type != "standard":
                save_dir = os.path.join(out_root, mode + frame, scope, save_dir_name)

            if eval_stage == "search":
                logger.info(f"User {uid} [Progress {percent}%]: Ingesting {len(new_batch)} sessions...")
                os.makedirs(save_dir, exist_ok=True)
                
                # --- Ingestion ---
                if mode == "baseline":
                    for i, session in enumerate(new_batch):
                        conv_list = session.get('conversation', []) if isinstance(session, dict) else session[0]
                        date_str = session.get('date', '') if isinstance(session, dict) else session[1]
                        date_dt = parse_date_with_period(date_str)
                        global_sess_idx = current_idx + i
                        session_id = f"{mem_user_key}_session_{global_sess_idx}"
                        try:
                            ingest_session(conv_list, date_dt, mem_user_key, session_id, args.mem_frame, client)
                        except Exception as e:
                            logger.warning(f"Ingest failed: {e}")
                    # Wait for index
                    if args.mem_frame not in ["memobase", "lightmem"]:
                        time.sleep(500) # Reduced wait time
                
                elif mode == "rag":
                    emb_path = os.path.join(save_dir, f"rag.npy")
                    if not os.path.exists(emb_path):
                        dialogs = []
                        for sess in all_sessions[:target_idx]:
                            c_list = sess.get('conversation', []) if isinstance(sess, dict) else sess[0]
                            dialogs.extend(c_list)
                        try:
                             # Batch embedding generation (simplified)
                             embeddings = []
                             for s_idx in range(0, len(dialogs), args.batch_size):
                                 session = dialogs[s_idx:s_idx+args.batch_size]
                                 embeddings.append(_encode_dialog(session, emb_model))
                             if embeddings:
                                 embeddings = np.asarray(embeddings)
                                 np.save(emb_path, embeddings)
                        except Exception as e:
                             logger.warning(f"Embedding failed: {e}")
                
                elif mode == "longcontext":
                    # Save context logic
                    ctx = []
                    for sess in all_sessions[:target_idx]:
                         c_list = sess.get('conversation', []) if isinstance(sess, dict) else sess[0]
                         ctx.extend(c_list)

                    save_json(os.path.join(save_dir, "longcontext.json"), {"context": ctx})

                # --- Search ---
                logger.info(f"User {uid} [Progress {percent}%]: Searching {len(selected_events)} tasks...")
                
                # Pre-load resources for search if needed
                rag_embeddings = None
                rag_dialogs_batch = []
                if mode == "rag":
                    emb_path = os.path.join(save_dir, "rag.npy")
                    if os.path.exists(emb_path):
                        rag_embeddings = np.load(emb_path)
                        # Reconstruct dialogs for retrieval
                        dialogs = []
                        for sess in all_sessions[:target_idx]:
                            c_list = sess.get('conversation', []) if isinstance(sess, dict) else sess[0]
                            dialogs.extend(c_list)
                        rag_dialogs_batch = [dialogs[i:i+args.batch_size] for i in range(0, len(dialogs), args.batch_size)]
                
                full_context_str = ""
                if mode == "longcontext":
                    ctx_list = []
                    for sess in all_sessions[:target_idx]:
                        c_list = sess.get('conversation', []) if isinstance(sess, dict) else sess[0]
                        ctx_list.extend(c_list)
                    full_context_str = "\n".join([f"{m['role']}: {m['content']}" for m in ctx_list])

                for ev in selected_events:
                    task_id = ev.get("task_id")
                    task_type = ev.get("type")
                    
                    meta = _load_meta(os.path.join(args.output_dir, f"user{uid}"), "overall", f"{task_id}_{task_type}")
                    query = meta.get("question", "")
                    qdate = meta.get("question_date", iso_or_default(None))
                    
                    search_results = ""
                    duration_ms = 0.0
                    
                    if mode == "baseline":
                        if args.mem_frame == "lightmem":
                            start_time = time.time()
                            search_results = client.search(query, top_k=args.top_k)
                            duration_ms = (time.time() - start_time) * 1000
                            if not isinstance(search_results, str):
                                search_results = json.dumps(search_results, ensure_ascii=False)
                        else:
                            result_obj = process_user(query, args.mem_frame, mem_user_key, args.top_k)
                            sr_list = result_obj.get(mem_user_key, [])
                            search_results = sr_list[0].get("search_context", "") if sr_list else ""
                            duration_ms = sr_list[0].get("search_duration_ms", 0.0) if sr_list else 0.0

                    elif mode == "rag" and rag_embeddings is not None:
                        try:
                            start_time = time.time()
                            q_vec = emb_model.encode([query], normalize_embeddings=True)[0]
                            sims = [(_cosine_similarity(q_vec, rag_embeddings[i]), i) for i in range(len(rag_embeddings))]
                            sims.sort(reverse=True)
                            selected = [idx for _, idx in sims[: max(1, args.top_k)]]
                            
                            ctx_parts = []
                            for idx2 in selected:
                                if idx2 < len(rag_dialogs_batch):
                                    d = rag_dialogs_batch[idx2]
                                    ctx_parts.append("\n".join([f"{m['role']}: {m['content']}" for m in d]))
                            search_results = "\n\n".join(ctx_parts)
                            duration_ms = (time.time() - start_time) * 1000
                        except Exception as e:
                            logger.warning(f"RAG search failed for {task_id}: {e}")

                    elif mode == "longcontext":
                        search_results = full_context_str
                    
                    # Save search results
                    save_json(os.path.join(save_dir, f"{task_id}_{task_type}.json"), {
                        "task_id": task_id,
                        "task_type": task_type,
                        "context": new_batch,
                        "question": query,
                        "question_date": qdate,
                        "search_context": search_results,
                        "search_duration_ms": duration_ms,
                        "progress_percent": percent
                    })

            elif eval_stage == "answer_eval":
                # --- Answer Evaluation ---
                tasks_to_eval = selected_events
                # For long tasks at 100%, evaluate ALL events (valid + invalid/test)
                if percent == 100 and dataset_type != "standard":
                     tasks_to_eval = selected_events + final_test_event
                
                for ev in tasks_to_eval:
                    task_id = ev.get("task_id")
                    task_type = ev.get("type")
                    spath = os.path.join(save_dir, f"{task_id}_{task_type}.json")
                    if not os.path.exists(spath): continue
                    sobj = json.load(open(spath, "r", encoding="utf-8"))
                    
                    meta = _load_meta(os.path.join(args.output_dir, f"user{uid}"), "overall", f"{task_id}_{task_type}")
                    query = meta.get("question", "")
                    option_resp = meta["options"]
                    gold_label = meta.get("gold_label", None)
                    search_results = sobj.get("search_context", [])
                    
                    # 1. Answer Option (Multiple Choice)
                    # Convert search_results to string if it's a list (for prompt formatting)
                    context_str = search_results
                    if isinstance(search_results, list):
                        context_str = "\n".join([json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item) for item in search_results])
                    elif isinstance(search_results, dict):
                        context_str = json.dumps(search_results, ensure_ascii=False)
                    else:
                        context_str = str(search_results)

                    prompt = ANSWER_OPTIONAL_PROMPT.format(context=context_str, question=query, options=option_resp)
                    answer_option = _single_eval_helper(prompt, eval_model)
                    sobj["answer_option"] = answer_option
                    sobj["answer_option_score"] = 1 if str(gold_label).strip() == str(answer_option).strip() else 0
                    
                    # 2. Complex Interaction & Metrics (only for long tasks or if requested)
                    if dataset_type != "standard":
                         task_meta = ev.get("task", {})
                         task_desc = task_meta.get("description", "")
                         task_goal = task_meta.get("task_goal", "")
                         topic = ev.get("topic", [])
                         
                         affinity_types = task_meta.get("Relevant Affinity Types", [])
                         all_preference = ev.get("preferences", [])
                         user_preference_history = []
                         for pref in all_preference:
                             for t in affinity_types:
                                 if t in pref:
                                     user_preference_history.append(pref)
                         task_preference = "\n".join(user_preference_history)
                         
                         aff_links = ev.get("affinity_links", [])
                         aff_dialogs = []
                         for idx_link, link in enumerate(aff_links):
                             pref_dialog = link.get("value")
                             date_link = link.get("date")
                             aff_dialogs.append(f"Conversation {idx_link + 1} ({date_link}):\n - Conversation" + "\n".join([f"{item['role']}: {item['content']}" for item in pref_dialog[:-1]]))
                         user_use_topic_dialog = "\n\n".join(aff_dialogs)
                         
                         kwargs = {
                            "demographic_profile":str([f"{k}: {v}" for k, v in user_profile.items()]),
                            "preference_str": task_preference,
                            "task_description": task_desc,
                            "task_goal": task_goal,
                            "topic": topic,
                            "relevant_domains": ", ".join(topic if isinstance(topic, list) else [topic]),
                            "user_use_topic_dialog": user_use_topic_dialog,
                         }
                         
                         inter_res = {"history": "", "turns": 0}
                         if mode != "longcontext" and args.interactive:
                            try:
                                inter_res = interact(
                                    user_llm=user_llm,
                                    question=query,
                                    max_turns=10,
                                    search_context=search_results,
                                    type_task=3,
                                    **kwargs,
                                )
                            except Exception as e:
                                logger.error(f"Interaction failed for {task_id}: {e}")
                         
                         sobj["inter_res"] = inter_res
                         
                         # Metrics
                         conversation = inter_res.get('history', "")
                         bert_f1 = 0.0
                         len_user_token = 0
                         if conversation:
                            bert_f1 = _metric_bert_f1(user_use_topic_dialog, context_str)
                            user_contents = re.findall(r"User:\s*(.*?)(?=\s*(?:Assistant:|User:|$))", conversation, re.DOTALL)
                            result_text = "".join(user_contents).strip()
                            len_user_token = len(token_encoding.encode(result_text))

                         tokens = len(token_encoding.encode(context_str))
                         
                         sobj.update({
                            "bert_f1": bert_f1,
                            "len_user_token": len_user_token,
                            "tokens": tokens,
                         })

                    # Save result
                    save_path_model = os.path.join(save_dir, eval_model)
                    os.makedirs(save_path_model, exist_ok=True)
                    save_json(os.path.join(save_path_model, f"{task_id}_{task_type}.json"), sobj)

            current_idx = target_idx
    # Execution Logic
    dataset_type = getattr(args, "dataset_type", "standard")
    stage_plan = []
    stage_set = set(args.stage or [])
    if "add" in stage_set or "search" in stage_set:
        stage_plan.append("search")
    if "answer" in stage_set or "eval" in stage_set:
        stage_plan.append("answer_eval")
    if not stage_plan:
        stage_plan = ["search"]

    for eval_stage in stage_plan:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for uid in selected_user_ids:
                futures.append(executor.submit(process_incremental_user_unified, uid, dataset_type, eval_stage))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"User task failed: {e}")


# ---------- CLI Entry ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline", choices=["rag", "longcontext", "baseline", "incremental"], help="Evaluation mode (no longer distinguishing single/multi)")
    parser.add_argument("--multi_domain", default=True, help="Whether to evaluate multi-domain tasks")
    parser.add_argument("--run_overall_eval", default=True, type=bool, help="Whether to run overall evaluation")
    parser.add_argument("--output_dir", type=str, default=f"{DATA_ROOT}/evaluation", help="Output directory for evaluation results")
    parser.add_argument("--mem_frame", type=str, default="supermemory", choices=[
        "mem0", "memos-api-online", "memobase", "supermemory", "lightmem"
    ], help="Memory system framework selection, passed directly to process_user")
    parser.add_argument("--top_k", type=int, default=10, help="Number of retrieval results") # multi changed to 20
    parser.add_argument("--batch_size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--max_turns", type=int, default=10, help="Maximum turns for multi-turn response (fixed to 1 for single mode)")
    parser.add_argument("--no_noise", default=False, type=bool, help="Whether to exclude noise")
    parser.add_argument("--interactive", default=False, type=bool, help="Whether to enable interactive mode")
    parser.add_argument("--style", default=False, type=bool, help="Whether to exclude style")
    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        default=["add", "search"],
        choices=["add", "search", "answer", "eval"],
        help="Evaluation stages (multiple allowed, executed in order)",
    )
    parser.add_argument("--debug", default=False, type=bool, help="Whether to enable debug mode")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of parallel processing threads")
    parser.add_argument("--dataset_type", type=str, default="standard", choices=["standard", "long", "long_multi"], help="Incremental dataset type")
    parser.add_argument("--smoke_test", action="store_true", help="Enable smoke test defaults (max_users=1, user_ids=5 if not explicitly set)")
    parser.add_argument("--max_users", type=int, default=None, help="Limit number of users to evaluate")
    parser.add_argument("--max_tasks", type=int, default=5, help="Limit number of tasks")

    args = parser.parse_args()

    if "baseline" in args.mode:
        logger.warning(f"Current memory system for evaluation: {args.mem_frame}")
    logger.info(f"Current evaluation mode: {args.mode}")
    logger.info(f"Current evaluation no_noise: {args.no_noise}")
    logger.info(f"Current evaluation style: {args.style}")
    logger.info(f"Current evaluation multi_domain: {args.multi_domain}")
    # ------------stage testing------------
    evaluate(
        args=args,
        mode=args.mode,
        max_turns=max(1, args.max_turns),
        stage=args.stage,
    )

    # summarize_eval_metrics(args, scope="overall")
    # run_incremental_eval(args)


if __name__ == "__main__":
    main()
