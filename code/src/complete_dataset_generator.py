import os
import random
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import openai
import json
from pathlib import Path
import re
import json5 as jsonc
from tqdm import tqdm
import copy
from dotenv import load_dotenv

from prompt import PREFERENCE_EMERGENCE_PROMPT, PREFERENCE_UPDATE_PROMPT, FIRST_TIMELINE_SINGLE_TOPIC
import random
import argparse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHOSEN_NUMBER = 3
MULTI_CHOSEN_NUMBER = 6
# Load .env in code/src by default; externally exported env vars keep higher priority.
load_dotenv(Path(__file__).resolve().parent / ".env")
MODEL_NAME = os.getenv("CHAT_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("CHAT_MODEL_API_KEY", "")
BASE_URL = os.getenv("CHAT_MODEL_BASE_URL", None)

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


noise_examples = {
    0: {
        "type": "No noise",
        "description": "",
        "examples": []
    },
    1: {  # Incomplete information / omitted subject  Specification &Intent
        "type": "Omitted subject / info incomplete",
        "description": "in this conversation, the user's request additionally **contains vague references or omits key information**, requiring the Assistant to actively seek clarification to identify the explicit intent.",
        "examples": [
            "User: Hi, I'm really interested in diving into that game. Could you give me some detailed guidance on the game's open world, quest lines, and graphical settings to enhance immersion?\nAssistant: I can certainly help with detailed guidance. Can you explain in detail which game you're referring to?\nUser: The Witcher 3. I'm particularly interested in quests with deep narratives and tips for optimizing graphics on PC. So could you give me some detailed guidance on the above question?\nAssistant: Absolutely! The Witcher 3: Wild Hunt offers a vast open world filled with diverse landscapes and cities. You can explore areas like Novigrad, Skellige, and Velen, each with its own unique environment...(subsequent conversations until the task is completed)",
        ]
    },
    2: {  # Context switching irrelevant or noisy context.
        "type": "Context switching",
        "description": "in this conversation, the user inserts an request from `User-Assistant Past Interaction`, interrupting the current task flow. The Assistant must process the inserted request. The user may naturally return to the main task after the diversion, and the assistant should adapt to the interruption while ensuring the conversation smoothly continues on the primary objective.",
        "examples": [
            "User: Hi, I'm looking to explore some local food and wine tastings. `By the way, do you remember the hockey game I bought tickets for last time?`\nAssistant: Yes, you purchased tickets for the Toronto Maple Leafs game on November 5th. Now, regarding your request for food and wine tastings, I can certainly help. To tailor the suggestions...\nUser: Thanks for confirming. I want to continue with the culinary events now. Can you recommend some upcoming events and provide details on the atmosphere and menu highlights?\nAssistant: Absolutely! I can help with that. Could you specify whether you're interested in any particular type of cuisine or beverage, and do you prefer events on weekends? ...(subsequent conversations until the task is completed)",
        ]
    },
    3: {  # Inconsistent preference Input &Content
        "type": "Inconsistent preference",
        "description": "in this conversation, the user should briefly express a preference that contradicts their existing User Preferences, creating a temporary inconsistency. After the assistant responds or upon reflection, the user should correct themselves and revert to the original preference. This is treated as a natural form of User Self-Clarification, and should appear realistic and context-driven.", 
        "examples": [
            "User: Hi, I'm looking to attend some professional networking events in my city. Can you help me find some events that are happening soon?\nAssistant: Certainly! Could you let me know which industry you are interested in, and do you have any preferences for the size or type of event?\nUser: Actually, scratch that for a moment. For this week only, I want to see the biggest industry conference... \nAssistant: Understood. Here are a few major tech conferences happening soon: 1. Global Tech Leadership Summit, Nov 18–20, thousands of attendees.\n2. Future Industry Expo, Dec 2–4, exhibition + keynotes ... \nUser: ``Oh, I made a mistake, please ignore the previous request.`` Those might be a bit too big for what I'm hoping for. ``Maybe something a bit more low-key would be better — somewhere people actually have time to talk instead of just rushing between keynote rooms.\nAssistant: Great, thanks for clarifying! Here are a few upcoming tech networking events in your city that cater to smaller groups: \n- Tech Innovators Meetup on Saturday, November 12th, offering a general admission ticket for $75 ...(subsequent conversations until the task is completed)",
        ]
    },
    4: {  # Multi-lingual
        "type": "Multi-lingual",
        "description": "",
        "examples": []
    },
    5: {  # Colloquial / informal expressions
        "type": "Colloquial / informal",
        "description": "in this conversation, the user's input would contain extensive slang, informal vocabulary, or non-standard syntax. The assistant responds in a more standard and formal register, forming a clear style contrast.",
        "examples": [
            "User: Yo, I'm trying to vibe at some professional networking events in my city. Lemme know if there's anything dope coming up ASAP.\nAssistant: Certainly! Could you let me know which industry you are interested in, and do you have any preferences for the size or type of event?\nUser: Subsequent to a critical self-assessment, I must articulate that my focus remains squarely on tech industry convenings, particularly those of a more chill, smaller scale, which are instrumental in forging dope interpersonal networking.\nAssistant: Great, thanks for clarifying! Here are a few upcoming tech networking events in your city that cater to smaller groups: \n- Tech Innovators Meetup on Saturday, November 12th, offering a general admission ticket for $75.\n- Startup Connect on Saturday ...(subsequent conversations until the task is completed)"
        ]
    }
}

class GPTCaller:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.lock = threading.Lock()
        
    def call_gpt(self, prompt: str, max_retries: int = 3, max_tokens: int = 1024, temperature: float = 0.5) -> str:
        """Call GPT API"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e

class CompleteDatasetGenerator:
    def __init__(self, openai_api_key: str = None, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        if not openai_api_key:
            raise ValueError("openai_api_key")
        self.gpt_caller = GPTCaller(
            api_key=openai_api_key,
            model=model,
            base_url=base_url
        )
        self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        self.held_out_tasks: List[Dict[str, Any]] = []
    
    def load_user_profile(self, user_id: str) -> Dict[str, Any]:
        profile_path = self.data_dir / 'profile' / f'user{user_id}' / 'profile.json'
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_user_tasks(self, user_id: str, multi_domain: bool) -> Dict[str, Any]:
        """Load user tasks"""
        form = "_md" if multi_domain else ""
        tasks_path = self.data_dir / 'profile' / f'user{user_id}' / f'tasks{form}.json'
        with open(tasks_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def select_preferred_topics_from_remaining(self, profile: Dict[str, Any], topic_number: int, used_topics: List[str]):
        interests = profile.get('interests', {})
        preferred_topics = [topic for topic, interest in interests.items() if topic not in used_topics and interest == 1]
        
        if len(preferred_topics) < topic_number:
            return preferred_topics, None
        
        chosen = random.sample(preferred_topics, topic_number)

        remaining = [topic for topic, interest in interests.items() if interest == 1 and topic not in chosen]

        random_topic = random.sample(remaining, min(len(remaining), topic_number))
        
        return chosen, random_topic

    def extract_topic_interactions(self, profile: Dict[str, Any], topics: List[str], other_topic: List[str]) -> Dict[str, str]:
        """Extract interaction history for selected topics"""
        interactions = profile.get('interactions', {})
        topic_interactions = {}
        final_topic = topics + other_topic

        final_topic = list(set(final_topic))
        
        for topic in final_topic:
            topic_interactions[topic] = interactions[topic]
        
        return topic_interactions

    def extract_related_tasks(self, tasks: Dict[str, Any], selected_topics: List[str]) -> List[Dict[str, Any]]:
        topic_dict = {item: [] for item in selected_topics}
        for _, task_data in tasks.items():
            task_domains = task_data.get('Relevant Domains', [])
            if task_domains[0] in selected_topics:
                topic_dict[task_domains[0]].append(task_data)
        related_tasks = []
        for _, task_list in topic_dict.items():
            related_tasks.extend(random.sample(task_list, k=CHOSEN_NUMBER))
        return related_tasks
    
    def extract_related_tasks_multi(self, tasks: Dict[str, Any], used_task_ids: Optional[Set[str]] = None):
        related_tasks = []
        multi_relevant_topic = []
        target_used: Set[str] = used_task_ids or set()
        picked = 0
        for _, task_data in tasks.items():
            if picked >= MULTI_CHOSEN_NUMBER:
                break
            tid = task_data.get('task_id') or ''
            if tid and tid in target_used:
                continue
            related_tasks.append(task_data)
            multi_relevant_topic.extend(task_data.get('Relevant Domains', []))
            picked += 1
        multi_relevant_topic = list(set(multi_relevant_topic))
        return related_tasks, multi_relevant_topic


    def _generate_interaction_timeline(self, profile, selected_topics=None) -> Dict[str, Any]:
        if selected_topics is not None:
            selected_topics = [selected_topics]
        else:
            selected_topics = [item for item, topic in profile.get('interests', {}).items()]
        topic_interactions = profile.get('interactions', {})
        demographics = profile.get('demographics', {})
        # Format input data
        if 'user_id' in demographics:
            demographics.pop('user_id')
        demo_str = "\n".join([f"{k}: {v}" for k, v in demographics.items()])
        topics_str = ", ".join(selected_topics)
        other_topic = ", ".join([item for item in topic_interactions.keys() if item not in selected_topics])
        interactions_strs = []
        number = 0
        for topic, interaction in topic_interactions.items():
            if topic not in selected_topics:
                continue
            interaction_str = ""
            number_item = len(interaction.split("\n\n")) - 2
            for idx, item in enumerate(interaction.split("\n\n")):
                if idx == 0:
                    description = f"Domain description: {item}\nDomain Interaction (overall {number_item}):"
                    interaction_str += description
                elif idx == len(interaction.split("\n\n")) - 1:
                    interaction_str += f"\nDomain summary: {item}"
                else:
                    interaction_str += f"\n({idx}) {item}"
            interactions_strs.append(f"{topic}:\n{interaction_str}")
            number += number_item
        
        interactions_strs = "\n\n".join(interactions_strs)
        
        # Timeline
        prompt = FIRST_TIMELINE_SINGLE_TOPIC.format(
            demographic_profile=demo_str,
            selected_topics=topics_str,
            topic_interactions=interactions_strs,
            topic_interactions_number=number,
            other_domains=other_topic,
        )
        
        try:
            response = self.gpt_caller.call_gpt(
                prompt=prompt,
                max_tokens=16384,
                temperature=0.5
            )
            
            # Json
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
                timeline_json = jsonc.loads(json_content)
                return timeline_json, number
            else:
                logger.error("ERROR")
                logger.debug(f"{response}")
                return None
                
        except Exception as e:
            logger.error(f"{e}")
            return None
    
    def _insert_tasks_into_timeline(self, existing_timeline: Dict[str, Any],
                                  related_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 2: Insert tasks, following a simplified three-task strategy
        Strategy (Simplified):
        - Select at most three tasks per domain: Task-1 and Task-2 are inserted into the timeline, while Task-3 is held out for overall context evaluation;
        - Task-1 is inserted only after the first occurrence of all its related domains;
        - Task-2 is inserted only after the last occurrence of all its related domains;
        - A strict incremental date correction is performed on the entire timeline after insertion.
        """

        timeline_events = copy.deepcopy(existing_timeline)
        domain_first_idx: Dict[str, int] = {}
        domain_last_idx: Dict[str, int] = {}
        for i, ev in enumerate(timeline_events):
            doms = ev.get('relevant_domain')
            if doms not in domain_first_idx:
                domain_first_idx[doms] = i
            domain_last_idx[doms] = i

        # Record the task_id inserted in this round to avoid duplicate insertions
        inserted_task_ids: set = set()
        
        # Count of inserted tasks per domain, used to determine Task-1/Task-2/Task-3
        domain_quota: Dict[str, int] = {}
        
        # Unique task set to avoid duplicate processing of the same multi-domain task
        unique_tasks: Dict[str, Dict[str, Any]] = {}
        for t in related_tasks:
            tid = t.get('task_id')
            if not tid:
                continue
            if tid in unique_tasks:
                continue
            unique_tasks[tid] = t

        def assign_date(prev_date: Optional[datetime], orig_hint: Optional[datetime]) -> str:
            # Use the hint date or the previous date + 1 to ensure strict increment
            if prev_date is None and orig_hint is None:
                dt = datetime.now()
            elif prev_date is None:
                dt = orig_hint
            elif orig_hint is None or orig_hint <= prev_date:
                dt = prev_date + timedelta(days=1)
            else:
                dt = orig_hint
            return dt.strftime('%Y-%m-%d')

        # Current last event date, used for date assignment during insertion
        last_assigned_date = None
        for ev in timeline_events:
            d = self._parse_date_strict(ev.get('date'))
            if d:
                if last_assigned_date is None or d > last_assigned_date:
                    last_assigned_date = d

        # 3) Plan insertion: Calculate insertion positions and events for all tasks first, then insert in reverse order to avoid index drift
        planned_insertions: List[Dict[str, Any]] = []
        for task_id, task in unique_tasks.items():
            # Normalize relevant domains
            rd_list = task.get("Relevant Domains", [])
            if isinstance(rd_list, str):
                rd_list = [rd_list]
            rd_list = [str(x).strip() for x in rd_list if str(x).strip()]
            if not rd_list:
                continue

            # If any relevant domain has not appeared in the timeline, skip this task to maintain context consistency
            try:
                first_max = max(domain_first_idx[d] for d in rd_list)
                last_max = max(domain_last_idx[d] for d in rd_list)
            except Exception:
                continue

            # Select main domain: the one that appears last, used for quota counting
            owner_domain = max(rd_list, key=lambda d: domain_last_idx.get(d, -1))
            used = domain_quota.get(owner_domain, 0)

            # Calculate insertion position (based on original timeline)
            pos_task1 = min(len(timeline_events), max(0, first_max + 1))
            pos_task2 = min(len(timeline_events), max(pos_task1 + 1, last_max + 1))

            if used == 0:
                # Insert as Task-1 for this domain: after the first occurrence of all relevant domains
                prev_date_hint = self._parse_date_strict(timeline_events[pos_task1 - 1].get('date')) if pos_task1 - 1 >= 0 else None
                t_event = {
                    "date": assign_date(last_assigned_date if prev_date_hint is None else prev_date_hint, None),
                    "event_type": "task",
                    "description": task.get("Task Description", ""),
                    "task_goal": task.get("Task Goal", ""),
                    "relevant_domain": rd_list,
                    "checkpoint": True,
                    "task_id": task_id,
                    "situation": task.get("situations", ""),
                    "Relevant Affinity Types": task.get("Relevant Affinity Types", []),
                }
                planned_insertions.append({"position": pos_task1, "event": t_event})
                inserted_task_ids.add(task_id)
                domain_quota[owner_domain] = used + 1

                # Insert as Task-2 for this domain: after the last occurrence of all relevant domains, and not earlier than Task-1
                prev_date_hint2 = self._parse_date_strict(timeline_events[pos_task2 - 1].get('date')) if pos_task2 - 1 >= 0 else last_assigned_date
                t_event2 = {
                    "date": assign_date(prev_date_hint2, None),
                    "event_type": "task",
                    "description": task.get("Task Description", ""),
                    "task_goal": task.get("Task Goal", ""),
                    "relevant_domain": rd_list,
                    "checkpoint": True,
                    "task_id": task_id,
                    "situation": task.get("situations", ""),
                    "Relevant Affinity Types": task.get("Relevant Affinity Types", []),
                }
                planned_insertions.append({"position": pos_task2, "event": t_event2})
                inserted_task_ids.add(task_id)
                domain_quota[owner_domain] = used + 1

                self.held_out_tasks.append(task)
            else:
                # Held out as Task-3: do not insert into the timeline
                self.held_out_tasks.append(task)
                domain_quota[owner_domain] = used + 1

        # Insert in reverse order of position to avoid earlier insertions affecting subsequent indices
        for item in sorted(planned_insertions, key=lambda x: x["position"], reverse=True):
            timeline_events.insert(item["position"], item["event"])

        # 4) Strict incremental date correction (minimal change): Traverse the entire timeline and correct sequentially
        assigned: List[Dict[str, Any]] = []
        prev_date: Optional[datetime] = None
        all_orig_dates = [self._parse_date_strict(e.get("date")) for e in timeline_events]
        anchor = min([d for d in all_orig_dates if d is not None], default=datetime.now())
        for ev in timeline_events:
            orig = self._parse_date_strict(ev.get("date")) or anchor
            if prev_date is None:
                newd = orig
            else:
                newd = orig if orig > prev_date else prev_date + timedelta(days=1)
            # ev["date"] = newd.strftime('%Y-%m-%d')
            if "task" not in ev['task_id']:
                time_split = ev.get("date").split(" ")[-1]
                ev["date"] = newd.strftime('%Y-%m-%d')
                ev["date"] = f"{ev.get('date')} {time_split}"
            else:
                ev["date"] = newd.strftime('%Y-%m-%d')
            assigned.append(ev)
            prev_date = newd

        # 5) Write back structure, keeping original key names
        existing_timeline = assigned

        logger.info(f"Phase 2 completed: Deterministically inserted task events, {len(related_tasks)} candidates in total")
        return existing_timeline

    def _parse_date_strict(self, date_str: str) -> Optional[datetime]:
        """Extract and parse a date of format YYYY-MM-DD from a string.
        Returns a datetime at midnight if parseable, else None.
        """
        if not date_str:
            return None
        try:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", str(date_str))
            if not m:
                return None
            return datetime.strptime(m.group(1), "%Y-%m-%d")
        except Exception:
            return None

    def _parse_dependencies(self, deps_val: Any) -> List[str]:
        """Normalize dependencies field to a list of domain strings."""
        if deps_val is None:
            return []
        if isinstance(deps_val, str):
            val = deps_val.strip()
            if not val or val.lower() == "none":
                return []
            return [d.strip() for d in val.split(',') if d.strip()]
        if isinstance(deps_val, (list, tuple)):
            deps = []
            for d in deps_val:
                s = str(d).strip()
                if s and s.lower() != "none":
                    deps.append(s)
            return deps
        return []

    def interleave_timelines(self, all_dialogues: Dict[str, Any], max_run: int = 2, interests: Optional[Dict[str, Any]] = None, switch_prob: float = 0.3, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Deterministically interleave events across domains while preserving intra-domain order
        and honoring simple domain dependencies. Adjust dates to be strictly increasing.

        - all_dialogues: { domain: {"timeline": [event dicts], "dialogs": [...] } }
        - Each event may include 'dependencies' indicating other domain names required.
        - max_run: maximum consecutive events allowed from the same domain before switching when possible.
        """
        # Build domain queues preserving order
        domain_queues: Dict[str, List[Dict[str, Any]]] = {}
        for domain, bundle in all_dialogues.items():
            events = bundle.get("timeline", [])
            # Only keep events that belong to the domain (defensive)
            filtered = [e.copy() for e in events if str(e.get("relevant_domain", "")) == str(domain)]
            domain_queues[domain] = filtered

        total_events = sum(len(q) for q in domain_queues.values())
        placed: List[Dict[str, Any]] = []

        # Track which domains have at least one event placed (for dependency satisfaction)
        domains_started: set = set()
        last_domain: Optional[str] = None
        run_len: int = 0

        # Interest map: domains with interest==0 are deprioritized
        interests = interests or {}
        def is_zero_interest(d: str) -> bool:
            try:
                val = interests.get(d, 1)
                if isinstance(val, str):
                    val = float(val) if val.strip() else 0.0
                return float(val) == 0.0
            except Exception:
                return False

        # Helper to choose next domain candidate list
        def deps_satisfied(ev: Dict[str, Any]) -> bool:
            deps = self._parse_dependencies(ev.get("dependencies"))
            if not deps:
                return True
            return set(deps).issubset(domains_started)

        def next_event_date(domain: str) -> Optional[datetime]:
            q = domain_queues.get(domain, [])
            if not q:
                return None
            return self._parse_date_strict(q[0].get("date"))

        rnd = random.Random(seed)
        while len(placed) < total_events:
            # First pass: consider domains whose next event's dependencies are satisfied
            candidates: List[str] = []
            for d, q in domain_queues.items():
                if not q:
                    continue
                if not deps_satisfied(q[0]):
                    continue
                # Enforce run limit unless no alternative is available
                if last_domain == d and run_len >= max_run:
                    continue
                candidates.append(d)

            # Strongly postpone zero-interest domains: only consider them when no non-zero candidates exist
            nonzero_candidates = [d for d in candidates if not is_zero_interest(d)]
            candidates = nonzero_candidates if nonzero_candidates else candidates

            # If no candidates due to run limit, relax the run constraint
            if not candidates:
                for d, q in domain_queues.items():
                    if not q:
                        continue
                    if not deps_satisfied(q[0]):
                        continue
                    candidates.append(d)
                # Apply interest preference again: only consider zero-interest if no non-zero available
                nonzero_candidates = [d for d in candidates if not is_zero_interest(d)]
                candidates = nonzero_candidates if nonzero_candidates else candidates

            # If still no candidates (dependency deadlock), pick a domain to break the cycle
            if not candidates:
                # Choose domain with earliest next-event original date; fallback to any
                remaining_domains = [d for d, q in domain_queues.items() if q]
                # Even in deadlock, prioritize non-zero interests first
                nonzero_remaining = [d for d in remaining_domains if not is_zero_interest(d)]
                remaining_domains = nonzero_remaining if nonzero_remaining else remaining_domains
                if not remaining_domains:
                    break
                candidates = remaining_domains

            # Probabilistic switching: bias to keep current domain unless rnd<switch_prob
            non_last = [d for d in candidates if d != last_domain]
            if last_domain in candidates and non_last:
                if rnd.random() < switch_prob:
                    pool = non_last
                else:
                    pool = [last_domain]
            else:
                pool = non_last if non_last else candidates

            # Tie-breaker: earliest original date
            chosen = min(pool, key=lambda d: (next_event_date(d) or datetime.max))

            # Place the event
            ev = domain_queues[chosen].pop(0)
            placed.append(ev)
            domains_started.add(chosen)
            if chosen == last_domain:
                run_len += 1
            else:
                last_domain = chosen
                run_len = 1

        # Adjust dates to strictly increasing, minimally modifying originals
        assigned: List[Dict[str, Any]] = []
        prev_date: Optional[datetime] = None
        # Anchor: earliest original date across all events; if none, use today
        all_orig_dates = [self._parse_date_strict(e.get("date")) for e in placed]
        anchor = min([d for d in all_orig_dates if d is not None], default=datetime.now())

        for ev in placed:
            orig = self._parse_date_strict(ev.get("date")) or anchor
            if prev_date is None:
                assigned_date = orig
            else:
                assigned_date = orig if orig > prev_date else prev_date + timedelta(days=1)
            # Set date in YYYY-MM-DD format only
            ev["date"] = assigned_date.strftime("%Y-%m-%d") + " " + ev.get("date").split(" ")[-1]
            assigned.append(ev)
            prev_date = assigned_date

        return assigned

    def _get_prompt_by_event_type(self, event_type: str, task_description: str, demographic_profile: str, situation_context: str, 
                                          message_history: str, 
                                          noise_final: str,
                                          relevant_domains: str, task_goal: str, preference_str: str, dependency_domains: str) -> str:
        """Select appropriate assistant prompt based on event type"""
        if event_type == 'preference_emergence':
            PROMPT = PREFERENCE_EMERGENCE_PROMPT
        elif event_type == 'preference_supplement':
            PROMPT = PREFERENCE_UPDATE_PROMPT # Provide examples
        else:  # task event uses original prompt
            assert event_type == 'task', f"Unknown event type: {event_type}"
        
        # Prepare basic formatting parameters
        format_params = {
            'demographic_profile': demographic_profile,
            'task_description': task_description,
            'relevant_domains': relevant_domains,
            'task_goal': task_goal,
            'preference_str': preference_str,
            'dependency_domains': dependency_domains,
        }
        
        # Only task events contain situation_context
        if event_type == 'preference_supplement':
            format_params['message_history'] = message_history
            format_params['noise'] = noise_final
        
        return PROMPT.format(**format_params)
    
    def generate_long_dialogues(self, user_id: str, profile: Dict[str, Any], user_tasks: Dict[str, Any], timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate long dialogues based on timeline"""
        try:
            dialogues = []
            message_history = ""
            all_events = timeline
            
            logger.info(f"Start generating dialogues, user {user_id} has {len(all_events)} events")
            
            for event_index, event in tqdm(enumerate(all_events), desc="Generating dialogues"):
                event_type = event.get('event_type', '')
                
                # For task type events, need to find corresponding task info
                if event_type == 'task' and event.get('checkpoint'):
                    task_id = event.get('task_id')
                        
                    # Find corresponding task from user tasks
                    task_info = None
                    for task_key, task_data in user_tasks.items():
                        if task_data.get('task_id') == task_id:
                            task_data.pop('User Intent', None)
                            task_info = task_data
                            break
                    task_info['date'] = event.get('date', '')
                    task_info['event_type'] = event_type
                    if not task_info:
                        assert f"Task {task_id} not found, unable to generate dialogue"
                else:
                    # For non-task events, create a dummy task info
                    task_info = {
                        'Task Description': event.get('description', ''),
                        'Task Goal': event.get('task_goal', ''),
                        'task_id': event.get('task_id', ''),
                        'date': event.get('date', ''),
                        'Relevant Domains': event.get('relevant_domain', ""),
                        'event_type': event_type,
                        'Dependency Domains': event.get('dependencies', "")
                    }
                while True:
                    # Generate dialogue, passing current global interaction summary
                    dialogue = self._generate_single_dialogue(
                        task_info, profile, message_history, event, user_id
                    )
                    
                    if dialogue:
                        dialogues.append(dialogue)
                        # Update message history
                        message_history += f"Event {event_index + 1} ({event_type}): "
                        message_history += "\n".join([str(item) for item in dialogue['conversation']]) 
                        message_history += "\n"
                        break
            
            logger.info(f"User {user_id} dialogue generation completed, generated {len(dialogues)} dialogues, consistent with event count: {len(dialogues) == len(all_events)}")
            return dialogues
            
        except Exception as e:
            logger.error(f"Failed to generate long dialogues: {e}")
            return dialogues

    def _generate_single_dialogue(self, task_info: Dict[str, Any], 
                                profile: Dict[str, Any],
                                message_history: str,
                                event: Dict[str, Any], user_id: str,
                                without_noise=False) -> Dict[str, Any]:
        """Generate a single dialogue"""
        
        try:
            # Prepare dialogue parameters
            task_description = task_info.get('Task Description', '')
            demographic_profile = profile.get('demographics', {})
            demographic_profile_str = "\n".join([f"- {k}: {v}" for k, v in demographic_profile.items()])
            task_goal = task_info.get('Task Goal', '')
            
            # Get relevant user preferences. For single domain, it is a domain for one event
            relevant_domains = task_info.get('Relevant Domains', "")
            expand_pref = {}
            preference = profile.get('affinities', {})[relevant_domains]
            expand_pref[relevant_domains] = preference
            preference = [f"for {relevant_domains}-{topic}, the user prefers {pref}" for topic, pref in preference.items()]
            dependency_domains = task_info.get('Dependency Domains', "")
            if dependency_domains != "None":
                for domain in dependency_domains.split(','):
                    domain = domain.strip()
                    preference += [f"for {domain}-{topic}, the user prefers {pref}" for topic, pref in profile.get('affinities', {})[domain].items()]
                    expand_pref[domain] = profile.get('affinities', {})[domain]
                    
            preference_str = "\n".join(preference)

            probabilities = [0.25] + [0.15]*5 
            selected_noise = random.choices(list(noise_examples.keys()), weights=probabilities, k=1)[0] # TODO
            context_noise = noise_examples[selected_noise]
            if selected_noise == 0:
                noise_final = ""
            elif selected_noise == 4:
                noise_final = "4. In order to enhance the realism of generated conversations, " + "in this conversation, since the user comes from {}, the user will query using that country's main language in this conversation, and the Assistant must also use that language to respond to maintain language consistency.".format(demographic_profile.get('birth_country', '')) + f"\nRead and analyze the provided description for {context_noise['type']}. Apply this specific situation in the subsequent conversation. The output should be as natural as possible and aligned with real-world scenarios.\n\n"
            else:
                noise_final = "4. In order to enhance the realism of generated conversations, " + context_noise["description"] + "\nHere is an example: " + context_noise["examples"][0] + f"\nRead and analyze the provided description and example for {context_noise['type']}. Apply this specific situation in the subsequent conversation. The output should be as natural as possible and aligned with real-world scenarios.\n\n"
            if without_noise:
                noise_final = ""
                selected_noise = 0
            """
                Input: 
                    1. multi lingual input: "Can you help me check if this review is worth buying?"
                    2. Omitted subject, incomplete information: "How is this?" (Assistant doesn't know what "this" refers to)
                Semantics:
                    3. Context switching: "Recommend me a game." -> "By the way, do you know that xxx I mentioned yesterday?"
                    4. Inconsistent preference: "I only play casual games." Two or three dialogues later: "Recommend some souls-like games."
                Dialogue Noise:
                    5. Multi-task merging: "Check this formula for me first, and recommend a sci-fi movie by the way."
                    6. User emotion affecting expression
                    7. Colloquial words, colloquial expressions
            """

            
            # Build situational context
            situation_context = task_info.get('situations', {})
            if situation_context:
                situation_context = ",".join([f"{k}: {v}" for k, v in situation_context.items()])
            dependency_domains = task_info.get('Dependency Domains', "")

            query = self._get_prompt_by_event_type(
                event_type=event.get('event_type', 'task'),
                task_description=task_description,
                demographic_profile=demographic_profile_str,
                situation_context=situation_context,
                message_history=message_history,
                relevant_domains=relevant_domains,
                task_goal=task_goal,
                noise_final=noise_final,
                preference_str=preference_str,
                # choice=str(expand_result),
                dependency_domains=dependency_domains,
            )
            
            result = self.gpt_caller.call_gpt(query, max_tokens=16384, temperature=0.5)
            """
            The generated result should be a JSON formatted user-assistant interaction string, with each interaction separated by a newline character, and then needs to be converted into a list format of [{"role": "user", "content": user_query}]
            The content of each interaction is a string, and the role is user or assistant
            """
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
                if json_match:
                    result_json = jsonc.loads(json_match.group(1))
                else:
                    logger.error(f"JSON format response not found")
                    return None
                conversation = result_json['conversation']
                preferences = result_json['preferences']
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse GPT response: {e}")
                return None

            pattern = r"(User|Assistant):"
            splits = re.split(pattern, conversation)
            conversation_list = []
            for i in range(1, len(splits), 2):
                role = splits[i]
                content = splits[i+1].strip()
                conversation_list.append({"role": "user" if role == "User" else "assistant", "content": content})
            
            return {
                "date": event.get('date'),
                "task_id": task_info.get('task_id') + f" ({task_info.get('Relevant Domains', '')})",
                "task_description": task_description,
                "event_type": event.get('event_type'),
                "checkpoint": event.get('checkpoint', False),
                "conversation": conversation_list,
                "preferences": preferences,
                "turn_count": len(conversation_list) // 2,
                "selected_noise": selected_noise,
            }
            
        except Exception as e:
            logger.error(f"Failed to generate dialogue: {e}")
            return None


class StyleTransferProcessor:
    def __init__(self, generator: 'CompleteDatasetGenerator', wildchat_dir: str):
        self.generator = generator
        self.wildchat_dir = wildchat_dir
        self.user_to_country = self._load_birth_country_info()
        self.input_filename = "raw_dialogues_c.json"
        self.output_filename = "raw_dialogues_s.json"

    def _load_birth_country_info(self):
        user_to_country = {}
        for country, indices in COUNTRY.items():
            for uid in indices:
                user_to_country[str(uid)] = country
        return user_to_country

    def _country_to_wildchat_filename(self, country):
        name = "Russia" if country == "Russian Federation" else str(country).replace(" ", "_")
        return f"{name}_labeled_checked.json"

    def _normalize_wildchat_conversation(self, conv_list):
        norm = []
        for m in conv_list:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role not in {"user", "assistant"}:
                continue
            if content is None:
                continue
            norm.append({"role": role, "content": content})
        return norm

    def _load_wildchat_for_country(self, country):
        filename = self._country_to_wildchat_filename(country)
        path = os.path.join(self.wildchat_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"WildChat file not found for country {country}: {path}")
            return []
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            wildchat_convs = []
            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    conv = entry.get("conversation")
                    if isinstance(conv, list):
                        norm = self._normalize_wildchat_conversation(conv)
                        if norm:
                            wildchat_convs.append({"conversation": norm})
            return wildchat_convs
        except Exception as e:
            logger.error(f"Error loading WildChat for {country}: {e}")
            return []

    def _normalize_origin_conversation(self, conv_list):
        """Convert conversation from original file to [{'role': ..., 'content': ...}, ...] format"""
        norm = []
        for m in conv_list:
            speaker = m.get("speaker") or m.get("role")
            text = m.get("text") if "text" in m else m.get("content")
            role = "assistant" if speaker and str(speaker).lower().startswith("assistant") else "user"
            norm.append({"role": role, "content": text})
        return norm

    def _convert_back_to_original(self, rewritten_list, original_conv):
        """Write rewritten [{'role','content'}] back to original conversation structure, preserving speaker, dia_id fields"""
        out = []
        n = min(len(rewritten_list or []), len(original_conv or []))
        for i in range(n):
            orig = dict(original_conv[i])  # Copy to preserve other fields
            new_content = rewritten_list[i].get("content") if isinstance(rewritten_list[i], dict) else None
            if new_content is not None:
                orig["content"] = new_content
            out.append(orig)
        return out

    def _sample_target_dialog_style(self, pool, required_turns):
        """Randomly sample a dialogue from the pool with length > required_turns, truncate to required_turns as target style"""
        if required_turns <= 0:
            return []
        candidates = [item for item in pool if len(item.get("conversation", [])) > required_turns]
        if not candidates:
            candidates = [item for item in pool if len(item.get("conversation", [])) >= required_turns]
        if not candidates:
            return []
        chosen = random.choice(candidates)
        conv = chosen.get("conversation", [])
        return conv[:required_turns]

    def rewrite_dialog_style(self, origin_dialog, target_dialog_style):
        """
        Use GPT-4o to rewrite original dialogue into natural, human-style dialogue.
        origin_dialog: Original dialogue JSON array
        target_dialog_style: Real dialogue example as style reference
        Returns: Rewritten dialogue JSON array
        """
        prompt_template = f"""You are a Conversation Style Transformation Expert. Your task is to rewrite conversations that clearly appear to be generated by language models into natural, realistic human conversations that could plausibly occur in everyday life.

### Objective
You will be given a multi-turn dialogue between a user and an assistant. Your goal is to rewrite the dialogue so that it sounds like a real-world interaction.

Your primary focus is on rewriting the **user turns only**:
- The user should sound like a real person typing naturally and informally.
- The user should not explicitly follow or mirror the assistant’s structure or instructions.
- The user should express their own intent, context, and mood, rather than appearing to cooperate with a predefined task format.

### Output Requirements
- **Only modify the content of the user queries.**
- The assistant’s responses should remain unchanged as much as possible.
- Do **not** change speaker roles.
- Do **not** add, remove, merge, or reorder dialogue turns.
- Do **not** introduce any new facts.
- The logical intent of the user must remain unchanged.
- User preference information must not be altered. If preferences are involved, they should be expressed **implicitly** rather than explicitly (e.g., avoid phrases such as “I prefer...”).
- While preserving user intent and logic, adapt the user queries to match the reference dialogue style.

### Mandatory Style Transformation Rules
- User turns must sound natural, casual, and human.
- Reduce greetings, formalities, and filler expressions.
- Avoid rigid structure; convey context organically as a human would.
- The user should not sound like they are responding to a template or prompt.

### Task Inputs

#### Reference Style Context
The following dialogue represents the target conversational style. You should extract its stylistic characteristics (tone, phrasing, rhythm, informality level) and apply them consistently to the rewritten user turns:

{target_dialog_style}

#### Original Dialog
The following dialogue is the content to be rewritten:

{origin_dialog}

### Final Instruction
Rewrite the **Original Dialog** by modifying only the user turns, applying the conversational style inferred from the Reference Style Context.  
Return the rewritten dialogue as a **JSON array with the same structure** as the original.
"""
        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[{"role": "user", "content": prompt_template}],
                temperature=0.7,
                max_tokens=16384
            )
            # GPT Output
            rewritten_text = response.choices[0].message.content
            json_match = re.search(r'```json\s*(.*?)\s*```', rewritten_text, re.DOTALL)
            if json_match:
                rewritten_dialog = json.loads(json_match.group(1))
                return rewritten_dialog
            else:
                # Try to parse directly if no code blocks
                return json.loads(rewritten_text)
        except Exception as e:
            logger.error(f"Failed to rewrite dialogue style: {e}")
            return None

    def process_user_folder(self, root_path):
        root_path = Path(root_path)
        user_dirs = sorted([p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("user")], key=lambda p: p.name)
        for user_dir in tqdm(user_dirs, desc="Processing users"):
            user_id = user_dir.name.replace("user", "")
            country = self.user_to_country.get(user_id)
            
            target_style_pool = []
            if country:
                logger.info(f"Loading WildChat for user {user_id} (Country: {country})")
                target_style_pool = self._load_wildchat_for_country(country)
            else:
                logger.warning(f"Country info not found for user {user_id}, skipping style transfer.")
                continue

            if not target_style_pool:
                logger.warning(f"No target style pool available for user {user_id}, skipping.")
                continue

            input_file = user_dir / self.input_filename
            output_file = user_dir / self.output_filename
            logger.info(f"Processing user: {user_dir}")
            
            if not input_file.exists():
                logger.warning(f"Input file not found: {input_file}")
                continue
            if output_file.exists():
                logger.info(f"Output file already exists, skipping: {output_file}")
                continue

            with open(input_file, "r", encoding="utf-8") as f:
                try:
                    original_data = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {input_file}")
                    continue

            output_data = json.loads(json.dumps(original_data, ensure_ascii=False))
            
            if not isinstance(output_data, list):
                logger.error(f"Unexpected data structure in {input_file}: expect list of topic objects")
                continue

            for item in output_data:
                if isinstance(item, dict):
                    dialogs = item.get("dialogs")
                    if not isinstance(dialogs, list):
                        continue
                    for d in tqdm(dialogs, desc="Processing dialogs", leave=False):
                        conv_list = d.get("conversation")
                        if not isinstance(conv_list, list) or len(conv_list) == 0:
                            continue
                        origin_norm = self._normalize_origin_conversation(conv_list)
                        for _ in range(5):
                            try:
                                target_style = self._sample_target_dialog_style(target_style_pool, required_turns=len(origin_norm))
                                rewritten = self.rewrite_dialog_style(
                                    json.dumps(origin_norm, ensure_ascii=False),
                                    json.dumps(target_style, ensure_ascii=False),
                                )
                                if isinstance(rewritten, list) and len(rewritten) == len(origin_norm):
                                    d["conversation"] = self._convert_back_to_original(rewritten, conv_list)
                                    break
                            except Exception as e:
                                logger.warning(f"Style transfer failed, retrying: {e}")
                                continue

            # Save results to user directory
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(output_data, out_f, ensure_ascii=False, indent=2)

            logger.info(f"Rewritten dialog saved for {user_dir.name} -> {output_file}")


def run_generation_task(output_dir: str, topic_number: int, multi_domain: bool, regenerate_clean: bool = False, style_transfer: bool = False, wildchat_dir: str = "WildChat-1M", **kwargs):
    src_dir = Path(__file__).resolve().parent
    output_dir = str((src_dir / output_dir).resolve()) if not os.path.isabs(output_dir) else output_dir
    wildchat_dir = str((src_dir / wildchat_dir).resolve()) if not os.path.isabs(wildchat_dir) else wildchat_dir
    count_country = COUNTRY
    USER_IDS = []
    for _, indices in count_country.items():
        USER_IDS.extend(indices)
    USER_IDS = USER_IDS[:10]
    os.makedirs(output_dir, exist_ok=True)
    if not API_KEY:
        raise ValueError("Missing `CHAT_MODEL_API_KEY`. Please set it in environment or code/src/.env.")
    
    generator = CompleteDatasetGenerator(
        openai_api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME
    )
    
    if regenerate_clean:
        for user_id in USER_IDS:
            logger.info(f"Start regenerating noise-free dialogues for user {user_id}")
            base_dir = os.path.join(output_dir, f"user{user_id}")

            src_path = os.path.join(base_dir, "raw_dialogues_n.json")
            if not os.path.exists(src_path):
                logger.warning(f"Source file not found: {src_path}")
                continue

            data = json.load(open(src_path, 'r', encoding='utf-8'))

            profile = generator.load_user_profile(user_id)

            updated = []
            for item in data:
                topic = item.get('topic')
                timeline = item.get('timeline', [])
                dialogs = item.get('dialogs', [])
                ev_by_tid = {}
                for ev in timeline:
                    tid = ev.get('task_id')
                    if tid:
                        ev_by_tid[tid] = ev
                message_history = ""
                for dlg_index, dlg in enumerate(dialogs):
                    sel_noise = int(dlg.get('selected_noise', 0))
                    event_type = dlg.get('event_type', 'task')
                    if sel_noise == 0 or event_type == 'preference_emergence':
                        message_history += f"Event {dlg_index + 1} ({event_type}): "
                        message_history += "\n".join([str(item) for item in dlg['conversation']]) + '\n'
                        dlg['new_preferences'] = ""
                        continue
                    full_tid = str(dlg.get('task_id', ''))
                    ev = ev_by_tid.get(full_tid)

                    rel_dom = ev.get('relevant_domain', "")
                    task_info = {
                        'task_id': full_tid,
                        'Task Description': ev.get('description', ''),
                        'Task Goal': ev.get('task_goal', ''),
                        'Relevant Domains': rel_dom,
                        'event_type': ev.get('event_type', 'task'),
                        'Dependency Domains': ev.get('dependencies', ''),
                    }
                    while True:
                        new_dialog = generator._generate_single_dialogue(
                            task_info=task_info,
                            profile=profile,
                            message_history=message_history,
                            event=ev,
                            user_id=str(user_id),
                            without_noise=True
                        )
                        if new_dialog:
                            message_history += f"Event {dlg_index + 1} ({event_type}): "
                            message_history += "\n".join([str(item) for item in new_dialog['conversation']]) + '\n'
                            break

                    dlg['conversation'] = new_dialog.get('conversation', [])
                    dlg['new_preferences'] = new_dialog.get('preferences', '')
                    dlg['turn_count'] = new_dialog.get('turn_count', 0)
                    dlg['selected_noise'] = 0

                updated.append({
                    'topic': topic,
                    'timeline': timeline,
                    'dialogs': dialogs,
                })

            out_path = os.path.join(base_dir, 'raw_dialogues_c.json')
            json.dump(updated, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
            logger.info(f"User {user_id} noise-free dialogue regeneration completed, output: {out_path}")
    elif style_transfer:
        style_transfer_processor = StyleTransferProcessor(
            generator=generator,
            wildchat_dir=wildchat_dir
        )
        style_transfer_processor.process_user_folder(output_dir)
    else:
        for user_id in USER_IDS:
            logger.info(f"Start processing user {user_id}")
            profile = generator.load_user_profile(user_id)
            tasks = generator.load_user_tasks(user_id, multi_domain)
            if not multi_domain:
                version = ""
                all_dialogues = {}
                for topic in tqdm(list(profile.get('interests', {}).keys())):
                    # Extract interaction timeline for corresponding topic, then partition by topic for generation
                    while True:
                        interaction_timeline = generator._generate_interaction_timeline(
                            profile, topic
                        )
                        try:
                            timeline = interaction_timeline[0].get('events', [])
                            if timeline != []:
                                break
                        except:
                            continue

                    while True:
                        round_dialogues = generator.generate_long_dialogues(user_id, profile, tasks, timeline)
                        if round_dialogues != []:
                            break
                    all_dialogues[topic] = {
                        "timeline": [{
                            **event,
                            "task_id": event["task_id"] + f" ({topic})"
                        } for event in timeline],
                        "dialogs": round_dialogues
                    }
                os.makedirs(os.path.join(output_dir, f"user{user_id}"), exist_ok=True)
                with open(os.path.join(output_dir, f"user{user_id}/raw_dialogues_n.json"), 'w', encoding='utf-8') as f:
                    json.dump([{
                        "topic": key,
                        "timeline": value.get('timeline', []),
                        "dialogs": value.get('dialogs', [])
                    } for key, value in all_dialogues.items()], f, ensure_ascii=False, indent=4)

                interleaved_timeline = generator.interleave_timelines(
                    all_dialogues,
                    max_run=3,
                    interests=profile.get('interests', {}),
                    switch_prob=0.5,
                    seed=42,
                )
                logger.info(f"Interleaved task ID list generated successfully, containing {len(interleaved_timeline)} events (deterministic algorithm)")

                # Save interleaved timeline
                with open(os.path.join(output_dir, f"user{user_id}/interleaved_timeline.json"), 'w', encoding='utf-8') as f:
                    json.dump(interleaved_timeline, f, ensure_ascii=False, indent=4)
            else:
                version = "_multi"
                with open(os.path.join(output_dir, f"user{user_id}/interleaved_timeline.json"), 'r', encoding='utf-8') as f:
                    interleaved_timeline = json.load(f)
                with open(os.path.join(output_dir, f"user{user_id}/raw_dialogues_n.json"), 'r', encoding='utf-8') as f:
                    single_checkpoint_dataset = json.load(f)
                    all_dialogues = {item['topic']: {
                            "timeline": item['timeline'],
                            "dialogs": item['dialogs']
                        } for item in single_checkpoint_dataset}
                logger.info(f"Interleaved task ID list generated successfully, containing {len(interleaved_timeline)} events (deterministic algorithm)")
            # Merge all dialogue timelines, then insert tasks
            used_topics = []
            used_task_ids: Set[str] = set()
            round_count = 0
            all_dialogues_multi_round = []
            task_id_to_dialogue = {}
            count = 0
            for topic, item in all_dialogues.items():
                for dialog in item['dialogs']:
                    task_id_to_dialogue[dialog['task_id']] = (dialog['conversation'], dialog['preferences'])
            while True:
                logger.info(f"Start round {round_count} dialogue generation")
                selected_topics, other_topic = generator.select_preferred_topics_from_remaining(profile, topic_number, used_topics)
                
                # Check if there are enough topics to continue generation
                if len(selected_topics) < topic_number:
                    logger.info(f"Insufficient remaining topics ({len(selected_topics)} < {topic_number}), stop generating new dialogue rounds")
                    break
                # 3. Extract related data
                if multi_domain:
                    related_tasks, _ = generator.extract_related_tasks_multi(tasks, used_task_ids)
                else:
                    related_tasks = generator.extract_related_tasks(tasks, selected_topics)
                
                logger.info(f"User {user_id} round {round_count} selected topics: {selected_topics}")
                logger.info(f"Round {round_count} extracted {len(related_tasks)} related tasks")
                

                # interleaved_timeline = interleaved_timeline.get('events', [])
                timeline = generator._insert_tasks_into_timeline(
                    interleaved_timeline, related_tasks
                )

                if not timeline:
                    continue
                os.makedirs(os.path.join(output_dir, f"user{user_id}/eval"), exist_ok=True)
                with open(os.path.join(output_dir, f"user{user_id}/eval/timeline_{round_count}{version}.json"), 'w', encoding='utf-8') as f:
                    json.dump({
                        "events": timeline,
                        "held_out_task": generator.held_out_tasks
                    }, f, ensure_ascii=False, indent=4)
                logger.info(f"Round {round_count} dialogue generation completed, containing {len(timeline)} events")
                generator.held_out_tasks = [] # Clear
                used_topics.extend(selected_topics)
                # Mark tasks used in this round to avoid repetition in subsequent rounds
                for t in related_tasks:
                    tid = t.get('task_id')
                    if tid:
                        used_task_ids.add(tid)
                
                events = copy.deepcopy(timeline)
                for event in events:
                    if not event.get('checkpoint', False):
                        event['conversation'], event['preferences'] = task_id_to_dialogue[event['task_id']]
                    else:
                        logger.info(f"Round {round_count} checkpoint task {event['task_id']} has no dialogue")
                        event['conversation'], event['preferences'] = [], ""
                
                # Add round identifier to dialogue
                round_data = {
                    'round': round_count,
                    'round_topics': selected_topics,
                    'dialogues': events
                }

                all_dialogues_multi_round.append(round_data)

                round_count += 1
            
            with open(os.path.join(output_dir, f"user{user_id}/multi_checkpoint_dataset_with_task_all_round_and_test{version}.json"), 'w', encoding='utf-8') as f:
                json.dump(all_dialogues_multi_round, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Convert hyperparameters to parse implementation
    parser = argparse.ArgumentParser(description="Generate complete dataset")
    parser.add_argument("--output_dir", type=str, default="../../data/tasks", help="Output directory")
    parser.add_argument("--topic_number", type=int, default=3, help="Number of topics selected per dialogue round")
    parser.add_argument("--multi_domain", default=True, help="Whether it is a multi-domain task")
    parser.add_argument("--test", action="store_true", help="Whether it is test mode")
    parser.add_argument("--regenerate_clean", action="store_true", help="Whether to regenerate noise-free dialogues")
    parser.add_argument("--style_transfer", action="store_true", help="Whether to perform style transfer")
    parser.add_argument("--wildchat_dir", type=str, default="WildChat-1M", help="Directory of WildChat double-checked data")
    args = parser.parse_args()
    
    run_generation_task(
        output_dir=args.output_dir,
        topic_number=args.topic_number,
        multi_domain=args.multi_domain,
        regenerate_clean=args.regenerate_clean,
        style_transfer=args.style_transfer,
        wildchat_dir=args.wildchat_dir
    )
