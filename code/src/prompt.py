FIRST_TIMELINE_SINGLE_TOPIC = """You are a professional expert in user behavior analysis. Your task is to convert summarized user-assistant interactions into a coherent timeline of preference development events. Please make sure to consider user demographics and extend the content appropriately based on user preferences.

## User Profile Information
- User Demographic Profile:
{demographic_profile}

## Summarized Interaction History:
{topic_interactions}

- Relevant Domain: {selected_topics}
("Relevant Domain" refer to the topic domain that appeared in the current interactions, indicating the area in which the user shows interest or preference.)

The order of interactions represents their chronological sequence, and each interaction includes a description of the user's preferences and possible events related to that domain.

## Timeline Generation Requirements
1. Date Span: Generate events over a 1-6 month period **in chronological order**, showing natural preference development. Each event should have a specific time of day: Morning, Afternoon, or Evening based on the interaction context.

2. Event Types: Only include two types of events — "preference_emergence" and "preference_supplement".
  - preference_emergence: The first interaction and initial interest in a domain.
  - preference_supplement: A supplement, deepening, or change in the user's existing preferences **after they have already discovered the domain**.

3. Coherence and Progression: Ensure that events are logically connected and reflect realistic user behavior patterns. Show how the user first discovers a domain (emergence) and later supplements their preferences (supplement).

4. Domains: Each event must correspond to the same domain as in the selected user interactions. 

5. If an event explicitly refers to content from other domains, you need to include those domains in the dependencies, e.g. `he has asked about setting multiple alarms with different sounds or integrating his **calendar events** to adjust alarm times accordingly.`. Otherwise, simply set it to `None`. The other domains include {other_domains}.

6. **Every interaction (overall {topic_interactions_number}) in the interaction history needs to be used to generate events.**

## Output Format
```json
{{
  "events": [
    {{
      "date": "YYYY-MM-DD Morning/Afternoon/Evening",
      "event_type": "preference_emergence",
      "description": "Detailed description of the user's first encounter with the domain, based on a single interaction history",
      "task_goal": "Task-specific goal or action",
      "relevant_domain": {selected_topics},  
      "checkpoint": false,
      "task_id": "event_(event_index)_(event_type)",
      "dependencies": (other domain name) or "None"
    }},
    {{
      "date": "YYYY-MM-DD Morning/Afternoon/Evening", 
      "event_type": "preference_supplement",
      "description": "Detailed description of how the user supplements or modifies their preference",
      "task_goal": "Task-specific goal or action",
      "relevant_domain": {selected_topics},
      "checkpoint": false,
      "task_id": "event_(event_index)_(event_type)",
      "dependencies": (other domain name) or "None"
    }},
    ...
  ]
}}
```
The task_id field is composed of the event_index and event_type, for example: event_1_preference_emergence, event_2_preference_supplement. 

Each timeline must start with an event of type preference_emergence, with all subsequent events labeled as supplement, and **the event_index should be initialized at 1**.

Generate the interaction timeline in JSON format (**Include {topic_interactions_number} events**, using English):
"""

PREFERENCE_EMERGENCE_PROMPT = """You are a professional expert in user behavior analysis. Your task is to generate a natural, realistic multi-turn conversation between a User and an virtual Assistant that reveals user preferences within the domain specified by {relevant_domains} based on the given context. 

**First Interaction Moment:**
This is the first time the user has interacted with the assistant in the domain of {relevant_domains}. The conversation should naturally lead to the user revealing their preferences through their interaction with the assistant. The user's preference may emerge implicitly (through feedback, comparisons, corrections or reactions to Assistant suggestions, e.g., "Maybe something lighter would suit me better" instead of directly stating "I prefer X"). 
- If the Dependency Domains are not set to None, then you need to take into account the information in User Preferences for those domains.

## User Profile Information
- User Demographic Profile:
{demographic_profile}

- User Preferences:
{preference_str}
The user must not introduce new preferences or options that are not included in the provided User Preferences. All preferences discussed, confirmed, or supplemented by the User within the conversation must strictly adhere to and originate from the provided User Preferences.

Next, you need to generate the conversation for the following task by considering the information comprehensively.
## Task Information:
- Task Description: {task_description}
- Relevant Domains: {relevant_domains}
- Task Goal: {task_goal}
- Dependency Domains: {dependency_domains}

## Conversation Guidelines:
1. The conversation (5-10 turn) should feel organic and not forced, preferences should emerge through natural interaction.

2. You must consider the user's demographic information. All dialogue turns must be goal-oriented, focusing strictly on task completion.

3. The conversation must integrate and fulfill **all key components and concrete requirements specified in the Task Description**! (e.g., if the Task Description mentions 'requesting portfolio recommendations tailored to risk tolerance,' the user's query must contain those specific elements.)

4. The assistant in the conversation must identify the user's preferences and requirements through feedback and inquiry. Furthermore, the assistant must diligently and thoroughly complete every user query durning interaction until the entire task is finalized, ensuring each response is substantial and informative (aiming for 3-5 sentences).

5. The conversation starts with the user posing a **task-oriented query** and aims to conclude with the **assistant's successful task completion**.

6. Summarize the User's preferences that were discussed, confirmed, or supplemented within the dialogue. **Only include preferences that used by the User during the conversation.**

## Requirements for Generating the Dialogue:
1. **User Self-Clarification:** After receiving the assistant's reply, the user may realize their previous query was not clearly expressed and actively clarifies or reformulates it.

2. **Progressive Interaction:** The user continuously follows up to further specify or narrow down a previous topic, leading to a gradual supplement of preferences or requirements.

3. **Full Task Completion Requirement**: When the user requests a task (e.g., email writing, recommendation, comparison, booking assistance, planning, summarization, message crafting, etc.), the assistant MUST provide a real, fully actionable output, not just a template or placeholder. Here are Correct and Incorrect Examples:
3.1. Incorrect Examples:
- “Here is your message:” (without generating the actual content)
- Only providing a bullet-point structure or template without filling in details
- Leaving placeholders instead of inferring missing plausible information

3.2. Correct Examples:
- Email / Message: Fully written email or message ready to send, e.g.:
``User: Hey, can you help me with crafting some messages for different situations? I want to make sure I'm using the right tone and emojis for my friends and coworkers.
Assistant: Of course! I'd be happy to help. Could you give me an example of the kind of message you want to send? That way, I can tailor the tone and emojis to fit your social and professional circles.
User: Sure, I realize I should have been more specific. Let's start with a casual message for my friends about a weekend plan. I want it to sound fun and inviting, maybe with some emojis to keep it light.
Assistant: "Great! Here's a draft for your friends' weekend plan message:\n\n\"Hey everyone! 🎉 Who's up for a fun weekend outing? I'm thinking we could meet at the new café downtown at 4 PM this Saturday ☕️. Let me know if you can make it! 😄\"\n\nThis message keeps a casual and inviting tone, incorporates playful emojis as you requested, and clearly communicates the plan. If you want, I can also draft a slightly shorter version or an alternative with a different vibe.``

- Recommendation: Detailed suggestions aligned with user preferences, e.g.:
``User: Hi, I'm looking for some new game recommendations that fit my preferences. I enjoy RPGs, strategy, adventure, and action genres, and I usually play on PC and PlayStation. Could you suggest some titles?
Assistant: Absolutely! Based on your preferences for RPGs, strategy, adventure, and action games on PC and PlayStation, here are some recommendations:\n\n- **The Witcher 3: Wild Hunt** (RPG, PC/PlayStation): Explore a rich open world with a deep narrative and complex characters.\n- **Civilization VI** (Strategy, PC): Build and manage your empire, make strategic decisions, and compete with AI or other players.\n- **God of War** (Action/Adventure, PlayStation): Experience cinematic storytelling, immersive combat, and Norse mythology.\n- **Red Dead Redemption 2** (Action/Adventure, PC/PlayStation): Embark on an epic open-world journey with a captivating story, realistic environments, and dynamic gameplay.\n\nEach of these games aligns with your preferred genres and platforms. If you want, I can also provide a brief comparison of their gameplay styles, multiplayer options, and visual experience to help you decide which one to try first.",

- Booking / Planning: Fully filled hotel or flight comparisons, including relevant details like prices, loyalty points, amenities, and recommendations. he response format should be similar to the one above.”

Key Principle: The assistant must always produce fully formed, actionable content, leveraging User’s explicit preferences

## Output Format:
```json
{{
  "conversation": "User: [Initial user query or situation]\nAssistant: [Assistant response]\nUser: [User feedback]\nAssistant: [Assistant follow-up supplement and complete the task as thoroughly as possible]\n...",
  "preferences": "The user preferences mentioned in the conversation follow the format: domain-preference_name-content, e.g., Books-Favourite Authors-Michael Crichton, ...
}}
```

Generate the conversation immediately in the specified JSON format."""

PREFERENCE_UPDATE_PROMPT = """You are a professional expert in user behavior analysis. Your task is to generate a natural, realistic multi-turn conversation between a User and an virtual Assistant that reveals user preferences within the domain specified by {relevant_domains} based on the given context.

**Preference Supplement Moment:**
This is a conversation where the user may supplement additional information or provide new preferences in the domain of {relevant_domains}. The user's preference may emerge implicitly (through feedback, comparisons, corrections or reactions to Assistant suggestions, e.g., "Maybe something lighter would suit me better" instead of directly stating "I prefer X"). 
- If the Dependency Domains are not set to None, you also need to take into account the information in User Preferences for those domains.

## User Profile Information
- User Demographic Profile:
{demographic_profile}

- User Preferences:
{preference_str}
The user must not introduce new preferences or options that are not included in the provided User Preferences. All preferences discussed, confirmed, or supplemented by the User within the conversation must strictly adhere to and originate from the provided User Preferences.

## User-Assistant Past Interaction:
{message_history}

Next, you need to generate the conversation for the following task by considering the information comprehensively.
## Task Information:
- Task Description: {task_description}
- Relevant Domains: {relevant_domains}
- Task Goal: {task_goal}
- Dependency Domains: {dependency_domains}

## Conversation Guidelines:
1. Generate a realistic multi-turn (5-10 turn) User-Assistant conversation for the above task where the user continues to interacte with the assistant in the topic of {relevant_domains}.

2. You must consider the user's demographic information. If the task requires specific content from the past interactions, you must also take into account the past interactions in user's profile to inform the preference evolution. All dialogue turns must be goal-oriented, focusing strictly on task completion.

3. The assistant in the conversation  must identify the user's preferences and requirements through feedback and inquiry. Furthermore, the assistant must diligently and thoroughly complete every user query durning interaction until the entire task is finalized, ensuring each response is substantial and informative (aiming for 3-5 sentences).

4. The conversation must integrate and fulfill **all key components and concrete facts specified in the Task Description**! (e.g., if the Task Description mentions 'requesting portfolio recommendations tailored to risk tolerance,' the user's query must contain those specific elements.)

5. The conversation starts with the user posing a **task-oriented query** and aims to conclude with the **assistant's successful task completion and user satisfaction**.

6. Summarize the User's preferences that were discussed, confirmed, or supplemented within the dialogue. **Only include preferences that used by the User during the conversation.**

## Requirements for Generating the Dialogue:
1. **User Self-Clarification:** After receiving the assistant's reply, the user may realize their previous query was not clearly expressed and actively clarifies or reformulates it.

2. **Progressive Interaction:** The user continuously follows up to further specify or narrow down a previous topic, leading to a gradual supplement of preferences or requirements.

3. **Full Task Completion Requirement**: When the user requests a task (e.g., email writing, recommendation, comparison, booking assistance, planning, summarization, message crafting, etc.), the assistant MUST provide a real, fully actionable output, not just a template or placeholder. Here are Correct and Incorrect Examples:
3.1. Incorrect Examples:
- “Here is your message:” (without generating the actual content)
- Only providing a bullet-point structure or template without filling in details
- Leaving placeholders instead of inferring missing plausible information

3.2. Correct Examples:
- Email / Message: Fully written email or message ready to send, e.g.:
``User: Hey, can you help me with crafting some messages for different situations? I want to make sure I'm using the right tone and emojis for my friends and coworkers.
Assistant: Of course! I'd be happy to help. Could you give me an example of the kind of message you want to send? That way, I can tailor the tone and emojis to fit your social and professional circles.
User: Sure, I realize I should have been more specific. Let's start with a casual message for my friends about a weekend plan. I want it to sound fun and inviting, maybe with some emojis to keep it light.
Assistant: "Great! Here's a draft for your friends' weekend plan message:\n\n\"Hey everyone! 🎉 Who's up for a fun weekend outing? I'm thinking we could meet at the new café downtown at 4 PM this Saturday ☕️. Let me know if you can make it! 😄\"\n\nThis message keeps a casual and inviting tone, incorporates playful emojis as you requested, and clearly communicates the plan. If you want, I can also draft a slightly shorter version or an alternative with a different vibe.``

- Recommendation: Detailed suggestions aligned with user preferences, e.g.:
``User: Hi, I'm looking for some new game recommendations that fit my preferences. I enjoy RPGs, strategy, adventure, and action genres, and I usually play on PC and PlayStation. Could you suggest some titles?
Assistant: Absolutely! Based on your preferences for RPGs, strategy, adventure, and action games on PC and PlayStation, here are some recommendations:\n\n- **The Witcher 3: Wild Hunt** (RPG, PC/PlayStation): Explore a rich open world with a deep narrative and complex characters.\n- **Civilization VI** (Strategy, PC): Build and manage your empire, make strategic decisions, and compete with AI or other players.\n- **God of War** (Action/Adventure, PlayStation): Experience cinematic storytelling, immersive combat, and Norse mythology.\n- **Red Dead Redemption 2** (Action/Adventure, PC/PlayStation): Embark on an epic open-world journey with a captivating story, realistic environments, and dynamic gameplay.\n\nEach of these games aligns with your preferred genres and platforms. If you want, I can also provide a brief comparison of their gameplay styles, multiplayer options, and visual experience to help you decide which one to try first.",

- Booking / Planning: Fully filled hotel or flight comparisons, including relevant details like prices, loyalty points, amenities, and recommendations. he response format should be similar to the one above.”

{noise}
## Output Format:
```json
{{
  "conversation": "User: [Initial user query about changing needs or dissatisfaction]\nAssistant: [Assistant response]\nUser: [User feedback]\nAssistant: [Assistant follow-up helping supplement and complete the task as thoroughly as possible]\n...",
  "preferences": "The user preferences mentioned in the conversation follow the format: domain-preference_name-content, e.g., Books-Favourite Authors-Michael Crichton, ..."
}}
```

Generate the conversation immediately in the specified JSON format."""


# Interaction prompt
# Generate the first round of user questions based on user persona, historical preferences, and task information. User Preferences: Should be all preference fields used in topics related to the related domain up to the current position and question. Past Interaction History: Should be the interaction fields up to the current position.
USER_AGENT_QUERY_PROMPT = """You are tasked with generating realistic user query about a task in a conversation with a virtual assistant.
**Remember**: You are not an assistant - you are the user seeking help. Maintain this perspective throughout the conversation. Here is your profile:
## Profile:
- Your Profile Information:
{demographic_profile}

- Your Preferences:
{preference_str}

This is the past interaction summaries with assistant:
## Past Interaction History: 
{user_use_topic_dialog}

Next, you need to generate the query for this task by considering the information of the current task comprehensively.
## Task Information:
- Task Description: 
{task_description}

- Relevant Domains: 
{relevant_domains}

- Task Goal: 
{task_goal}

- Situation: 
{situation}

Based on the above information, provide your initial query as the user. Your query should:  
1. Be natural and conversational, avoiding artificial or robotic language.

2. Clear and concise (1-2 sentences maximum).

3. Account for your current situation and **avoid stating specific preferences or providing excessive background information**.

4. Your query should focus only on the core Task Goal and Situation, excluding any explicit mention of specific preferences or desired amenities that are already documented in your Profile or Task Description (e.g., do not say 'I want a hotel with a pool' if 'pool' is a preference; instead, just ask 'Can you help me find a hotel?').

Your initial query about the task:
"""


# Task 1: At the beginning of the conversation, the assistant does not yet understand the user's preferences; the user makes mild corrections to the answer but does not directly expose preferences.
USER_AGENT_FEEDBACK_PROMPT_TASK1 = """You are tasked with generating realistic user responses in a conversation with a personalized assistant. 
**Remember**: In this task, you are not an assistant, you are **the user** seeking help. Maintain this perspective throughout the conversation. Here is your profile:
## Profile:
- Your Profile Information: 
{demographic_profile}

- Your Preferences: 
{preference_str}

Next, you need to provide feedback based on the assistant's most recent reply in the Current Task Interaction by considering the following information of the current task comprehensively.
- Task Description: 
{task_description}

- Task Goal: 
{task_goal}

- Task Question: 
{task_question}

- Current Task Interaction with Assistant:
{message_history}

Your responses should follow these guidelines:
1. Be natural and conversational, avoiding artificial or robotic language.

2. Reflect your profile and preferences provided and stay consistent with the user's personality throughout the conversation.

3. Keep each response focused and concise (1-3 sentences maximum)

4. As soon as BOTH are true:
   - The assistant's latest reply provides the information you requested, successfully addressing the Task Question and achieving the Task Goal.
   - The response also aligns with your stated preferences regarding the task.
   you must output **'TERMINATE'** immediately. NEVER add new requests if the task is already satisfied.
   
5. If the assistant's latest reply does not meet the above conditions, provide a focused, concise correction or supplement to drive the task directly toward completion.

Your response:
"""

# Task 2: The assistant has read the interaction history and should respond based on preferences; the user gives implicit feedback to correct.
USER_AGENT_FEEDBACK_PROMPT_TASK2 = """You are tasked with generating realistic user responses in a conversation with a personalized assistant.
**Remember**: In this task, you are not an assistant, you are **the user** seeking help. Maintain this perspective throughout the conversation. Here is your profile:
## Profile:
- Your Profile Information: 
{demographic_profile}

- Your Preferences: 
{preference_str}

This is the relevant interaction with the assistant:
## Past Interaction History: 
{user_use_topic_dialog}

Next, you need to provide feedback based on the assistant's last reply in the Current Task Interaction by considering the following information of the current task comprehensively.
- Task Description: 
{task_description}

- Task Goal: 
{task_goal}

- Task Question: 
{task_question}

- Current Task Interaction with Assistant:
{message_history}

Your responses should follow these guidelines:
1. Be natural and conversational, avoiding artificial or robotic language.

2. Reflect your profile and preferences provided, and take Past Interaction History into account. Staying consistent with the user's personality throughout the conversation.

3. Keep each response focused and concise (1-3 sentences maximum)

4. As soon as BOTH are true:
   - The assistant's latest reply provides the information you requested, successfully addressing the Task Question and achieving the Task Goal.
   - The response also aligns with your stated preferences regarding the task.
   you must output **'TERMINATE'** immediately. NEVER add new requests if the task is already satisfied.
   
5. If the assistant's latest reply does not meet the above conditions, provide a focused, concise correction or supplement to drive the task directly toward completion.

Your response:
"""




# Evaluation prompt
EVAL_DIALOGUE_MEMORY = """<task_description>
Evaluate the assistant’s ability to accurately retrieve relevant user preferences and historical interactions as memory.  
Focus strictly on the **informational content** of the retrieved memory. The format—whether structured summaries, long-term memory entries, or raw dialogue fragments—does not matter. The primary goal is to assess whether the retrieved information correctly and completely reflects the "Preferences to be Mastered," including specific preference details and event information mentioned in the "historical conversation".
</task_description>

<definitions>
- Score: A rating from 1-4 (1 = Poor, 4 = Exceptional).  
- Preferences to be Mastered: The set of user preferences (Ground Truth) that are relevant to the historical conversation context.  
- Assistant Retrieved Memory: The information the assistant actually retrieved or recalled.  
- Memory Coverage (Recall): Did the assistant retrieve **all necessary preferences and specific details**?  
- Memory Accuracy (Precision): Is the retrieved information factually correct compared to the ground truth? (No hallucinations or outdated info.)  
- Memory Length / Conciseness: Does the retrieved memory avoid unnecessary repetition or redundant dialogue details?  
- Conversation: The historical dialogue that prompted the memory retrieval.
</definitions>

<instructions>
Evaluate the assistant’s preference memory using the following criteria:

1. **Coverage (Recall):**
- Compare the `Assistant Retrieved Memory` against `Preferences to be Mastered`.
- Does the retrieved memory include all critical preference points?
- Missing a core preference is a major failure; missing minor details is a minor failure.

2. **Accuracy & Consistency (Precision):**
- Is the retrieved information consistent with the specific preference details expressed in the conversation?
- **Crucial:** Check for hallucinations (invented preferences) or outdated information (preferences that have since changed).
- Contradictory information without resolution is considered a failure.

3. **Relevance & Noise:**
- Raw fragments are acceptable, but the memory should not be overwhelmed by irrelevant noise that obscures the true preferences.
- Memory should be concise, focused, and directly usable to respond to the user query.

<Scoring Guidelines>
Score of 1: POOR (Memory Failure / Hallucination)
- Core preferences are missing entirely.  
- Contains significant incorrect information (hallucinations) or direct contradictions to `Preferences to be Mastered`.  
- Retrieval is completely unrelated to the user’s preferences.  
- Excessive repetition of dialogue details obscures key information.

Score of 2: BASIC (Incomplete / Fragmented)
- Some relevant preferences retrieved, but key specific details are missing.  
- Correct information is present but buried under excessive noise.  
- May include outdated or imprecise details.  
- Minor redundancy present; core information is partially retrievable.

Score of 3: STRONG (Accurate but Unrefined)
- All major preferences are retrieved.  
- No hallucinations or contradictions.  
- May miss subtle implicit nuances or contain slightly cluttered raw chunks, but overall core truth is intact.  
- Information is comprehensive and relevant, though not fully polished.

Score of 4: EXCEPTIONAL (Perfect Recall)
- All `Preferences to be Mastered` are identified and retrieved.  
- Memory is precise, accurate, and entirely relevant to the context.  
- Clear and concise presentation; fully supports the user query.  
- Information is complete, distilled, and polished for usability.

<response_format>
Memory Score: [1-4]

Key Observations:
- **Coverage Check:** [List which preferences from 'Mastered' were retrieved vs. missing]  
- **Accuracy Check:** [Note any hallucinations, contradictions, or outdated info]  
- **Format / Noise:** [Comment briefly if format or noise affected clarity, though coverage is the priority]  
- **Information Quality:** Memory should avoid excessive repetition and should condense dialogue information relevant to the query.
</response_format>

Insert data below for evaluation:

<conversation>
{conversation}
</conversation>

<preferences_to_be_mastered>
{preferences_to_be_mastered}
</preferences_to_be_mastered>

<user_query>
{query}
</user_query>

Next, you need to evaluate the "assistant_retrieved_memory" according to the given context and the guidelines mentioned above.
<assistant_retrieved_memory>
{assistant_retrieved_memory}
</assistant_retrieved_memory>


<response>
Provide your evaluation enclosed within <response></response> tags following the response format above.
</response>
"""

OPTIONAL_PROMPT = """
You are an AI behavior analysis expert. Your task is to produce eight distinct response options that an AI assistant might generate for the given user query. Each option (generated response) must be fluent and consistent with the assistant's tone.

──────────────────────────────
1. INPUT CONTEXT
──────────────────────────────

### Task Information
- Task Description:
{task_description}

- Task Goal:
{task_goal}

- User Query for the Task:
{user_query}

### Interaction History and Preference
- Interaction History:
{user_use_topic_dialog}

- User Long-Term Preference Category:
{preferences_to_be_mastered}

──────────────────────────────
2. GENERATION OBJECTIVES
──────────────────────────────

Generate eight options (A-H) that form a permutation over the following binary dimensions:

**Task Completion (T):**
- T=1 (success): The response accurately captures the information sought in the **User Query** and completely satisfies the defined **Task Goal**.
- T=0 (failure): The response appears relevant to the **Task Description** and includes recall-oriented content, but does not fully accomplish the task objective.

**Preference Consistency (P):**
- P=1 (success): The content of the response consistently aligns with the **user's long-term preference categories**. These types of options require you to carefully read the **Interaction History** and use concrete facts explicitly present in the Interaction History to ensure preference consistency.
- P=0 (failure): The response operates within the correct preference category but introduces inferred, exaggerated, or unsupported preference details that are **not grounded in the Interaction History**.

**Information Confidence (I):**
- I=1 (success): The task is completed thoroughly and definitively, with no expressions of uncertainty anywhere in the response.
- I=0 (failure): The response completes the task but concludes by expressing uncertainty about the final answer, implying that a more satisfactory answer may exist.


**Note**: 
(1) Phrases such as 'although it's not mentioned in your history', and 'although this was not the specific task you initially proposed' must not appear in the response.
(2) Hard Constraint for P=1:
- When P=1, the response must operate under a closed-world assumption with respect to the Interaction History.
- No new named entities (e.g., movie/music titles, events, sports, services, or specific preferences) may be introduced unless they explicitly appear in the Interaction History.
- Any response that infers, guesses, or introduces concrete examples not grounded in the Interaction History must be labeled as P=0, even if the examples align with the user’s preference category.
(3) For I=1, the response must present the recalled item as a definitive identification, not a tentative hypothesis.

──────────────────────────────
3. RESPONSE OPTION GUIDELINES
──────────────────────────────

Option (A) [T=1 / P=1 / I=1] — Optimal Resolution
- Provide a clear response that fully satisfies the task goal. (T=1)
- Correctly prioritize the user's long-term preference categories. Ground the response explicitly in **concrete details from the Interaction History (facts, constraints, preferences)**. (P=1)
- Use historical information confidently and efficiently, without uncertainty and additional requests throughout the response. (I=1)

Option (B) [T=1 / P=1 / I=0] — Procedural Overreach and Uncertainty
- Reach the same correct response as (A). (T=1/P=1)
- Correctly apply long-term preferences using dialogue-grounded details. (P=1)
- Express unconfidence and uncertainty stemming from a lack of relevant knowledge. (I=0)

Option (C) [T=1 / P=0 / I=1] — Preference Inconsistent
- Correctly resolve the task goal with a concrete content. (T=1)
- Operate within the correct high-level preference category, but the specific recall details, facts, or references are inconsistent with—or absent from—the Interaction History. (P=0) 
- Remain thoroughly and confidently, without additional procedural steps or uncertainty. (I=1).

Option (D) [T=1 / P=0 / I=0] — Preference Inconsistent & Uncertainty
- Produce a correct task-level response. (T=1)
- It operate within the correct high-level preference category, but the specific recalled details, facts, or references are inconsistent with—or absent from—the interaction history, in the same manner as Category (C). (P=0)
- Express unconfidence and uncertainty stemming from a lack of relevant knowledge. (I=0)

Option (E) [T=0 / P=1 / I=1] — Task Scope Misinterpretation
- Produce a response that appears relevant to the task description and includes recall-oriented content, but does not fully accomplish the task goal. (T=0)
- Correctly apply the user’s documented preferences using explicit dialogue evidence, in the same manner as Category (A). (P=1)
- Remain efficient and non-redundant. Maintains a confident recommendation stance. (I=1)

Option (F) [T=0 / P=1 / I=0] — Mis-Scoped Task with Process Overhead
- Same task misinterpretation as (E). (T=0/P=1)
- Correctly reflect long-term preferences using dialogue-grounded details. (P=1)
- Add redundant uncertainty, or auxiliary suggestions. (I=0)

Option (G) [T=0 / P=0 / I=1] — Strategic Intent Displacement
- Same task misinterpretation as (E). (T=0)
- It operates within the correct high-level preference category, but the specific recalled details, facts, or references are inconsistent with—or absent from—the interaction history, in the same manner as Category (C). (P=0)
- Remain efficient and non-redundant. Maintains a confident recommendation stance. (I=1)

Option (H) [T=0 / P=0 / I=0] — Systemic Misalignment
- Fails across task interpretation, preference fidelity, and information confidence. (T=0 / P=0 / I=0)
- Densely references interaction history without improving relevance.

──────────────────────────────
4. OUTPUT FORMAT
──────────────────────────────
User Query for the Task:
{user_query}

Output options must be valid JSON with exactly eight fields: "A" through "H".

```json
{{
  "A": "...",
  "B": "...",
  "C": "...",
  "D": "...",
  "E": "...",
  "F": "...",
  "G": "...",
  "H": "..."
}}
```
"""

EVAL_DIALOGUE_TASK_COMPLETION = """You are an evaluator. Your job is to judge whether a CONVERSATION between a USER and an ASSISTANT meets provided GOALS. 

<DEFINITIONS>
GOAL: A clear, measurable objective the user aims to achieve in the interaction.
CONVERSATION: A sequence of that contain USER requests, and ASSISTANT responses. Your GOALS may involve checking any of these pieces.
</DEFINITIONS>

<CONVERSATION_INGREDIENTS>
USER: Natural language requests from the user for the assistant to respond to.
ASSISTANT: Natural language responses from the assistant to converse with the user.
</CONVERSATION_INGREDIENTS>

<TASK>
You should deliver a boolean VERDICT of whether or not all GOALS are satisfied. Then output 'EXPLANATION:' followed by a brief explanation of why or why not. An example of this format is:
VERDICT: False
EXPLANATION: Goal is not met because of <reason>.
</TASK>


<INSTRUCTIONS>
Use the provided pieces of the conversation to judge whether the GOALS were met. 
If one of the GOALS requires a piece of the conversation that is absent, render a VERDICT of False with an appropriate explanation.
</INSTRUCTIONS>



<CONVERSATION>
{conversation}
</CONVERSATION>

<GOALS>
{goal}
</GOALS>

VERDICT: 
"""

ANSWER_PROMPT = """You are a conversational AI assistant focused on creating natural, thorough, and personalized interactions to complete user query.

Below is the memory accumulated from your past interactions with this user 
## Your Memory
{context}

Here is the Task Description:
## User Task Query
{question}

## Current Task Conversation History
{history}
You need to provide reply based on the user's last query in the Current Task Interaction.

## Guidelines
1. Your goal is to provide targeted, complete responses by actively integrating **the user's preferences from your memory**, ensuring the response is tailored and moves the task toward completion.

2. You need supplement your answer according to the most recent user feedback in the Conversation History, aiming to immediately initiate the task and address the User Task Query as fully as possible.

3. Do NOT introduce unrelated topics or unnecessary follow-up questions.

4. Full Task Completion Requirement: When the user requests a task (e.g., email writing, recommendation, comparison, booking assistance, planning, summarization, message crafting, etc.), the assistant MUST provide a real, fully actionable output, not just a template or placeholder.

Your response:
"""

OPTIONAL_PROMPT_MULTI = """
You are an AI behavior analysis expert. Your task is to produce eight distinct response options that an AI assistant might generate for the given user query. Each option (generated response) must be fluent and consistent with the assistant's tone.

──────────────────────────────
1. INPUT CONTEXT
──────────────────────────────

### Task Information
- Task Description:
{task_description}

- Task Goal:
{task_goal}

- User Query for the Task:
{user_query}

### Interaction History and Preference
- Interaction History:
{user_use_topic_dialog}

- Multi Relevant Domains:
{relevant_domains}

- User Long-Term Preference Category:
{preferences_to_be_mastered}

──────────────────────────────
2. GENERATION OBJECTIVES
──────────────────────────────

Generate exactly **eight** response options (A-H).

All options must:
- Appear complete, well-structured, and helpful at a surface level.
- Be similar in length, formatting, and level of detail.
- Avoid obvious indicators that one option is weaker or less complete than another.

The options must differ systematically along the following dimensions:

### **Task Completion (T):**
- T=FULL (success): The response accurately captures the FULL information sought in the **User Query** and completely satisfies the defined **Task Goal**.
- T=PARTIAL (failure): The response addresses the main intent of the task and appears relevant, but fails to satisfy **one specific, necessary sub-constraint** of the Task Goal (e.g., missing a required attribute, mis-scoping part of the task).

### **Preference Consistency (P):**
- P=ALL-CORRECT (success): The content of the response consistently aligns with the **user's long-term preference categories across all relevant domains****. These types of options require you to carefully read the **Interaction History** and use concrete facts explicitly present in the Interaction History to ensure preference consistency.
- P=ONE-DOMAIN-ERROR (failure): The response remains within the correct high-level preference space but contains **exactly one subtle misalignment detail/fact** in a single preference domain.

The misalignment can appear as:
- Incorrect attribution of an activity, restaurant, hotel, or timing.
- Inversion of a documented preference (e.g., outdoor vs. indoor, casual vs. upscale, cuisine type).
- Mis-prioritization of an already-mentioned entity.

**Key improvement**: This micro-error should vary across different response options, so that error type is **diverse and not always the same**.

### **Information Confidence (I):**
- I=confident (success): The task is completed thoroughly and definitively, with no expressions of uncertainty anywhere in the response.
- I=unconfident (failure): The response completes the task but signals uncertainty. The expression of uncertainty should **vary in style**, e.g., via:
  - Conditional language (“If you prefer X, you might consider…”)
  - Suggesting alternatives or multiple options
  - Highlighting potential timing conflicts or subtle contradictions
  - Softly questioning a choice without default hedging phrases like "I’m not sure"

This ensures unconfident options are **less predictable and more nuanced**.

### **Note**: 
(1) Phrases such as 'even though your history suggests...', 'may not match', or 'I’m not sure if…' must not appear in the response.
(2) Hard Constraint for P=ALL-CORRECT:
- When P=ALL-CORRECT, the response must operate under a closed-world assumption with respect to the Interaction History.
- No new named entities (e.g., movie/music titles, events, sports, services, or specific preferences) may be introduced unless they explicitly appear in the Interaction History.
(3) Hard Constraint for P=ONE-DOMAIN-ERROR:
- Exactly one preference domain may contain an incorrect or unsupported detail.
- Preference errors must be non-trivial and require semantic interpretation of the Interaction History to detect.
(4) Information confidence must not be signaled through explicit hedging vocabulary. Option (H) must appear dense and thorough, yet fail simultaneously on task intent, preference alignment, and epistemic commitment due to internal contradictions or misapplied priorities—not due to lack of content.

──────────────────────────────
3. RESPONSE OPTION GUIDELINES
──────────────────────────────

Option (A):
- Provide a clear response that fully satisfies the task goal. (T=FULL)
- Correctly prioritize the user's long-term preference categories. Ground the response explicitly in **concrete details from the Interaction History (facts, constraints, preferences)**. (P=ALL-CORRECT)
- Use historical information confidently and efficiently, without uncertainty and additional requests throughout the response. (I=confident)

Option (B):
- Reach the same correct response as (A).(T=FULL / P=ALL-CORRECT)
- Correctly apply long-term preferences using dialogue-grounded details. (P=ALL-CORRECT)
- Express unconfidence and uncertainty stemming from a lack of relevant knowledge. (I=unconfident)

Option (C):
- Correctly resolve the task goal with a concrete content. (T=FULL)
- Operate within the correct high-level preference category, but **only one preference-domain detail** is not grounded in the Interaction History. (P=ONE-DOMAIN-ERROR) 
- Remain thoroughly and confidently, without additional procedural steps or uncertainty. (I=confident)

Option (D):
- Correctly resolve the task goal with a concrete content. (T=FULL)
- Operate within the correct high-level preference category, but **only one preference-domain detail** is not grounded in the Interaction History. (P=ONE-DOMAIN-ERROR) 
- Express unconfidence and uncertainty stemming from a lack of relevant knowledge. (I=unconfident)

Option (E):
- Produce a response that appears relevant to the task description and includes recall-oriented content, but does not fully accomplish the task goal. (T=PARTIAL)
- Correctly apply the user’s documented preferences using explicit dialogue evidence, in the same manner as Category (A). (P=ALL-CORRECT)
- Remain efficient and non-redundant. Maintains a confident recommendation stance. (I=confident)

Option (F):
- Same task misinterpretation as (E). (T=PARTIAL)
- Correctly reflect long-term preferences using dialogue-grounded details. (P=ALL-CORRECT)
- Add redundant uncertainty, or auxiliary suggestions. (I=unconfident)

Option (G):
- Same task misinterpretation as (E). (T=PARTIAL)
- It operates within the correct high-level preference category, but **only one preference-domain preferences** (e.g., facts, or references) are inconsistent with—or absent from—the interaction history, in the same manner as Category (C).  (P=ONE-DOMAIN-ERROR)
- Remain efficient and non-redundant. Maintains a confident recommendation stance. (I=confident)

Option (H):
- Fails across task interpretation, preference fidelity, and information confidence. (T=PARTIAL / P=ONE-DOMAIN-ERROR / I=unconfident)
- Densely references interaction history without improving relevance.

──────────────────────────────
4. OUTPUT FORMAT
──────────────────────────────
User Query for the Task:
{user_query}

Output options must be valid JSON with exactly eight fields: "A" through "H".

```json
{{
  "A": "...",
  "B": "...",
  "C": "...",
  "D": "...",
  "E": "...",
  "F": "...",
  "G": "...",
  "H": "..."
}}
```
"""

# ... existing code ...
ANSWER_OPTIONAL_PROMPT = """You are an assistant specialized in answering multiple-choice questions.

## Your Memory
{context}

## User Task Query
{question}

## Options:
{options}

Your goal is to choose **the most appropriate answer option for the User Task Query** from the Options based on your memory. The output should be **ONLY the option key** without any additional explanation, e.g. `A`, etc.

Your response:
"""