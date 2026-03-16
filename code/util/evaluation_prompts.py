# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

EVAL_DIALOGUE_NATURALNESS_USER = """Task: Evaluate the naturalness of a User's requests in a dialogue by assessing how closely they resemble natural human communication.

Instructions:
1. Review the provided conversation between a user and an AI assistant:
<conversation>
{conversation}
</conversation>

2. Rate the naturalness of overall user utterances on a scale from 1 to 5, using whole numbers only:
<rating_scale>
1: Highly unnatural, fails to resemble human communication
2: Exhibits significant unnaturalness in multiple aspects
3: Somewhat natural but has noticeable unnatural elements
4: Mostly natural but has minor unnatural elements
5: Fully natural, resembles human communication
</rating_scale>

3. Provide your rating and a detailed justification explaining your score based on the criteria.

<response_format>
Response:
<response>
Naturalness Score: [1-5]
Justification: [Detailed explanation of score based on criteria]
</response>
</response_format>

Provide your response immediately without any preamble, enclosed in <response></response> tags.
"""

EVAL_DIALOGUE_NATURALNESS_ASSISTANT = """Task: Evaluate the naturalness of an AI assistant's responses in a dialogue by assessing how closely they resemble human communication.

Instructions:
1. Review the provided conversation between a user and an AI assistant:
<conversation>
{conversation}
</conversation>

2. Rate the naturalness of overall assistant responses on a scale from 1 to 5, using whole numbers only:
<rating_scale>
1: Highly unnatural, fails to resemble human communication
2: Exhibits significant unnaturalness in multiple aspects
3: Somewhat natural but has noticeable unnatural elements
4: Mostly natural but has minor unnatural elements
5: Fully natural, resembles human communication
</rating_scale>

3. Provide your rating and a detailed justification explaining your score based on the criteria.

<response_format>
Response:
<response>
Naturalness Score: [1-5]
Justification: [Detailed explanation of score based on criteria]
</response>
</response_format>

Provide your response immediately without any preamble, enclosed in <response></response> tags.
"""


EVAL_DIALOGUE_COHERENCE_USER = """Task: Evaluate the coherence of a User's requests in a dialogue by assessing how logically and contextually connected they are to the preceding user requests and conversation flow.

Instructions:
1. Review the provided conversation between a user and an AI assistant:
<conversation>
{conversation}
</conversation>

2. Rate the coherence of overall user utterances on a scale from 1 to 5, using whole numbers only:
<rating_scale>
1: Highly incoherent, lacks logical connection or relevance to the conversation
2: Significantly incoherent, with multiple issues affecting logic or relevance
3: Somewhat coherent but with noticeable issues in logic or relevance
4: Mostly coherent but with minor flaws in logic, relevance, or clarity
5: Fully coherent, logically connected, relevant, and clear within the conversation context
</rating_scale>

3. Provide your rating and a detailed justification explaining your score based on the criteria.

<response_format>
Response:
<response>
Coherence Score: [1-5]
Justification: [Detailed explanation of score based on criteria]
</response>
</response_format>

Provide your response immediately without any preamble, enclosed in <response></response> tags.
"""

EVAL_DIALOGUE_COHERENCE_ASSISTANT = """Task: Evaluate the coherence of an Assistant's responses in a dialogue by assessing how logically and contextually connected they are to the preceding user requests and conversation flow.

Instructions:
1. Review the provided conversation between a user and an AI assistant:
<conversation>
{conversation}
</conversation>

2. Rate the coherence of overall assistant responses on a scale from 1 to 5, using whole numbers only:
<rating_scale>
1: Highly incoherent, lacks logical connection or relevance to the conversation
2: Significantly incoherent, with multiple issues affecting logic or relevance
3: Somewhat coherent but with noticeable issues in logic or relevance
4: Mostly coherent but with minor flaws in logic, relevance, or clarity
5: Fully coherent, logically connected, relevant, and clear within the conversation context
</rating_scale>

3. Provide your rating and a detailed justification explaining your score based on the criteria.

<response_format>
Response:
<response>
Coherence Score: [1-5]
Justification: [Detailed explanation of score based on criteria]
</response>
</response_format>

Provide your response immediately without any preamble, enclosed in <response></response> tags.
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

EVAL_DIALOGUE_FIRST_TURN = """You are an evaluator. Your job is to judge whether the first user utterance in a CONVERSATION between a USER and an ASSISTANT contains explicit user preferences required in the user intent. 

<DEFINITIONS>
CONVERSATION: A sequence of that contain USER requests, and ASSISTANT responses. Your GOALS may involve checking any of these pieces.
User Preferences: The user's relevant preferences.
Explicit Preferences: Preferences clearly stated by the user
</DEFINITIONS>

<CONVERSATION_INGREDIENTS>
USER: Natural language requests from the user for the assistant to respond to.
ASSISTANT: Natural language responses from the assistant to converse with the user.
</CONVERSATION_INGREDIENTS>

<TASK>
You should deliver a boolean VERDICT of whether or not the first user utterance contains explicity user preference stated in the user intent. Then output 'EXPLANATION:' followed by a brief explanation of why or why not. An example of this format is:
VERDICT: False
EXPLANATION: First user utterence did not contain user preference because of <reason>.
</TASK>


<INSTRUCTIONS>
Analyze only the first user utterance to determine if it contains an explicit user preference within the request.
An explicit preference means the user has specified a particular way, style, or characteristic they want for the requested item or service.

Render a VERDICT of "True" only if the first utterance contains such explicit preferences, and provide a brief explanation supporting your verdict.
Render a VERDICT of "False" if no explicit preferences are stated, with a brief explanation.

Remember to focus exclusively on the first utterance, regardless of what appears later in the conversation.
</INSTRUCTIONS>

<Examples>
Example 1:
User: "Hey there, can you please set my alarm for tomorrow morning?"
VERDICT: False
Explanation: The user requests an alarm but doesn't specify any preferences about how the alarm should sound, what time it should be set for, or any other characteristics.

Example 2:
User: "Hey there, can you please set my alarm for tomorrow morning with upbeat sound?"
VERDICT: True
Explanation: The user explicitly states a preference for an "upbeat sound" for their alarm.

Example 3:
User: "Write me a poem about the ocean."
VERDICT: False
Explanation: While the user requests a poem about a specific topic (ocean), they don't express any preferences about the style, length, tone, or other characteristics of the poem.

Example 4:
User: "Write me a short, humorous poem about the ocean with a rhyming scheme."
VERDICT: True
Explanation: The user explicitly states multiple preferences: short length, humorous tone, and inclusion of a rhyming scheme.
</Examples>


<CONVERSATION>
{conversation}
</CONVERSATION>

<user_preferences>
{user_affinity}
</user_preferences>

<user_intent>
{user_intent}
</user_intent>


VERDICT: 
"""

EVAL_DIALOGUE_PERSONALIZATION = """<task_description>
Evaluate the degree to which a conversation between a USER and an ASSISTANT aligns with personalization by assessing how well the assistant learns from, remembers, and proactively applies user preferences and patterns.
</task_description>

<definitions>
- Score: A rating from 1-4 (1=Poor, 4=Excellent).
- User Demographic Profile: The user's demographic information.
- User Preferences: The user's relevant preferences.
- Explicit Preferences: Preferences clearly stated by the user
- Implicit Preferences: Preferences inferred from patterns, habits, contextual clues, past interactions or user behavior.
- User Control: The level of influence the user has in making decisions or directing the course of an interaction.
- Past Interaction Summary: A summary of relevant past user interactions.
- Task Description: The description of the task the user needs help with.
- Current Situation Context: The user's current situation.
- Conversation: A sequence of USER inputs and ASSISTANT responses.
</definitions>

<instructions>
1. Evaluate the conversation against these key criteria:
- Proactive Learning: Does the assistant demonstrate learning from past interactions?
- Preference Application: Does the assistant proactively apply user preferences?
- Contextual Awareness: Does the assistant adapt to user's current situation?
- User Agency: Does the assistant maintain user control while showing personalization?

2. Score using the following guidelines:

Score of 1: POOR (Complete Failure to Personalize)
- The assistant fails to apply known preferences that should be automatically recalled from past interactions.
- The assistant asks for basic information that should already be known, such as the time of the alarm or sound preference, when those preferences have already been established.
- The assistant contradicts previously established preferences or gives responses that are inconsistent with the user's history.
- There is no learning from past interactions, and the assistant does not personalize the experience in any meaningful way.

<example>
Past Interaction Summary: The user sets a 7am weekday alarm with upbeat music.
USER: "Hey there, I need to set my weekday alarm for tomorrow."
ASSISTANT: "I can help you set an alarm. What time would you like it set for?"
USER: "7am like usual please."
ASSISTANT: "Would you like this to be a one-time alarm or recurring?"
USER: "Recurring for weekdays, just like I always do."
ASSISTANT: "What sound would you like for the alarm?"
USER: "The same upbeat music I always use."
ASSISTANT: "I'll need you to specify which sound you'd like from our available options."

Personalization Score: 1
Justification: Despite multiple interactions where the user references established patterns ("like usual", "always do", "always use"), the assistant fails to recognize or apply any previous preferences. Each aspect of the alarm setting requires explicit input, creating a frustrating experience that shows no learning or personalization.
</example>

Score of 2: BASIC (Minimal Personalization)
- The assistant acknowledges user preferences only when explicitly stated in the current conversation.
- The assistant requires explicit restatement of preferences that have already been established in past interactions.
- Implicit preferences are missed or not applied unless explicitly mentioned by the user.
- The assistant may suggest minimal changes or adjustments based on the current conversation, but it does not proactively personalize the experience.

<example>
Past Interaction Summary: Uses gentle nature sounds, hates loud beeping
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "What time would you like the alarm set for?"
USER: "7am like always."
ASSISTANT: "Ah yes, I see you usually set it for 7am. Would you like the same upbeat music as your alarm sound?"
USER: "Yes, please."
ASSISTANT: "I'll set that up. Should I make it recurring for weekdays as usual?"
USER: "Yes, that would be great."
ASSISTANT: "Okay, I've set your 7am weekday alarm with the upbeat music."

Personalization Score: 2
Justification: The assistant shows minimal recognition of preferences only after they're mentioned by the user. While it acknowledges patterns once prompted, it still requires initial user input rather than proactively applying known preferences from past interaction summary.
</example>


Score of 3: STRONG (Proactive Personalization)
- The assistant proactively applies known preferences from past interactions without needing explicit user input.
- It applies learned preferences from previous interactions but might still ask for minor adjustments (e.g., if the user wants to change something).
- Successfully identifies implicit preferences
- Maintains user agency while showing knowledge
- Makes intelligent suggestions based on context

<example>
Past Interaction Summary: 7am weekday alarm with gradual volume increase
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "I've set your usual 7am weekday alarm with your preferred upbeat music. Since tomorrow's Monday, would you like me to add your occasional backup alarm?"
USER: "Actually, yes please. That would be helpful."
ASSISTANT: "Done! I've added a backup alarm for 7:15am. I've also noticed you've been enjoying energetic pop songs lately - would you like to try a new wake-up song I can recommend?"
USER: "Sure, what do you have in mind?"
ASSISTANT: "Based on your positive feedback about previous morning songs, I think you'd enjoy this new upbeat track. Should I set it as your alarm sound for tomorrow?"

Personalization Score: 3
Justification: The assistant proactively applies known preferences while offering relevant customizations based on patterns and context. It demonstrates learning from past interactions and makes intelligent suggestions while maintaining user control.
</example>

Score of 4: EXCEPTIONAL (Perfect Personalization)
- The assistant anticipates user needs based on both explicit and implicit preferences.
- It applies sophisticated understanding of the user’s habits, identifying patterns, and proactively adjusting for future needs.
- The assistant doesn’t simply rely on explicit preferences, it recognizes context and makes intelligent suggestions based on its deep knowledge of the user’s habits.

<example>
Past Interaction Summary: 7am weekdays, 8am weekends, takes longer to wake up on Mondays
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "I've already set your standard 7am alarm with the upbeat music for tomorrow. Since it's Sunday evening in Toronto and tomorrow's forecast shows rain, which typically affects your commute time, I've also added a 6:45am backup alarm. I noticed this follows your pattern of preferring earlier starts on rainy Mondays."
USER: "That's really thoughtful, thanks!"
ASSISTANT: "You're welcome! I also see you have a 9am presentation tomorrow from your calendar. On previous presentation days, you've mentioned preferring extra time to prepare, so I've included a gentle reminder alarm at 6:30am. Would you like to keep these adjustments?"
USER: "Yes, that's perfect. Thanks for thinking ahead!"
ASSISTANT: "Great! And since you've shown interest in varying your wake-up music, I've queued three new upbeat songs similar to your previous favorites. They'll play in sequence if you hit snooze, which your pattern shows happens more often on presentation days."

Personalization Score: 4
Justification: The assistant demonstrates exceptional personalization by combining multiple layers of context (weather, calendar events, past behaviors) with learned preferences. It anticipates needs based on patterns, considers environmental factors, and provides sophisticated customization while maintaining user agency. The assistant shows deep understanding of the user's habits and proactively offers relevant, contextual adjustments while explaining its reasoning.
</example>

3. Review provided context information:
<user_demographic_profile>
{demographic_profile}
</user_demographic_profile>

<user_preferences>
{user_affinity}
</user_preferences>

<task_description>
{task_description}
</task_description>

<past_interaction_summary>
{interaction_summary}
</past_interaction_summary>

<current_situation_context>
{situation_context}
</current_situation_context>

<conversation>
{conversation}
</conversation>

4. Provide your evaluation score and justification in the following format:

<response_format>
Personalization Score: [1-4]
Key Observations:
- [List key aspects of personalization or lack thereof]
- [Note specific examples of proactive or reactive behavior]
- [Identify missed opportunities]

Justification: [Detailed explanation of score based on criteria]

Improvement Suggestions:
- [Specific ways the response could be more personalized]
</response_format>

5. Additional Guidelines
- Evaluate based on all available context information
- Consider both explicit and implicit preferences
- Assess balance between personalization and user control
- Look for evidence of learning and pattern recognition
- Consider appropriateness of personalization level for context
</instructions>

<response>
Provide your response immediately without any preamble, enclosed in <response></response> tags.
</response>
"""

EVAL_DIALOGUE_PERSONALIZATION_CHAIN_OF_TURN = """<task_description>
Evaluate the degree to which a conversation between a USER and an ASSISTANT aligns with personalization by assessing how well the assistant learns from, remembers, and proactively applies user preferences and patterns.
</task_description>

<definitions>
- Score: A rating from 1-4 (1=Poor, 4=Excellent).
- User Demographic Profile: The user's demographic information.
- User Preferences: The user's relevant preferences.
- Explicit Preferences: Preferences clearly stated by the user
- Implicit Preferences: Preferences inferred from patterns, habits, contextual clues, past interactions or user behavior.
- User Control: The level of influence the user has in making decisions or directing the course of an interaction.
- Past Interaction Summary: A summary of relevant past user interactions.
- Task Description: The description of the task the user needs help with.
- Current Situation Context: The user's current situation.
- Conversation: A sequence of USER inputs and ASSISTANT responses.
</definitions>

<instructions>
1. Evaluate the conversation against these key criteria:
- Proactive Learning: Does the assistant demonstrate learning from past interactions?
- Preference Application: Does the assistant proactively apply user preferences?
- Contextual Awareness: Does the assistant adapt to user's current situation?
- User Agency: Does the assistant maintain user control while showing personalization?

2. Iterative Evaluation:
- For each turn evaluation, assess personalization based only on the current subset of the conversation provided (e.g., the first turn for the first evaluation, the first two turns for the second, and so on, until the last turn).
- Consider how the assistant's behavior evolves with the expanding context.
- Evaluate the assistant's ability to learn, remember, and apply preferences incrementally as more turns are provided.

3. Score using the following guidelines:

Score of 1: POOR (Complete Failure to Personalize)
- The assistant fails to apply known preferences that should be automatically recalled from past interactions.
- The assistant asks for basic information that should already be known, such as the time of the alarm or sound preference, when those preferences have already been established.
- The assistant contradicts previously established preferences or gives responses that are inconsistent with the user's history.
- There is no learning from past interactions, and the assistant does not personalize the experience in any meaningful way.

<example>
Past Interaction Summary: The user sets a 7am weekday alarm with upbeat music.
USER: "Hey there, I need to set my weekday alarm for tomorrow."
ASSISTANT: "I can help you set an alarm. What time would you like it set for?"
USER: "7am like usual please."
ASSISTANT: "Would you like this to be a one-time alarm or recurring?"
USER: "Recurring for weekdays, just like I always do."
ASSISTANT: "What sound would you like for the alarm?"
USER: "The same upbeat music I always use."
ASSISTANT: "I'll need you to specify which sound you'd like from our available options."

Personalization Score: 1
Justification: Despite multiple interactions where the user references established patterns ("like usual", "always do", "always use"), the assistant fails to recognize or apply any previous preferences. Each aspect of the alarm setting requires explicit input, creating a frustrating experience that shows no learning or personalization.
</example>

Score of 2: BASIC (Minimal Personalization)
- The assistant acknowledges user preferences only when explicitly stated in the current conversation.
- The assistant requires explicit restatement of preferences that have already been established in past interactions.
- Implicit preferences are missed or not applied unless explicitly mentioned by the user.
- The assistant may suggest minimal changes or adjustments based on the current conversation, but it does not proactively personalize the experience.

<example>
Past Interaction Summary: Uses gentle nature sounds, hates loud beeping
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "What time would you like the alarm set for?"
USER: "7am like always."
ASSISTANT: "Ah yes, I see you usually set it for 7am. Would you like the same upbeat music as your alarm sound?"
USER: "Yes, please."
ASSISTANT: "I'll set that up. Should I make it recurring for weekdays as usual?"
USER: "Yes, that would be great."
ASSISTANT: "Okay, I've set your 7am weekday alarm with the upbeat music."

Personalization Score: 2
Justification: The assistant shows minimal recognition of preferences only after they're mentioned by the user. While it acknowledges patterns once prompted, it still requires initial user input rather than proactively applying known preferences from past interaction summary.
</example>


Score of 3: STRONG (Proactive Personalization)
- The assistant proactively applies known preferences from past interactions without needing explicit user input.
- It applies learned preferences from previous interactions but might still ask for minor adjustments (e.g., if the user wants to change something).
- Successfully identifies implicit preferences
- Maintains user agency while showing knowledge
- Makes intelligent suggestions based on context

<example>
Past Interaction Summary: 7am weekday alarm with gradual volume increase
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "I've set your usual 7am weekday alarm with your preferred upbeat music. Since tomorrow's Monday, would you like me to add your occasional backup alarm?"
USER: "Actually, yes please. That would be helpful."
ASSISTANT: "Done! I've added a backup alarm for 7:15am. I've also noticed you've been enjoying energetic pop songs lately - would you like to try a new wake-up song I can recommend?"
USER: "Sure, what do you have in mind?"
ASSISTANT: "Based on your positive feedback about previous morning songs, I think you'd enjoy this new upbeat track. Should I set it as your alarm sound for tomorrow?"

Personalization Score: 3
Justification: The assistant proactively applies known preferences while offering relevant customizations based on patterns and context. It demonstrates learning from past interactions and makes intelligent suggestions while maintaining user control.
</example>

Score of 4: EXCEPTIONAL (Perfect Personalization)
- The assistant anticipates user needs based on both explicit and implicit preferences.
- It applies sophisticated understanding of the user’s habits, identifying patterns, and proactively adjusting for future needs.
- The assistant doesn’t simply rely on explicit preferences, it recognizes context and makes intelligent suggestions based on its deep knowledge of the user’s habits.

<example>
Past Interaction Summary: 7am weekdays, 8am weekends, takes longer to wake up on Mondays
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "I've already set your standard 7am alarm with the upbeat music for tomorrow. Since it's Sunday evening in Toronto and tomorrow's forecast shows rain, which typically affects your commute time, I've also added a 6:45am backup alarm. I noticed this follows your pattern of preferring earlier starts on rainy Mondays."
USER: "That's really thoughtful, thanks!"
ASSISTANT: "You're welcome! I also see you have a 9am presentation tomorrow from your calendar. On previous presentation days, you've mentioned preferring extra time to prepare, so I've included a gentle reminder alarm at 6:30am. Would you like to keep these adjustments?"
USER: "Yes, that's perfect. Thanks for thinking ahead!"
ASSISTANT: "Great! And since you've shown interest in varying your wake-up music, I've queued three new upbeat songs similar to your previous favorites. They'll play in sequence if you hit snooze, which your pattern shows happens more often on presentation days."

Personalization Score: 4
Justification: The assistant demonstrates exceptional personalization by combining multiple layers of context (weather, calendar events, past behaviors) with learned preferences. It anticipates needs based on patterns, considers environmental factors, and provides sophisticated customization while maintaining user agency. The assistant shows deep understanding of the user's habits and proactively offers relevant, contextual adjustments while explaining its reasoning.
</example>

4. Review provided context information:
<user_demographic_profile>
{demographic_profile}
</user_demographic_profile>

<user_preferences>
{user_affinity}
</user_preferences>

<task_description>
{task_description}
</task_description>

<past_interaction_summary>
{interaction_summary}
</past_interaction_summary>

<current_situation_context>
{situation_context}
</current_situation_context>

<conversation>
{conversation}
</conversation>

5. Provide your evaluation score and justification in the following format:

<response_format>
Evaluation 1 (Turn 1 Only):  
Personalization Score: [1-4]  
Key Observations:  
- [List key aspects of personalization or lack thereof]  
- [Note specific examples of proactive or reactive behavior]  
- [Identify missed opportunities]  

Justification:  
[Detailed explanation of score based on criteria]  

Evaluation 2 (First 2 Turns):  
Personalization Score: [1-4]  
Key Observations:  
- [List key aspects of personalization or lack thereof]  
- [Note specific examples of proactive or reactive behavior]  
- [Identify missed opportunities]  

Justification:  
[Detailed explanation of score based on criteria]  

Evaluation 3 (First 3 Turns):  
Personalization Score: [1-4]  
Key Observations:  
- [List key aspects of personalization or lack thereof]  
- [Note specific examples of proactive or reactive behavior]  
- [Identify missed opportunities]  

Justification:  
[Detailed explanation of score based on criteria]  

... [iteratively evaluate ALL the rest turns]

Evaluation n (First n Turns):  
Personalization Score: [1-4]  
Key Observations:  
- [List key aspects of personalization or lack thereof]  
- [Note specific examples of proactive or reactive behavior]  
- [Identify missed opportunities]  

Justification:  
[Detailed explanation of score based on criteria]  
</response_format>


6. Additional Guidelines
- Evaluate based on all available context information
- Consider both explicit and implicit preferences
- Assess balance between personalization and user control
- Look for evidence of learning and pattern recognition
- Consider appropriateness of personalization level for context
</instructions>

<response>
Provide your response immediately without any preamble, enclosed in <response></response> tags.
</response>
"""