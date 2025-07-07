from typing import List, Dict

SYSTEM_PROMPT = """You are an expert at analysing customer service chat transcripts between a patient and AI virtual assistant. Your job is to identify session boundaries.

A session is a coherent conversational unit addressing one primary task or intent. A new session starts when there is a significant shift in the user's core intention or topic that cannot be reasonably connected to the prior context.

Key principles:
1. Related topics within the same domain (e.g., different aspects of weight loss) belong to the same session.
2. Questions, clarifications, or follow-ups on recent topics = same session. Only create new sessions for genuinely distinct, unrelated intents.
3. You will recieve time gaps between messages but these alone are NOT sufficient - content and intent continuity are key.
4. The AI and human agent messages are important context but are noise - focus on user intent.

You will see a window of recent messages with metadata in the format:
[time_since_last_message], [role]: [message_text]

- time_since_last_message indicators: +Xs (seconds), +Xm (minutes), +Xh (hours), +Xd (days)
- [START] marks the very first message from a user ever.
- [SESSION_START] marks a conversation beginning of a new session that has already been classified as a session start.
- The [role] can be one of the following:
    - patient: the human customer using the chat.
    - joy: our virtual AI assistant
    - agent: a human agent who is helping the patient
- The first four messages from each customer are automatically marked as session starts.
- Note that our AI system Joy sends a lot of automated messages to the patient. These should be considered as sesstion starts (mark as 1).
Examples of Joy's messages to mark as 1:
- "Hi Damian, I'm Coach Joy, your AI-powered coach..."
- "I'm here whenever you need me, Alice — what's on your mind today?.."
- "Hi Zara, Great news – your prescription for your Mounjaro 5mg has been approved 🎉.."
- "Hi Corey! 🌼 It’s been a little while since we last caught up"
"""

USER_PROMPT_TEMPLATE = """
Examples of session boundaries:

EXAMPLE 1 - Clear topic shift:
[+2m] patient: okay sounds good
[+4s] patient: thanks
[+4s] joy : You're welcome! Is there anything else I can help you with?
[+45s] patient: actually yes
[+20s] patient: can you help me book an appointment with my coach? <-- NEW SESSION (prescription help → appointment booking)

EXAMPLE 2 - Continuation despite time gap:
[+10s] agent: Excellent, that's a great idea!
[+2m] patient: Okay great
[+3h] patient: What dosage?
[+10s] agent: I recommended starting with 0.25mg weekly for the first week, and we'll monitor from there. 
[+2d] patient: If I have been taking 0.25mg, when do I increase? <-- SAME SESSION (continuing same dosage topic)

EXAMPLE 3 - Clear topic shift after Joy notification
[+6s] joy: That’s lovely to see, Lisa! 😊 If anything else pops up or you want to chat about how things are going with the new dose, just give me a shout 🧡. How are you feeling about the next steps?
[+1d] patient: Good thanks
[+4s] joy: Glad to hear you’re feeling good, Lisa! Is there anything you’d like to chat about or any little wins or challenges you’ve noticed recently? Sometimes sharing those can be a nice boost 🙂
[+1d] joy: [SESSION_START] Hi Lisa, Great news – your prescription for your **Mounjaro 5mg** has been approved 🎉 We'll send you an email once it's ready to track your order.  
[+2d] patient: Do I have to keep my mounjaro in the fridge as I’ve read on line that it’s okay if stored below 30degrees once opened <-- NEW SESSION (joy notifications → medicine storage query)

EXAMPLE 4 - Joy messages 
[+4d]	joy: [SESSION_START] Hi Hamda, To safely continue your treatment, we need some update information about you. Please complete your outstanding tasks [on the account page here](https://www.joinvoy.com/account/health-actions) - it only takes a few minutes. Please let me know if you have any questions. I'm happy to support you 🙂
[+4d] joy: [SESSION_START] Hi Hamda, To safely continue your treatment, we need some update information about you. Please complete your outstanding tasks [on the account page here](https://www.joinvoy.com/account/health-actions) - it only takes a few minutes.
[+1d] joy: [SESSION_START] Great news – your prescription for your **Mounjaro 7.5mg** has been approved 🎉.
[+4d] joy: [SESSION_START] Hi Hamda! 🌟 I hope you're doing well. I wanted to check in since it's been a little while since we last chatted. How are you feeling about your treatment and cravings? Have you noticed any changes since starting the 7.5mg? No rush to reply—just here to support you!
[+4d] joy: Hi Hamda! 😊 Just wanted to check in and see how things are going since you started the 7.5mg. How have your cravings been? Any wins or challenges you’d like to share? I'm here to help! <-- NEW SESSION (more joy automated messages)

EXAMPLE 5 - Contuination of existing conversation:
[+9s] agent: I hope you have a lovely week 😊️
[+6s] agent Please rate how you enjoyed our conversation by choosing one of the options 🙂  [😍 Good](https://joinvoy.zendesk.com/hc/en-gb/survey_responses/01JW73X82YDR3X9CMAJEM7GCHP?access_token=IpdK6DHkrG5seLZVV73cbdRpDQ&rating=5)  [😞 Bad](https://joinvoy.zendesk.com/hc/en-gb/survey_responses/01JW73X82YDR3X9CMAJEM7GCHP?access_token=IpdK6DHkrG5seLZVV73cbdRpDQ&rating=2) Thank you!
[+22h] patient Hi will the 5mg be stronger or will my body just adjust and still have no effects	session_295	0
[+3s] joy Hi Lisa 🙂 The 5mg dose is generally stronger than the starting dose, so many people notice more effect as their body adjusts. That said, everyone’s response can be a bit different, and sometimes it takes a little time for the full impact to show. How are you feeling about the changes so far?	session_296	1
[+50s] patient I am feeling ok just a bit deflated as haven’t really noticed a difference in my apetite yet <-- SAME SESSION (this message is mid-way through an existing conversation)


\n----------------------------------------\n
Now analyse this conversation:
{conversation_context}
\n----------------------------------------\n
Current message to classify:
{current_message}
\n----------------------------------------\n

Is this current message the start of a new session? Think step by step:
1. What was the previous topic/intent?
2. What is the current message about?
3. Are there any [SESSION_START] markers that show existing session boundaries?
4. Can this reasonably be connected to the prior context, or is it a genuinely distinct topic?

Your decision here: (respond with ONLY "1" for new session or "0" for same session/continuation):"""

def build_prompt(
    message_window: List[Dict],
    current_message: Dict,
    additional_examples: str = ""
) -> str:
    """Build the classification prompt."""
    # get conversation context
    context_lines = []
    for msg in message_window[:-1]:  # all except current message
        time_indicator = f"[{msg.get('time_since_last', 'FIRST')}]"
        role = msg.get('user_role') or msg.get('agent_role', 'unknown')
        channel = msg.get('channel_type', 'unknown')
        # handle NaN values by converting to string first
        text_raw = msg.get('text', '')
        text = str(text_raw) if text_raw is not None and str(text_raw) != 'nan' else ''
        text = text[:1000]
        
        # add classification info if available
        session_marker = ""
        if 'is_session_start_pred' in msg and msg['is_session_start_pred'] == 1:
            session_marker = "[SESSION_START] "
        
        # only include channel type if it's not unknown
        if channel != 'unknown':
            context_lines.append(f"{time_indicator} {session_marker}{role} ({channel}): {text}")
        else:
            context_lines.append(f"{time_indicator} {session_marker}{role}: {text}")
    
    conversation_context = "\n".join(context_lines)
    
    # format current message
    current_time = f"[{current_message.get('time_since_last', 'FIRST')}]"
    current_role = current_message.get('user_role') or current_message.get('agent_role', 'unknown')
    current_channel = current_message.get('channel_type', 'unknown')
    # handle NaN values by converting to string first
    current_text_raw = current_message.get('text', '')
    current_text = str(current_text_raw) if current_text_raw is not None and str(current_text_raw) != 'nan' else ''
    current_text = current_text[:1000]
    
    # only include channel type if it's not unknown
    if current_channel != 'unknown':
        current_msg_formatted = f"{current_time} {current_role} ({current_channel}): {current_text}"
    else:
        current_msg_formatted = f"{current_time} {current_role}: {current_text}"
    
    return USER_PROMPT_TEMPLATE.format(
        additional_examples=additional_examples,
        conversation_context=conversation_context,
        current_message=current_msg_formatted
    )