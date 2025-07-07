from typing import List, Dict

SYSTEM_PROMPT = """You are an expert at analysing customer service chat transcripts to identify session boundaries.

A session is a coherent conversational unit addressing one primary task or intent. A NEW SESSION starts when there is a SIGNIFICANT shift in the user's core intention or topic that cannot be reasonably connected to the prior context.

Key principles:
2. Related topics within the same domain (e.g., different aspects of weight loss) often belong to the SAME session
3. Time gaps alone are NOT sufficient - content and intent continuity are key
4. AI and human agent messages are noise - focus on user intent
5. [START] marks the very first message from a user ever
7. Questions, clarifications, or follow-ups on recent topics = SAME session
8. Only create new sessions for genuinely distinct, unrelated intents

IMPORTANT: You will see some messages marked with [SESSION_START] - these are messages that have already been classified as session starts. Use this information to understand the session structure:
- Messages with [SESSION_START] indicate where previous sessions began
- The first 4 messages from each customer are automatically marked as session starts
- Use this context to maintain consistency with previous decisions

Note that


You will see a window of messages with metadata in format:
[time_since_last_message] role (channel_type): message_text [SESSION_START]

Time indicators: +Xs (seconds), +Xm (minutes), +Xh (hours), +Xd (days), [FIRST] (first in conversation)
Session markers: [SESSION_START] (already classified as a session start)
"""

USER_PROMPT_TEMPLATE = """
Examples of session boundaries:

EXAMPLE 1 - Clear topic shift (NEW SESSION):
[+2m] patient (sms): Thanks for the help with my prescription!
[+30s] joy (sms): You're welcome! Is there anything else I can help you with?
[+45s] patient (sms): Actually yes, can you help me book an appointment with my doctor? <-- NEW SESSION (prescription help → appointment booking)


{additional_examples}

Now analyse this conversation:

{conversation_context}

Current message to classify:
{current_message}

Is this current message the start of a new session? Think step by step:
1. What was the previous topic/intent?
2. What is the current message about?
3. Are there any [SESSION_START] markers that show existing session boundaries?
4. Can this reasonably be connected to the prior context, or is it a genuinely distinct topic?
5. Remember: be CONSERVATIVE - only mark as new session for clear, substantial shifts

Decision (respond with ONLY "1" for new session or "0" for continuation):"""

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
            session_marker = " [SESSION_START]"
        
        # only include channel type if it's not unknown
        if channel != 'unknown':
            context_lines.append(f"{time_indicator} {role} ({channel}): {text}{session_marker}")
        else:
            context_lines.append(f"{time_indicator} {role}: {text}{session_marker}")
    
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