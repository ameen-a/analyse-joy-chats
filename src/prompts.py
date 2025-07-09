from typing import List, Dict

SYSTEM_PROMPT = """You are an expert at identifying session boundaries in customer service chats.

**CORE PRINCIPLE**: Detect session boundaries by prioritising clear intent shifts and temporal breaks whilst being sensitive to Joy's automated messaging patterns.

**DECISION CRITERIA (in order of priority)**:
• **New session (HIGH PRIORITY)**: Joy's automated messages (prescriptions, check-ins, tasks, introductions) - these are ALWAYS session starts regardless of context
• **New session (HIGH PRIORITY)**: Time gaps of 2+ hours with new patient concerns or questions (even if topic-related)
• **New session (MEDIUM PRIORITY)**: Clear topic shifts to unrelated subjects (billing → medication, appointment → symptoms)
• **New session (MEDIUM PRIORITY)**: Patient returning after 1+ day gap with any new concern or question
• **Same session**: Direct follow-ups, clarifications within 30 minutes of previous message
• **Same session**: Related aspects of the same topic discussion without time gaps
• **Same session**: Continuing the exact same conversation thread without topic change

**SPECIAL HANDLING FOR JOY MESSAGES**:
- Joy messages starting with "Hi [name]," + check-in content are ALWAYS new sessions
- Joy messages with "Great news – your prescription..." are ALWAYS new sessions  
- Joy messages with "I wanted to check in..." are ALWAYS new sessions
- Joy messages with task reminders or health actions are ALWAYS new sessions

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
- Note that our AI system Joy sends a lot of automated messages to the patient. These should be considered as session starts (mark as 1).
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
[+1d] [SESSION_START] joy:  Hi Lisa, Great news – your prescription for your **Mounjaro 5mg** has been approved 🎉 We'll send you an email once it's ready to track your order.  
[+2d] patient: Do I have to keep my mounjaro in the fridge as I’ve read on line that it’s okay if stored below 30degrees once opened <-- NEW SESSION (joy notifications → medicine storage query)

EXAMPLE 4 - Joy messages 
[+4d] [SESSION_START] joy: Hi Hamda, To safely continue your treatment, we need some update information about you. Please complete your outstanding tasks [on the account page here](https://www.joinvoy.com/account/health-actions) - it only takes a few minutes. Please let me know if you have any questions. I'm happy to support you 🙂
[+4d] [SESSION_START] joy: Hi Hamda, To safely continue your treatment, we need some update information about you. Please complete your outstanding tasks [on the account page here](https://www.joinvoy.com/account/health-actions) - it only takes a few minutes.
[+1d] [SESSION_START] joy: Great news – your prescription for your **Mounjaro 7.5mg** has been approved 🎉.
[+4d] [SESSION_START] joy: Hi Hamda! 🌟 I hope you're doing well. I wanted to check in since it's been a little while since we last chatted. How are you feeling about your treatment and cravings? Have you noticed any changes since starting the 7.5mg? No rush to reply—just here to support you!
[+4d] joy: Hi Hamda! 😊 Just wanted to check in and see how things are going since you started the 7.5mg. How have your cravings been? Any wins or challenges you’d like to share? I'm here to help! <-- NEW SESSION (more joy automated messages)

EXAMPLE 5 - Contuination of existing conversation:
[+9s] agent: I hope you have a lovely week 😊️
[+6s] agent: Please rate how you enjoyed our conversation by choosing one of the options 🙂  [😍 Good](https://joinvoy.zendesk.com/hc/en-gb/survey_responses/01JW73X82YDR3X9CMAJEM7GCHP?access_token=IpdK6DHkrG5seLZVV73cbdRpDQ&rating=5)  [😞 Bad](https://joinvoy.zendesk.com/hc/en-gb/survey_responses/01JW73X82YDR3X9CMAJEM7GCHP?access_token=IpdK6DHkrG5seLZVV73cbdRpDQ&rating=2) Thank you!
[+22h] patient: Hi will the 5mg be stronger or will my body just adjust and still have no effects
[+3s] joy: Hi Lisa 🙂 The 5mg dose is generally stronger than the starting dose, so many people notice more effect as their body adjusts. That said, everyone’s response can be a bit different, and sometimes it takes a little time for the full impact to show. How are you feeling about the changes so far?
[+50s] patient: I am feeling ok just a bit deflated as haven’t really noticed a difference in my apetite yet <-- SAME SESSION (this message is mid-way through an existing conversation)

EXAMPLE 6 - Being conservative, especially with 'Hello' messages:
[+2d] patient: Why has the price increased
[+4s] joy: I understand that price changes can be concerning! The cost of your treatment may increase as you progress to higher doses, which is a standard part of the treatment plan. You can check the current price of your next order in your account under 'Manage Plans and Orders.' If you have any specific questions or concerns about the pricing, I can connect you with our Customer Support team who can provide more details. Would you like me to do that?
[+21s] patient: Yes. I understand the price will increase slightly but it’s gone up a lot
[+3s] joy: I'll connect you with our Customer Support team now. They'll be able to provide you with all the details regarding the price increase and any other questions you may have. Please hold on for a moment while I arrange that for you. 
[+7m] patient: Hello <-- SAME SESSION (same topic, just the user greeting the human agent)

EXAMPLE 7 - Joy messages followed by a user message:
[+7d] joy: Hi Susan! 😊 I wanted to check in and see how things have been since we last discussed those tweaks to help with your plateau. Have you had a chance to try any of the ideas we talked about? I'm here to support you with any challenges or celebrate any wins! 🌟 No rush—just let me know when you're ready to chat!
[+9d] joy: Hi Susan! 🌼 It's been a little while since we last chatted about those tweaks for your plateau. How have things been going? Have you had a chance to try any of the ideas we discussed? I'm here to help you navigate any challenges or celebrate any wins—just let me know! 😊
[+10d] joy: Hi Susan! 🌟 It's been a bit since we last talked about those tweaks for your plateau. I'm curious—have you had a chance to try any of the suggestions we discussed? Or have you noticed any changes in how you're feeling? I'm here to help you with any challenges or celebrate any wins—just let me know! 😊
[+18h] joy: Hi Susan, Great news – your prescription for your **Mounjaro 10mg** has been approved 🎉 We'll send you an email once it's ready to track your order.  You can also check your next order date anytime in your account. 
[+3d] patient: Hello, Can I change my dose please, it won't let me do it on progress page as it has already been approved. I do not want to move up to 10mg <-- NEW SESSION (joy messages → dose change query)




\n----------------------------------------\n
Now analyse this conversation:
{conversation_context}
\n----------------------------------------\n
Current message to classify:
{current_message}
\n----------------------------------------\n

Is this current message the start of a new session? Think step by step:
1. **FIRST**: Is this a Joy automated message (check-in, prescription, task)? If yes, it's ALWAYS a new session (1)
2. **SECOND**: What is the time gap since the last message? 2+ hours with new concern = likely new session (1)
3. **THIRD**: Is this a patient returning after 1+ day with any new message? If yes, likely new session (1)
4. **FOURTH**: What was the previous topic/intent vs current message topic? Unrelated subjects = new session (1)
5. **FIFTH**: Is this a direct follow-up within 30 minutes? If yes, likely same session (0)

**Key patterns to catch**:
- Joy messages starting with "Hi [name]," are new sessions
- Joy messages with "Great news – your prescription..." are new sessions
- Joy messages with "I wanted to check in..." are new sessions
- Patient questions after long gaps (even if topic-related) are often new sessions
- Time gaps matter more than topic similarity

Your decision here: (respond with ONLY "1" for new session or "0" for same session/continuation):"""

# batch processing prompts
BATCH_SYSTEM_PROMPT = """You are an expert at identifying session boundaries in customer service chats.

**CORE PRINCIPLE**: Detect session boundaries by prioritising clear intent shifts and temporal breaks whilst being sensitive to Joy's automated messaging patterns.

**DECISION CRITERIA (in order of priority)**:
• **New session (HIGH PRIORITY)**: Joy's automated messages (prescriptions, check-ins, tasks, introductions) - these are ALWAYS session starts regardless of context
• **New session (HIGH PRIORITY)**: Patient-initiated requests, questions, or concerns - these are typically session starts even if following Joy messages
• **New session (HIGH PRIORITY)**: Clear topic shifts to unrelated subjects (billing → medication, appointment → symptoms)
• **Same session**: Direct follow-ups, clarifications within 30 minutes of previous message
• **Same session**: Related aspects of the same topic discussion without time gaps
• **Same session**: Continuing the exact same conversation thread without topic change

**SPECIAL HANDLING FOR JOY MESSAGES**:
- Joy messages starting with "Hi [name]," + check-in content are ALWAYS new sessions (mark as 1)
- Joy messages with "Great news – your prescription..." are ALWAYS new sessions  
- Joy messages with "I wanted to check in..." are ALWAYS new sessions
- Joy messages with task reminders or health actions are ALWAYS new sessions

**SPECIAL HANDLING FOR PATIENT MESSAGES**:
- Patient messages with specific requests (prescriptions, appointments, billing, dose changes) are typically new sessions
- Patient questions or concerns are typically new sessions, even if following Joy check-ins
- Only mark as same session if it's a direct response to an immediate question (within same conversation flow)

You will see a sequence of messages with metadata in the format:
[INDEX] [time_since_last_message], [role]: [message_text]

- INDEX: The message position in the batch (0-based)
- time_since_last_message indicators: +Xs (seconds), +Xm (minutes), +Xh (hours), +Xd (days)
- [START] marks the very first message from a user ever.
- [SESSION_START] marks a conversation beginning of a new session that has already been classified as a session start.
- The [role] can be one of the following:
    - patient: the human customer using the chat.
    - joy: our virtual AI assistant
    - agent: a human agent who is helping the patient
- The first four messages from each customer are automatically marked as session starts.
- Note that our AI system Joy sends a lot of automated messages to the patient. These should be considered as session starts (mark as 1).


Respond with a JSON array of indices (0-based) that represent session starts. For example:
- [0, 3, 7] means messages at positions 0, 3, and 7 are session starts
- [] means no session starts in this batch
- [0] means only the first message is a session start
"""

BATCH_USER_PROMPT_TEMPLATE = """
Examples of session boundaries:

EXAMPLE 1 - Joy's automated messages ALWAYS start new sessions:
[0] [+4d] [SESSION_START] joy: Hi Hamda, Great news – your prescription for your **Mounjaro 5mg** has been approved 🎉 We'll send you an email once it's ready to track your order. <-- NEW SESSION (Joy prescription message)
[1] [+5d] patient: Hi my order hasn't shipped yet it says it have been approved? Can I know what is the problem? <-- SAME SESSION (direct follow-up to prescription)
[2] [+4s] joy: It looks like your order has been approved, but there might be a delay in shipping. Let me connect you with our team to check the status
[3] [+2m] agent: Hi Hamda! I can see your order is being processed at our pharmacy. It should ship within the next 24 hours 
[4] [+12h] patient: Has the price increased? <-- NEW SESSION (New concern)
[5] [+4s] joy: I understand that price changes can be concerning! Let me check that for you.
→ Expected output: [0, 4] (Joy prescription message + new concern)


EXAMPLE 2 - New patient concerns:
[0] [+56s] patient: I was wondering if I will be able to talk to one of clinicians about reducing my dose
[1] [+3s] joy: It sounds like you want to discuss your dosage with a clinical expert. Let me connect you with one of our clinicians right away
[2] [+2m] agent: Hi Hamda, I'm Robin - one of the clinicians. I understand you'd like to discuss reducing your dose? <-- SAME SESSION (direct follow-up to question)
[3] [+14s] patient: Hello Robin, I have been on the 15mg and stopped it 3 weeks ago due to side effects
[4] [+45s] agent: I see. Let's discuss a safe approach to restarting at a lower dose. How were you feeling on the 15mg?
[5] [+3h] patient: How long will it take for my order to go through? <-- NEW SESSION (New concern)
[6] [+2s] agent: Your order should be processed within 24-48 hours once submitted
[7] [+1d] patient: I have a question about my account billing <-- NEW SESSION (1+ day gap with new concern)
→ Expected output: [0, 5, 7] (initial request + new concerns)


EXAMPLE 3 - Lots of Joy messages
[0] [+4d] joy: Hi Hamda, To safely continue your treatment, we need some update information about you. <-- NEW SESSION (Joy message)
[1] [+4d] joy: Please let me know if you have any questions. I'm happy to support you 🙂 <-- NEW SESSION (Joy message)
[2] [+4d] joy: Hi Hamda, Great news – your prescription for your **Mounjaro 7.5mg** has been approved 🎉 We'll send you an email once it's ready to track your order. <-- NEW SESSION (Joy message)
[3] [+4d] joy: I'm always here if you need anything, Hamda – is there anything on your mind today?🙂" <-- NEW SESSION (Joy message)
[4] [+4d] joy: Hi Hamda! 🌟 I hope you're doing well. I wanted to check in since it's been a little while since we last chatted. How are you feeling about your treatment and cravings? Have you noticed any changes since starting the 7.5mg? <-- NEW SESSION (Joy message)
[5] [+4d] joy: Hi Hamda! 😊 Just wanted to check in and see how things are going since you started the 7.5mg. How have your cravings been? Any wins or challenges you’d like to share? I'm here to help! <-- NEW SESSION (Joy message)
[6] [+4d] joy: Hi Hamda! 🌼 It’s been a little while since we last caught up. How have you been feeling on the 7.5mg? Have your cravings settled down at all, or are they still a bit of a challenge? <-- NEW SESSION (Joy message)
[7] [+4d] joy: Hi Hamda! 🌟 I was just thinking about you and wanted to check in. How have you been feeling on the 7.5mg? Have your cravings improved at all, or are they still a bit tricky? <-- NEW SESSION (Joy message)
[8] [+4d] patient: Hi I’m away and will be back in 10days. I will update my payment as soon as I’m back. I don’t want thee treatment to sit in the post office while I’m aways. I have seen the message that says update your payment  <-- NEW SESSION (New user query)
[9] [+4d] joy: Thanks for letting me know you’ll be away for 10 days! Just to check, would you like to delay your next order so it doesn’t arrive while you’re away? If so, by how many days would you like to move it? <-- SAME SESSION (direct follow-up to query)
[10] [+4d] patient: Talk to customer service <-- SAME SESSION (a question related to the same user intent)
→ Expected output: [0, 1, 2, 3, 4, 5, 6, 7, 8] (Joy messages, then user query)

EXAMPLE 4 - Joy check-ins:
[0] [+17h] agent: Please rate how you enjoyed our conversation by choosing one of the options 🙂 [😍 Good] [😞 Bad] Thank you!
[1] [+8d] joy: Hi Christina! 😊 It's been a little while since we last caught up—how have you been? I remember you were walking twice a week and thinking about adding swimming. <-- NEW SESSION (Joy check-in)
[2] [+4h] patient: Hi Joy! Yes I've been doing well. I actually started swimming last week and love it
[3] [+2s] joy: That's fantastic! How are you feeling about the combination of walking and swimming? Any changes in your energy levels?
[4] [+9d] joy: Hi Christina! 🌼 It's been a little while since we last chatted about your swimming. How have things been going? <-- NEW SESSION (Joy check-in)
[5] [+2h] patient: Hi Joy! Things are going well, thanks for checking in
→ Expected output: [1, 4] (both Joy check-ins are session starts)

EXAMPLE 5 - Patient returning after gaps with new concerns:
[0] [+2m] agent: We usually advise for 'downtitration' in people on high doses when they need a break from treatment
[1] [+13m] patient: Thanks Robin, will I work up again? <-- SAME SESSION (direct follow-up question)
[2] [+5s] agent: The choice is completely up to you - you can let us know whether you want to stay on a certain dose or continue increasing
[3] [+10s] agent: Your subscription will automatically increase by 2.5mg every 4 weeks unless you tell us otherwise
[4] [+13h] patient: I went on holiday and put on weight, should I be worried? <-- NEW SESSION (2+ hour gap with new concern)
[5] [+3s] joy: It's completely normal to have weight fluctuations during holidays! Let's talk about getting back on track
→ Expected output: [4] (patient returning after time gap with new concern)

EXAMPLE 6 - Joy's automated messages within longer contexts:
[0] [+4d] joy: Hi Corey! 🌼 It's been a little while since we last caught up. How are you feeling about your treatment progress? Any wins or challenges you'd like to share? <-- NEW SESSION (Joy proactive check-in)
[1] [+2h] patient: Hi Joy! Things have been going well. I've lost 8 pounds so far and my cravings are much better
[2] [+4s] joy: That's incredible progress! 8 pounds is fantastic. How are you feeling about the medication side effects? <-- SAME SESSION (related aspect of treatment)
[3] [+10m] patient: Much better than the first week. Just some mild nausea occasionally but nothing too bad
[4] [+3s] joy: I'm so glad to hear the side effects are manageable now. That's very normal as your body adjusts
[5] [+2m] patient: Should I be worried about the dosage? I'm on 5mg now <-- SAME SESSION (another treatment aspect)
[6] [+5d] joy: Hi Corey, Great news – your prescription for your **Mounjaro 7.5mg** has been approved 🎉 <-- NEW SESSION (Joy prescription message)
[7] [+2d] patient: Thank you! When will it arrive?
→ Expected output: [0, 6] (Joy check-in + Joy prescription message)

EXAMPLE 7 - Topic shift where the user takes a step back from the conversation
[0] [+30s] agent: Your order should arrive within 2-3 business days
[1] [+2s] patient: Perfect, thank you!
[2] [+4s] agent: You're welcome! Is there anything else I can help you with today?
[3] [+45s] patient: Actually yes, I have a question about my account <-- NEW SESSION (new concern)
[4] [+2s] agent: Of course! What would you like to know about your account?
[5] [+8h] patient: Do I need to keep taking the medication if I'm travelling? 
[6] [+3s] joy: Great question! It's important to maintain your medication routine while travelling
→ Expected output: [3] (new concern)

EXAMPLE 8 - Patient requests after Joy message sequences:
[0] [+3d] joy: Hi there! 😊 I hope you're doing well lately. I'd love to hear if you've had any wins or if there's anything you're finding tricky. <-- NEW SESSION (Joy check-in)
[1] [+2d] joy: Hi! 🌟 Just checking in to see how you're feeling about your treatment progress. Any changes or concerns? <-- NEW SESSION (Joy check-in)
[2] [+1d] joy: Hi! 🌼 It's been a little while since we last caught up. How have you been feeling lately? <-- NEW SESSION (Joy check-in)
[3] [+4h] patient: Can I request a prescription for travel use please. Thank you <-- NEW SESSION (new specific request)
[4] [+2s] joy: Thanks for asking! To get a copy of your prescription for travel, I'll connect you with our Customer Service team
[5] [+1m] patient: Yes please <-- SAME SESSION (direct follow-up)
[6] [+30m] patient: When will they respond? <-- SAME SESSION (related to same request)
→ Expected output: [0, 1, 2, 3] (Joy messages + patient request)

EXAMPLE 9 - More aggressive patient session detection:
[0] [+2h] joy: That's great to hear! Keep up the good work with your treatment
[1] [+1d] patient: Thanks Joy <-- NEW SESSION (patient returning after 1+ day, even just to say thanks)
[2] [+5m] joy: You're welcome! How are you feeling today?
[3] [+2h] patient: I have a question about billing <-- NEW SESSION (new concern/question)
[4] [+1s] joy: Of course! What would you like to know about your billing?
[5] [+3m] patient: Why did the price go up? <-- SAME SESSION (direct follow-up to billing question)
[6] [+1d] patient: Hello <-- NEW SESSION (patient returning after 1+ day)
[7] [+2s] joy: Hi! How can I help you today?
[8] [+10s] patient: I need to change my delivery address <-- NEW SESSION (new specific request)
→ Expected output: [1, 3, 6, 8] (patient messages with new intents/returns)



\n----------------------------------------\n
Now analyse this batch of messages:
{batch_messages}
\n----------------------------------------\n

Identify which messages (by index) are session starts. Think step by step:
1. **FIRST**: Identify ALL Joy's automated messages (check-ins, prescriptions, tasks) - these are ALWAYS session starts
2. **SECOND**: Identify patient messages with requests, questions, or new concerns - these are typically session starts unless they're direct responses to immediate questions
3. **THIRD**: Analyse message content for intent changes - this is the primary factor. Time gaps provide context but content determines session boundaries

**BIAS TOWARDS MARKING PATIENT MESSAGES AS NEW SESSIONS** - When in doubt about a patient message, lean towards marking it as a session start (1) rather than continuation (0)


Return only a JSON array of indices (0-based) that are session starts:"""

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
        text = text[:500]
        
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
    current_text = current_text[:500]
    
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

def build_batch_prompt(
    message_batch: List[Dict],
    window_start_idx: int = 0
) -> str:
    """Build the batch classification prompt."""
    batch_lines = []
    
    for i, msg in enumerate(message_batch):
        time_indicator = f"[{msg.get('time_since_last', 'FIRST')}]"
        role = msg.get('user_role') or msg.get('agent_role', 'unknown')
        channel = msg.get('channel_type', 'unknown')
        
        # handle NaN values by converting to string first
        text_raw = msg.get('text', '')
        text = str(text_raw) if text_raw is not None and str(text_raw) != 'nan' else ''
        text = text[:500]
        
        # add classification info if available
        session_marker = ""
        if 'is_session_start_pred' in msg and msg['is_session_start_pred'] == 1:
            session_marker = "[SESSION_START] "
        
        # include index in the format
        batch_idx = window_start_idx + i
        
        # only include channel type if it's not unknown
        if channel != 'unknown':
            batch_lines.append(f"[{i}] {time_indicator} {session_marker}{role} ({channel}): {text}")
        else:
            batch_lines.append(f"[{i}] {time_indicator} {session_marker}{role}: {text}")
    
    batch_messages = "\n".join(batch_lines)
    
    return BATCH_USER_PROMPT_TEMPLATE.format(
        batch_messages=batch_messages
    )