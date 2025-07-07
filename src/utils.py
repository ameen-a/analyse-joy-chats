from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

def format_message_for_prompt(row: pd.Series) -> str:
    """Format a single message with metadata for the prompt."""
    time_indicator = f"[{row['time_since_last']}]" if pd.notna(row['time_since_last']) else "[FIRST]"
    role = row['user_role'] if pd.notna(row['user_role']) else row['agent_role']
    channel = row.get('channel_type', 'unknown')
    
    return f"{time_indicator} {role} ({channel}): {row['text'][:1000]}"  # truncate long messages

def get_message_window(
    df: pd.DataFrame, 
    current_idx: int, 
    channel_id: str,
    window_size: int = 8,
    classifications: Dict[str, int] = None
) -> List[Dict]:
    """Get sliding window of messages for context, including classification info if available."""
    channel_df = df[df['gpt_channel_id'] == channel_id].sort_values('created_at')
    channel_df = channel_df.reset_index(drop=True)
    
    # get current message index in channel
    current_msg_id = df.iloc[current_idx]['gpt_stream_id']
    current_channel_idx = channel_df[channel_df['gpt_stream_id'] == current_msg_id].index[0]
    
    # get window
    start_idx = max(0, current_channel_idx - window_size + 1)
    end_idx = current_channel_idx + 1
    
    window_messages = channel_df.iloc[start_idx:end_idx].to_dict('records')
    
    # add classification info if available
    if classifications:
        for msg in window_messages:
            msg_id = msg['gpt_stream_id']
            if msg_id in classifications:
                msg['is_session_start_pred'] = classifications[msg_id]
    
    return window_messages

def assign_session_ids(df: pd.DataFrame, classifications: Dict[str, int]) -> pd.DataFrame:
    """Assign session IDs based on classification results."""
    df = df.copy()
    # map classifications, filling missing values with 0 (not a session start)
    df['is_session_start_pred'] = df['gpt_stream_id'].map(classifications).fillna(0).astype(int)
    
    # sort by channel and time
    df = df.sort_values(['gpt_channel_id', 'created_at'])
    
    # assign session IDs
    session_id = 0
    current_channel = None
    
    session_ids = []
    for _, row in df.iterrows():
        if row['gpt_channel_id'] != current_channel:
            session_id += 1
            current_channel = row['gpt_channel_id']
        elif row.get('is_session_start_pred') == 1:
            session_id += 1
        
        session_ids.append(f"session_{session_id}")
    
    df['session_id'] = session_ids
    return df