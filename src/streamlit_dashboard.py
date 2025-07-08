import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Chat Session Explorer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and improved readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4ecdc4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .chat-container {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #404040;
    }
    
    .message-row {
        margin: 8px 0;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .message-row.current {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        border: 2px solid #ff6b6b;
        background: linear-gradient(135deg, #3a2a2a 0%, #4a3a3a 100%);
    }
    
    .message-row.context {
        opacity: 0.6;
        transform: scale(0.95);
    }
    
    .time-badge {
        background: #4ecdc4;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: bold;
        margin-right: 8px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    
    .current-indicator {
        background: #ff6b6b;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin-left: 10px;
        animation: pulse 2s infinite;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .role-patient {
        background: #2a4a6b;
        color: #e8f4f8;
        border-left: 6px solid #4a90e2;
    }
    
    .role-joy {
        background: #2a5a3f;
        color: #e8f5e8;
        border-left: 6px solid #52b788;
    }
    
    .role-agent {
        background: #6b2a5a;
        color: #f8e8f5;
        border-left: 6px solid #b787c7;
    }
    
    .session-info {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #404040;
        color: #e0e0e0;
    }
    
    .metric-card {
        background: #2d2d2d;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border: 1px solid #404040;
        color: #e0e0e0;
    }
    
    .nav-buttons {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 20px 0;
    }
    
    /* Dark theme adjustments */
    .stApp {
        background: #0f0f0f;
        color: #e0e0e0;
    }
    
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4ecdc4, #45b7b8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(78, 205, 196, 0.4);
    }
    
    .prediction-badge {
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        margin-left: 10px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    
    .pred-correct {
        background: #52b788;
        color: white;
    }
    
    .pred-incorrect {
        background: #e63946;
        color: white;
    }
    
    .pred-unknown {
        background: #6c757d;
        color: white;
    }
    
    /* Improved text readability */
    .stMarkdown, .stText {
        color: #e0e0e0;
    }
    
    .stSidebar {
        background: #1a1a1a;
    }
    
    .stSelectbox label {
        color: #e0e0e0;
    }
    
    .stSlider label {
        color: #e0e0e0;
    }
    
    .stMultiSelect label {
        color: #e0e0e0;
    }
    
    .stNumberInput label {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_session_data():
    """Load the latest session detection results."""
    data_dir = Path("data")
    
    # find the most recent session detection file
    session_files = list(data_dir.glob("session_detection_results_*.xlsx"))
    if not session_files:
        st.error("No session detection results found!")
        return None, None
    
    latest_file = max(session_files, key=lambda x: x.stat().st_mtime)
    
    # load the data
    try:
        df = pd.read_excel(latest_file)
        st.success(f"📊 Loaded {len(df)} messages from {latest_file.name}")
        
        # load evaluation results if available
        eval_files = list(data_dir.glob("corrected_evaluation_results_*.json"))
        eval_data = None
        if eval_files:
            latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)
            with open(latest_eval, 'r') as f:
                eval_data = json.load(f)
                
        return df, eval_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def format_time_since(time_str):
    """Format time since last message."""
    if pd.isna(time_str) or time_str == 'FIRST':
        return "FIRST"
    return str(time_str)

def get_role_class(role):
    """Get CSS class for role."""
    if role == 'patient':
        return 'role-patient'
    elif role == 'joy':
        return 'role-joy'
    elif role == 'agent':
        return 'role-agent'
    else:
        return 'role-patient'

def display_chat_session(session_data, eval_data=None, single_message_mode=False, current_message_idx=0):
    """Display a single chat session with styling."""
    if session_data.empty:
        st.warning("No messages in this session")
        return
    
    # session info
    session_id = session_data['session_id'].iloc[0]
    channel_id = session_data['gpt_channel_id'].iloc[0]
    start_time = session_data['created_at'].min()
    end_time = session_data['created_at'].max()
    duration = (end_time - start_time).total_seconds() / 60 if start_time != end_time else 0
    
    # count messages by role
    role_counts = session_data['user_role'].value_counts().to_dict()
    
    # calculate prediction accuracy for this session
    if 'is_session_start' in session_data.columns and 'is_session_start_pred' in session_data.columns:
        session_data_clean = session_data.copy()
        session_data_clean['ground_truth'] = session_data_clean['is_session_start'].apply(
            lambda x: 1 if (pd.notna(x) and x in [1, 1.0, '[START]']) else 0
        )
        session_data_clean['prediction'] = session_data_clean['is_session_start_pred'].fillna(0).astype(int)
        
        correct_preds = (session_data_clean['ground_truth'] == session_data_clean['prediction']).sum()
        total_preds = len(session_data_clean)
        session_accuracy = correct_preds / total_preds if total_preds > 0 else 0
    else:
        session_accuracy = None
    
    # session info card
    st.markdown(f"""
    <div class="session-info">
        <h3>📋 Session Details</h3>
        <p><strong>Session ID:</strong> {session_id}</p>
        <p><strong>Channel:</strong> {channel_id}</p>
        <p><strong>Duration:</strong> {duration:.1f} minutes</p>
        <p><strong>Messages:</strong> {len(session_data)}</p>
        <p><strong>Participants:</strong> {', '.join([f'{role}: {count}' for role, count in role_counts.items()])}</p>
        {f'<p><strong>Prediction Accuracy:</strong> {session_accuracy:.2%}</p>' if session_accuracy is not None else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # chat messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Convert to list for easier indexing
    messages = list(session_data.iterrows())
    
    if single_message_mode:
        # Single message view with context
        buffer_before = 3
        buffer_after = 0  # Set to 0 as requested
        
        start_idx = max(0, current_message_idx - buffer_before)
        end_idx = min(len(messages), current_message_idx + buffer_after + 1)
        
        for i in range(start_idx, end_idx):
            if i >= len(messages):
                break
                
            idx, row = messages[i]
            time_badge = format_time_since(row.get('time_since_last', 'FIRST'))
            role = row['user_role']
            text = str(row['text'])[:1000] if pd.notna(row['text']) else ""
            
            # Determine if this is the current message
            is_current = (i == current_message_idx)
            is_context = not is_current
            
            # Add prediction info if available
            pred_info = ""
            if 'is_session_start_pred' in row and pd.notna(row['is_session_start_pred']):
                pred_val = int(row['is_session_start_pred'])
                if pred_val == 1:
                    pred_info = '<span class="prediction-badge pred-correct">NEW SESSION</span>'
            
            # Determine if prediction is correct
            if 'is_session_start' in row and 'is_session_start_pred' in row:
                ground_truth = 1 if (pd.notna(row['is_session_start']) and row['is_session_start'] in [1, 1.0, '[START]']) else 0
                prediction = int(row['is_session_start_pred']) if pd.notna(row['is_session_start_pred']) else 0
                
                if ground_truth == prediction:
                    pred_class = "pred-correct"
                    pred_text = "✓ CORRECT"
                else:
                    pred_class = "pred-incorrect"
                    pred_text = "✗ INCORRECT"
                
                if prediction == 1:
                    pred_info = f'<span class="prediction-badge {pred_class}">{pred_text} - NEW SESSION</span>'
                elif ground_truth == 1:  # missed a session start
                    pred_info = f'<span class="prediction-badge pred-incorrect">✗ MISSED SESSION START</span>'
            
            role_class = get_role_class(role)
            
            # Add current or context class
            extra_class = ""
            current_indicator = ""
            if is_current:
                extra_class = " current"
                current_indicator = '<span class="current-indicator">← CURRENT</span>'
            elif is_context:
                extra_class = " context"
                
            st.markdown(f"""
            <div class="message-row {role_class}{extra_class}">
                <span class="time-badge">{time_badge}</span>
                <strong>{role}:</strong> {text}
                {current_indicator}
                {pred_info}
            </div>
            """, unsafe_allow_html=True)
            
    else:
        # Full session view
        for idx, row in session_data.iterrows():
            time_badge = format_time_since(row.get('time_since_last', 'FIRST'))
            role = row['user_role']
            text = str(row['text'])[:1000] if pd.notna(row['text']) else ""
            
            # add prediction info if available
            pred_info = ""
            if 'is_session_start_pred' in row and pd.notna(row['is_session_start_pred']):
                pred_val = int(row['is_session_start_pred'])
                if pred_val == 1:
                    pred_info = '<span class="prediction-badge pred-correct">NEW SESSION</span>'
            
            # determine if prediction is correct
            if 'is_session_start' in row and 'is_session_start_pred' in row:
                ground_truth = 1 if (pd.notna(row['is_session_start']) and row['is_session_start'] in [1, 1.0, '[START]']) else 0
                prediction = int(row['is_session_start_pred']) if pd.notna(row['is_session_start_pred']) else 0
                
                if ground_truth == prediction:
                    pred_class = "pred-correct"
                    pred_text = "✓ CORRECT"
                else:
                    pred_class = "pred-incorrect"
                    pred_text = "✗ INCORRECT"
                
                if prediction == 1:
                    pred_info = f'<span class="prediction-badge {pred_class}">{pred_text} - NEW SESSION</span>'
                elif ground_truth == 1:  # missed a session start
                    pred_info = f'<span class="prediction-badge pred-incorrect">✗ MISSED SESSION START</span>'
            
            role_class = get_role_class(role)
            
            st.markdown(f"""
            <div class="message-row {role_class}">
                <span class="time-badge">{time_badge}</span>
                <strong>{role}:</strong> {text}
                {pred_info}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_evaluation_metrics(eval_data):
    """Display evaluation metrics."""
    if not eval_data:
        return
    
    st.markdown("## 📊 Model Performance")
    
    # key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Accuracy</h4>
            <h2>{eval_data['exact_accuracy']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔍 Precision</h4>
            <h2>{eval_data['precision']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎣 Recall</h4>
            <h2>{eval_data['recall']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚖️ F1 Score</h4>
            <h2>{eval_data['f1_score']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # confusion matrix
    cm = eval_data['confusion_matrix']
    fig = go.Figure(data=go.Heatmap(
        z=[[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]],
        x=['Predicted: Continue', 'Predicted: New Session'],
        y=['Actual: Continue', 'Actual: New Session'],
        colorscale='RdYlBu_r',
        text=[[f"True Negatives<br>{cm['tn']}", f"False Positives<br>{cm['fp']}"],
              [f"False Negatives<br>{cm['fn']}", f"True Positives<br>{cm['tp']}"]],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_dark",
        width=500,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit app."""
    st.markdown('<h1 class="main-header">💬 Chat Session Explorer</h1>', unsafe_allow_html=True)
    
    # load data
    df, eval_data = load_session_data()
    if df is None:
        return
    
    # ensure created_at is datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
    
    # sidebar
    st.sidebar.markdown("## 🎮 Navigation")
    
    # get unique sessions
    sessions = df['session_id'].unique()
    total_sessions = len(sessions)
    
    # session selector
    if 'current_session_idx' not in st.session_state:
        st.session_state.current_session_idx = 0
    if 'single_message_mode' not in st.session_state:
        st.session_state.single_message_mode = False
    if 'current_message_idx' not in st.session_state:
        st.session_state.current_message_idx = 0
    
    # view mode toggle
    view_mode = st.sidebar.radio(
        "View Mode:",
        ["📖 Full Session", "🔍 Single Message"],
        index=0 if not st.session_state.single_message_mode else 1
    )
    
    st.session_state.single_message_mode = (view_mode == "🔍 Single Message")
    
    # navigation buttons
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("⏮️ Previous"):
            st.session_state.current_session_idx = (st.session_state.current_session_idx - 1) % total_sessions
            st.session_state.current_message_idx = 0  # Reset to first message
            st.rerun()
    
    with col2:
        if st.button("🎲 Random"):
            st.session_state.current_session_idx = np.random.randint(0, total_sessions)
            st.session_state.current_message_idx = 0  # Reset to first message
            st.rerun()
    
    with col3:
        if st.button("⏭️ Next"):
            st.session_state.current_session_idx = (st.session_state.current_session_idx + 1) % total_sessions
            st.session_state.current_message_idx = 0  # Reset to first message
            st.rerun()
    
    # session index input
    new_idx = st.sidebar.number_input(
        "Jump to session:",
        min_value=0,
        max_value=total_sessions - 1,
        value=st.session_state.current_session_idx,
        key="session_input"
    )
    
    if new_idx != st.session_state.current_session_idx:
        st.session_state.current_session_idx = new_idx
        st.session_state.current_message_idx = 0  # Reset to first message
        st.rerun()
    
    # filter options
    st.sidebar.markdown("## 🔍 Filters")
    
    # role filter
    available_roles = df['user_role'].unique()
    selected_roles = st.sidebar.multiselect(
        "Show messages from:",
        available_roles,
        default=available_roles
    )
    
    # session length filter
    min_messages = st.sidebar.slider(
        "Minimum messages in session:",
        min_value=1,
        max_value=df.groupby('session_id').size().max(),
        value=1
    )
    
    # apply filters
    filtered_df = df[df['user_role'].isin(selected_roles)]
    
    # filter by session length
    session_lengths = filtered_df.groupby('session_id').size()
    valid_sessions = session_lengths[session_lengths >= min_messages].index
    filtered_df = filtered_df[filtered_df['session_id'].isin(valid_sessions)]
    
    # update sessions list
    sessions = filtered_df['session_id'].unique()
    if len(sessions) == 0:
        st.warning("No sessions match the current filters")
        return
    
    # ensure current index is valid
    if st.session_state.current_session_idx >= len(sessions):
        st.session_state.current_session_idx = 0
    
    current_session_id = sessions[st.session_state.current_session_idx]
    session_data = filtered_df[filtered_df['session_id'] == current_session_id].sort_values('created_at')
    
    # single message navigation (only show if in single message mode)
    if st.session_state.single_message_mode and not session_data.empty:
        st.sidebar.markdown("## 💬 Message Navigation")
        
        total_messages = len(session_data)
        
        # message navigation buttons
        col1, col2, col3 = st.sidebar.columns(3)
        
        with col1:
            if st.button("⏪ Prev Msg"):
                st.session_state.current_message_idx = (st.session_state.current_message_idx - 1) % total_messages
                st.rerun()
        
        with col2:
            if st.button("🎲 Random Msg"):
                st.session_state.current_message_idx = np.random.randint(0, total_messages)
                st.rerun()
        
        with col3:
            if st.button("⏩ Next Msg"):
                st.session_state.current_message_idx = (st.session_state.current_message_idx + 1) % total_messages
                st.rerun()
        
        # message index input
        new_msg_idx = st.sidebar.number_input(
            "Jump to message:",
            min_value=0,
            max_value=total_messages - 1,
            value=st.session_state.current_message_idx,
            key="message_input"
        )
        
        if new_msg_idx != st.session_state.current_message_idx:
            st.session_state.current_message_idx = new_msg_idx
            st.rerun()
        
        # ensure message index is valid
        if st.session_state.current_message_idx >= total_messages:
            st.session_state.current_message_idx = 0
        
        # display current session with single message mode
        if st.session_state.single_message_mode:
            st.markdown(f"## Session {st.session_state.current_session_idx + 1} of {len(sessions)} - Message {st.session_state.current_message_idx + 1} of {total_messages}")
        else:
            st.markdown(f"## Session {st.session_state.current_session_idx + 1} of {len(sessions)}")
    else:
        # display current session
        st.markdown(f"## Session {st.session_state.current_session_idx + 1} of {len(sessions)}")
    
    display_chat_session(
        session_data, 
        eval_data, 
        single_message_mode=st.session_state.single_message_mode,
        current_message_idx=st.session_state.current_message_idx
    )
    
    # evaluation metrics
    if eval_data:
        display_evaluation_metrics(eval_data)
    
    # session statistics
    st.markdown("## 📈 Session Statistics")
    
    # session length distribution
    session_lengths = df.groupby('session_id').size()
    fig = px.histogram(
        x=session_lengths,
        nbins=50,
        title="Session Length Distribution",
        template="plotly_dark"
    )
    fig.update_layout(xaxis_title="Messages per Session", yaxis_title="Number of Sessions")
    st.plotly_chart(fig, use_container_width=True)
    
    # role distribution
    role_counts = df['user_role'].value_counts()
    fig = px.pie(
        values=role_counts.values,
        names=role_counts.index,
        title="Message Distribution by Role",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 