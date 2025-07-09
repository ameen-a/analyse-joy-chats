#!/usr/bin/env python3
"""
Script to analyse patterns in the labelled Joy sessions dataset
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import re

def load_data():
    """Load the labelled sessions data"""
    df = pd.read_csv('/Users/ameen/Library/Mobile Documents/com~apple~CloudDocs/Dev/analyse-joy-chats/data/labelled_joy_sessions_with_sessions.csv')
    return df

def analyse_session_patterns(df):
    """Analyse session start patterns in the data"""
    
    # group by session_id to analyse batches
    session_groups = df.groupby('session_id')
    
    patterns = {
        'no_session_starts': [],
        'many_session_starts': [],
        'joy_introductions': [],
        'prescription_approvals': [],
        'check_ins': [],
        'noisy_examples': [],
        'learning_behaviours': []
    }
    
    for session_id, group in session_groups:
        # sort by created_at to get chronological order
        group = group.sort_values('created_at').reset_index(drop=True)
        
        # count session starts
        session_starts = group['is_session_start'].sum()
        
        # store some metrics
        group_info = {
            'session_id': session_id,
            'total_messages': len(group),
            'session_starts': session_starts,
            'roles': group['agent_role'].value_counts().to_dict(),
            'customer_id': group['customer_id'].iloc[0],
            'days_span': group['days_span'].iloc[0],
            'messages': group[['agent_role', 'user_role', 'text', 'is_session_start', 'time_since_last', 'created_at']].to_dict('records')
        }
        
        # categorise examples
        if session_starts == 0:
            patterns['no_session_starts'].append(group_info)
        elif session_starts >= 3:
            patterns['many_session_starts'].append(group_info)
        
        # check for specific Joy message patterns
        joy_messages = group[group['agent_role'] == 'joy']['text'].tolist()
        for msg in joy_messages:
            if isinstance(msg, str):
                if "I'm **Coach Joy**" in msg:
                    patterns['joy_introductions'].append(group_info)
                    break
                elif "Great news – your prescription" in msg and "approved" in msg:
                    patterns['prescription_approvals'].append(group_info)
                    break
                elif "Just checking in" in msg or "It's been a little while" in msg:
                    patterns['check_ins'].append(group_info)
                    break
        
        # identify noisy examples (lots of different agent types)
        unique_roles = len(group['agent_role'].unique())
        if unique_roles >= 3:
            patterns['noisy_examples'].append(group_info)
        
        # identify good learning examples (mix of joy and patient interactions)
        if 'joy' in group['agent_role'].values and 'patient' in group['agent_role'].values:
            patterns['learning_behaviours'].append(group_info)
    
    return patterns

def print_example(example, title, max_messages=10):
    """Print a formatted example"""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {title}")
    print(f"{'='*60}")
    print(f"Session ID: {example['session_id']}")
    print(f"Customer ID: {example['customer_id']}")
    print(f"Total Messages: {example['total_messages']}")
    print(f"Session Starts: {example['session_starts']}")
    print(f"Days Span: {example['days_span']}")
    print(f"Agent Roles: {example['roles']}")
    print(f"\nMessages (showing first {max_messages}):")
    print("-" * 60)
    
    for i, msg in enumerate(example['messages'][:max_messages]):
        role = msg['agent_role'] if msg['agent_role'] else msg['user_role']
        session_start = " [SESSION START]" if msg['is_session_start'] == 1.0 else ""
        time_info = f" ({msg['time_since_last']})" if msg['time_since_last'] else ""
        
        print(f"{i+1}. {role.upper()}{session_start}{time_info}:")
        text = msg['text'][:200] + "..." if len(str(msg['text'])) > 200 else msg['text']
        print(f"   {text}")
        print()

def main():
    """Main analysis function"""
    print("Loading data...")
    df = load_data()
    
    print(f"Loaded {len(df)} messages")
    print(f"Unique sessions: {df['session_id'].nunique()}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    
    print("\nAnalysing patterns...")
    patterns = analyse_session_patterns(df)
    
    print(f"\nPattern Summary:")
    print(f"- Sessions with no session starts: {len(patterns['no_session_starts'])}")
    print(f"- Sessions with many session starts (3+): {len(patterns['many_session_starts'])}")
    print(f"- Joy introduction examples: {len(patterns['joy_introductions'])}")
    print(f"- Prescription approval examples: {len(patterns['prescription_approvals'])}")
    print(f"- Check-in examples: {len(patterns['check_ins'])}")
    print(f"- Noisy examples (3+ agent types): {len(patterns['noisy_examples'])}")
    print(f"- Learning behaviour examples: {len(patterns['learning_behaviours'])}")
    
    # Print representative examples
    print("\n" + "="*80)
    print("REPRESENTATIVE EXAMPLES FOR FEW-SHOT LEARNING")
    print("="*80)
    
    # Example 1: No session starts
    if patterns['no_session_starts']:
        print_example(patterns['no_session_starts'][0], "No Session Starts - Continuation")
    
    # Example 2: Many session starts
    if patterns['many_session_starts']:
        print_example(patterns['many_session_starts'][0], "Many Session Starts - Multiple Automated Messages")
    
    # Example 3: Joy introduction
    if patterns['joy_introductions']:
        print_example(patterns['joy_introductions'][0], "Joy Introduction - New Customer Onboarding")
    
    # Example 4: Prescription approval
    if patterns['prescription_approvals']:
        print_example(patterns['prescription_approvals'][0], "Prescription Approval - Automated Notification")
    
    # Example 5: Check-in
    if patterns['check_ins']:
        print_example(patterns['check_ins'][0], "Check-in Message - Proactive Engagement")
    
    # Example 6: Noisy dataset
    if patterns['noisy_examples']:
        print_example(patterns['noisy_examples'][0], "Noisy Example - Multiple Agent Types")
    
    # Example 7: Good learning behaviour
    if patterns['learning_behaviours']:
        good_example = None
        for example in patterns['learning_behaviours']:
            if example['total_messages'] >= 5 and example['session_starts'] >= 1:
                good_example = example
                break
        
        if good_example:
            print_example(good_example, "Good Learning Example - Joy-Patient Interaction")

if __name__ == "__main__":
    main()