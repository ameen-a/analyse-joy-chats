#!/usr/bin/env python3
"""
Simple analysis of patterns in the labelled Joy sessions dataset
"""

import csv
import json
from collections import defaultdict

def load_data():
    """Load the labelled sessions data"""
    data = []
    with open('/Users/ameen/Library/Mobile Documents/com~apple~CloudDocs/Dev/analyse-joy-chats/data/labelled_joy_sessions_with_sessions.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def analyse_patterns(data):
    """Analyse session patterns"""
    
    # group by session_id
    sessions = defaultdict(list)
    for row in data:
        sessions[row['session_id']].append(row)
    
    examples = {
        'no_session_starts': [],
        'many_session_starts': [],
        'joy_introductions': [],
        'prescription_approvals': [],
        'check_ins': [],
        'noisy_examples': [],
        'learning_behaviours': []
    }
    
    for session_id, messages in sessions.items():
        # sort by created_at
        messages = sorted(messages, key=lambda x: x['created_at'])
        
        # count session starts
        session_starts = sum(1 for msg in messages if msg['is_session_start'] == '1.0')
        
        # count agent roles
        roles = defaultdict(int)
        for msg in messages:
            role = msg['agent_role'] if msg['agent_role'] else msg['user_role']
            roles[role] += 1
        
        session_info = {
            'session_id': session_id,
            'total_messages': len(messages),
            'session_starts': session_starts,
            'roles': dict(roles),
            'customer_id': messages[0]['customer_id'],
            'days_span': messages[0]['days_span'],
            'messages': messages
        }
        
        # categorise examples
        if session_starts == 0:
            examples['no_session_starts'].append(session_info)
        elif session_starts >= 3:
            examples['many_session_starts'].append(session_info)
        
        # check for specific Joy patterns
        joy_messages = [msg for msg in messages if msg['agent_role'] == 'joy']
        for msg in joy_messages:
            text = msg['text']
            if "I'm **Coach Joy**" in text:
                examples['joy_introductions'].append(session_info)
                break
            elif "Great news – your prescription" in text and "approved" in text:
                examples['prescription_approvals'].append(session_info)
                break
            elif "Just checking in" in text or "It's been a little while" in text:
                examples['check_ins'].append(session_info)
                break
        
        # identify noisy examples
        unique_roles = len(set(msg['agent_role'] for msg in messages if msg['agent_role']))
        if unique_roles >= 3:
            examples['noisy_examples'].append(session_info)
        
        # identify good learning examples
        has_joy = any(msg['agent_role'] == 'joy' for msg in messages)
        has_patient = any(msg['agent_role'] == 'patient' for msg in messages)
        if has_joy and has_patient:
            examples['learning_behaviours'].append(session_info)
    
    return examples

def print_example(example, title, max_messages=8):
    """Print a formatted example"""
    print(f"\n{'='*70}")
    print(f"EXAMPLE: {title}")
    print(f"{'='*70}")
    print(f"Session ID: {example['session_id']}")
    print(f"Customer ID: {example['customer_id']}")
    print(f"Total Messages: {example['total_messages']}")
    print(f"Session Starts: {example['session_starts']}")
    print(f"Days Span: {example['days_span']}")
    print(f"Agent Roles: {example['roles']}")
    print(f"\nMessages (showing first {max_messages}):")
    print("-" * 70)
    
    for i, msg in enumerate(example['messages'][:max_messages]):
        role = msg['agent_role'] if msg['agent_role'] else msg['user_role']
        session_start = " [SESSION START]" if msg['is_session_start'] == '1.0' else ""
        time_info = f" ({msg['time_since_last']})" if msg['time_since_last'] else ""
        
        print(f"{i+1}. {role.upper()}{session_start}{time_info}:")
        text = msg['text'][:300] + "..." if len(msg['text']) > 300 else msg['text']
        print(f"   {text}")
        print()

def main():
    """Main analysis function"""
    print("Loading data...")
    data = load_data()
    
    print(f"Loaded {len(data)} messages")
    
    print("\nAnalysing patterns...")
    examples = analyse_patterns(data)
    
    print(f"\nPattern Summary:")
    print(f"- Sessions with no session starts: {len(examples['no_session_starts'])}")
    print(f"- Sessions with many session starts (3+): {len(examples['many_session_starts'])}")
    print(f"- Joy introduction examples: {len(examples['joy_introductions'])}")
    print(f"- Prescription approval examples: {len(examples['prescription_approvals'])}")
    print(f"- Check-in examples: {len(examples['check_ins'])}")
    print(f"- Noisy examples (3+ agent types): {len(examples['noisy_examples'])}")
    print(f"- Learning behaviour examples: {len(examples['learning_behaviours'])}")
    
    # Print representative examples
    print("\n" + "="*80)
    print("REPRESENTATIVE EXAMPLES FOR FEW-SHOT LEARNING")
    print("="*80)
    
    # Example 1: No session starts
    if examples['no_session_starts']:
        print_example(examples['no_session_starts'][0], "No Session Starts - Continuation")
    
    # Example 2: Many session starts
    if examples['many_session_starts']:
        print_example(examples['many_session_starts'][0], "Many Session Starts - Multiple Automated Messages")
    
    # Example 3: Joy introduction
    if examples['joy_introductions']:
        print_example(examples['joy_introductions'][0], "Joy Introduction - New Customer Onboarding")
    
    # Example 4: Prescription approval
    if examples['prescription_approvals']:
        print_example(examples['prescription_approvals'][0], "Prescription Approval - Automated Notification")
    
    # Example 5: Check-in
    if examples['check_ins']:
        print_example(examples['check_ins'][0], "Check-in Message - Proactive Engagement")
    
    # Example 6: Noisy dataset
    if examples['noisy_examples']:
        print_example(examples['noisy_examples'][0], "Noisy Example - Multiple Agent Types")
    
    # Example 7: Good learning behaviour
    if examples['learning_behaviours']:
        good_example = None
        for example in examples['learning_behaviours']:
            if example['total_messages'] >= 5 and example['session_starts'] >= 1:
                good_example = example
                break
        
        if good_example:
            print_example(good_example, "Good Learning Example - Joy-Patient Interaction")

if __name__ == "__main__":
    main()