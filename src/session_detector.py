import os
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import time
import warnings
from datetime import datetime
from pydantic import BaseModel

from config import Config
from prompts import SYSTEM_PROMPT, build_prompt, BATCH_SYSTEM_PROMPT, build_batch_prompt
from utils import get_message_window, assign_session_ids

class SessionClassification(BaseModel):
    """Structured output for session boundary classification."""
    is_new_session: int  # 0 or 1

class BatchSessionClassification(BaseModel):
    """Structured output for batch session boundary classification."""
    session_start_indices: List[int]  # list of 0-based indices that are session starts

class SessionDetector:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.classifications: Dict[str, int] = {}
        
        # prompt saving setup
        self.saved_prompts_count = 0
        self.prompts_folder = None
        if self.config.save_prompts_count > 0:
            self._setup_prompts_folder()
    
    def _setup_prompts_folder(self):
        """Create folder for saving prompts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.prompts_folder = f"../data/prompts_{timestamp}"
        os.makedirs(self.prompts_folder, exist_ok=True)
        print(f"Created prompts folder: {self.prompts_folder}")
    
    def _save_prompt_as_markdown(self, system_prompt: str, user_prompt: str, message_id: str, classification_result: int):
        """Save system and user prompts as markdown file with LLM classification result."""
        if not self.prompts_folder:
            return
        
        filename = f"prompt_{self.saved_prompts_count + 1:03d}_{message_id}.md"
        filepath = os.path.join(self.prompts_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Prompt for Message ID: {message_id}\n\n")
            f.write(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## System Prompt\n\n")
            f.write(f"```\n{system_prompt}\n```\n\n")
            f.write("## User Prompt\n\n")
            f.write(f"```\n{user_prompt}\n```\n\n")
            f.write("## LLM Classification Result\n\n")
            f.write(f"**Classification:** {classification_result}\n\n")
            
            # interpret result based on whether it's batch or individual classification
            if message_id.startswith('batch_'):
                f.write(f"**Interpretation:** Found {classification_result} session start(s) in batch\n")
            else:
                f.write(f"**Interpretation:** {'New session start' if classification_result == 1 else 'Continue existing session'}\n")
        
        self.saved_prompts_count += 1    
        
    def classify_message(
        self, 
        message_window: List[Dict],
        current_message: Dict,
        additional_examples: str = ""
    ) -> int:
        """Classify if current message starts a new session using structured outputs."""
        prompt = build_prompt(message_window, current_message, additional_examples)
        system_prompt = SYSTEM_PROMPT + "\n\nRespond with 0 if the message continues the existing session, or 1 if it starts a new session."
        
        # determine if we should save this prompt (before classification)
        should_save_prompt = (self.config.save_prompts_count > 0 and 
                              self.saved_prompts_count < self.config.save_prompts_count and
                              random.random() < 0.1)  # 10% chance to save each prompt
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=SessionClassification,
                    temperature=self.config.temperature,
                    max_tokens=50
                )
                
                result = response.choices[0].message.parsed
                classification_result = result.is_new_session if result else 0
                
                # save prompt with classification result after successful classification
                if should_save_prompt:
                    message_id = current_message.get('gpt_stream_id', 'unknown')
                    self._save_prompt_as_markdown(system_prompt, prompt, message_id, classification_result)
                
                return classification_result
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    print(f"Error classifying message: {e}")
                    # save prompt with error result if we were going to save it
                    if should_save_prompt:
                        message_id = current_message.get('gpt_stream_id', 'unknown')
                        self._save_prompt_as_markdown(system_prompt, prompt, message_id, 0)  # default to 0 on error
                    return 0
                time.sleep(2 ** attempt)
    
    def classify_message_batch(
        self,
        message_batch: List[Dict],
        window_start_idx: int = 0
    ) -> List[int]:
        """Classify a batch of messages using structured outputs."""
        prompt = build_batch_prompt(message_batch, window_start_idx)
        system_prompt = BATCH_SYSTEM_PROMPT
        
        # determine if we should save this prompt (before classification)
        should_save_prompt = (self.config.save_prompts_count > 0 and 
                              self.saved_prompts_count < self.config.save_prompts_count and
                              random.random() < 0.1)  # 10% chance to save each prompt
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=BatchSessionClassification,
                    temperature=self.config.temperature,
                    max_tokens=200
                )
                
                result = response.choices[0].message.parsed
                if result:
                    # validate indices are within bounds
                    valid_indices = [idx for idx in result.session_start_indices 
                                   if 0 <= idx < len(message_batch)]
                    
                    # save prompt with classification result after successful classification
                    if should_save_prompt:
                        batch_id = f"batch_{window_start_idx}_{len(message_batch)}"
                        self._save_prompt_as_markdown(system_prompt, prompt, batch_id, len(valid_indices))
                    
                    return valid_indices
                else:
                    # save prompt with no results if we were going to save it
                    if should_save_prompt:
                        batch_id = f"batch_{window_start_idx}_{len(message_batch)}"
                        self._save_prompt_as_markdown(system_prompt, prompt, batch_id, 0)
                    return []
                    
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    print(f"Error classifying batch: {e}")
                    # save prompt with error result if we were going to save it
                    if should_save_prompt:
                        batch_id = f"batch_{window_start_idx}_{len(message_batch)}"
                        self._save_prompt_as_markdown(system_prompt, prompt, batch_id, 0)  # default to 0 on error
                    return []
                time.sleep(2 ** attempt)
        
        return []

    def process_dataframe_batch(
        self,
        df: pd.DataFrame,
        additional_examples: str = ""
    ) -> pd.DataFrame:
        """Process entire dataframe using batch classification."""
        # first, filter users with minimum messages to ensure we only consider valid users
        user_message_counts = df.groupby('customer_id').size()
        valid_users = user_message_counts[user_message_counts >= self.config.min_messages_threshold].index
        df_valid_users = df[df['customer_id'].isin(valid_users)].copy()
        
        print(f"Filtered to {len(df_valid_users)} messages from {len(valid_users)} users (minimum {self.config.min_messages_threshold} messages per user)")
        
        # ensure we get AT LEAST the target dataset size (not just close to it)
        if self.config.max_dataset_size and len(df_valid_users) > self.config.max_dataset_size:
            # sort by customer and time to ensure consistent sampling
            df_sorted = df_valid_users.sort_values(['customer_id', 'created_at'])
            
            # get user message counts for optimisation
            user_msg_counts = df_sorted.groupby('customer_id').size().to_dict()
            users_list = list(user_msg_counts.keys())
            
            # find the combination that gets us AT LEAST the target size
            # keep adding users until we reach at least the target
            selected_users = []
            current_count = 0
            
            for user in users_list:
                selected_users.append(user)
                current_count += user_msg_counts[user]
                
                # stop once we have AT LEAST the target number of messages
                if current_count >= self.config.max_dataset_size:
                    break
            
            # if we still don't have enough, add more users
            if current_count < self.config.max_dataset_size:
                remaining_users = [u for u in users_list if u not in selected_users]
                for user in remaining_users:
                    selected_users.append(user)
                    current_count += user_msg_counts[user]
                    if current_count >= self.config.max_dataset_size:
                        break
            
            # filter to only include complete conversations for selected users
            df_filtered = df_sorted[df_sorted['customer_id'].isin(selected_users)]
            
            print(f"Selected {len(df_filtered)} messages from {len(selected_users)} users (target: AT LEAST {self.config.max_dataset_size})")
            print(f"Included complete conversations to ensure minimum target is met")
        else:
            df_filtered = df_valid_users
        
        # sort by customer and time to identify first N messages per customer
        df_filtered = df_filtered.sort_values(['customer_id', 'created_at'])
        
        # mark first N messages per customer as session starts (1) 
        def mark_first_messages(group):
            group = group.copy()
            # mark first auto_session_start_count messages as session starts
            for i in range(min(self.config.auto_session_start_count, len(group))):
                self.classifications[group.iloc[i]['gpt_stream_id']] = 1
            return group
        
        # suppress pandas FutureWarning for groupby apply
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df_filtered = df_filtered.groupby('customer_id', group_keys=False).apply(mark_first_messages)
        
        # now process remaining messages using batch classification
        messages_to_classify = []
        for customer_id in df_filtered['customer_id'].unique():
            customer_msgs = df_filtered[df_filtered['customer_id'] == customer_id].sort_values('created_at')
            if len(customer_msgs) > self.config.auto_session_start_count:
                # collect messages that need batch classification
                remaining_msgs = customer_msgs.iloc[self.config.auto_session_start_count:]
                messages_to_classify.extend(remaining_msgs.index.tolist())
        
        df_to_classify = df_filtered.loc[messages_to_classify] if messages_to_classify else pd.DataFrame()
        
        print(f"Automatically marked first {self.config.auto_session_start_count} messages per customer as session starts")
        print(f"Processing {len(df_to_classify)} additional messages for batch classification from {df_filtered['customer_id'].nunique()} users")
        
        # sort by channel and time for batch classification
        df_to_classify = df_to_classify.sort_values(['gpt_channel_id', 'created_at'])
        
        # group messages by channel for batch processing
        processed_messages = set()
        channels = df_to_classify['gpt_channel_id'].unique()
        
        # calculate total batches for overall progress
        total_batches = 0
        for channel_id in channels:
            channel_messages = df_to_classify[df_to_classify['gpt_channel_id'] == channel_id]
            total_messages = len(channel_messages)
            if total_messages > 0:
                batches_in_channel = max(1, (total_messages - self.config.batch_window_size) // (self.config.batch_window_size - self.config.batch_overlap) + 1)
                total_batches += batches_in_channel
        
        # overall progress bar
        with tqdm(total=total_batches, desc="Processing all channels", unit="batch", position=0) as overall_pbar:
            for channel_id in channels:
                channel_messages = df_to_classify[df_to_classify['gpt_channel_id'] == channel_id].copy()
                
                # process this channel with sliding windows
                self._process_channel_batch(channel_messages, processed_messages, overall_pbar)
        
        # ensure all messages have classifications (default to 0 if not processed)
        for idx, row in df_to_classify.iterrows():
            if row['gpt_stream_id'] not in self.classifications:
                self.classifications[row['gpt_stream_id']] = 0
        
        # assign session IDs to the filtered dataset
        result_df = assign_session_ids(df_filtered, self.classifications)
        
        print(f"\nDetected {result_df['session_id'].nunique()} sessions")
        print(f"Average messages per session: {len(result_df) / result_df['session_id'].nunique():.1f}")
        
        return result_df

    def _process_channel_batch(self, channel_messages: pd.DataFrame, processed_messages: set, overall_pbar):
        """Process a single channel using sliding windows with overlap."""
        messages_list = channel_messages.to_dict('records')
        total_messages = len(messages_list)
        
        if total_messages == 0:
            return
        
        # collect all session start decisions from overlapping windows
        session_start_votes = {}  # message_id -> count of votes for session start
        
        # sliding window with overlap
        window_size = self.config.batch_window_size
        overlap = self.config.batch_overlap
        step = window_size - overlap
        
        channel_id = channel_messages.iloc[0]['gpt_channel_id']
        batches_in_channel = max(1, (total_messages - window_size) // step + 1)
        
        with tqdm(total=batches_in_channel, 
                 desc=f"Channel {channel_id} ({total_messages} msgs)", 
                 unit="batch", position=1, leave=False) as channel_pbar:
            
            for start_idx in range(0, total_messages, step):
                end_idx = min(start_idx + window_size, total_messages)
                window_messages = messages_list[start_idx:end_idx]
                
                # skip if window is too small
                if len(window_messages) < 2:
                    break
                
                # classify batch
                session_start_indices = self.classify_message_batch(window_messages, start_idx)
                
                # record votes
                for rel_idx in session_start_indices:
                    abs_idx = start_idx + rel_idx
                    if abs_idx < total_messages:
                        message_id = messages_list[abs_idx]['gpt_stream_id']
                        session_start_votes[message_id] = session_start_votes.get(message_id, 0) + 1
                
                channel_pbar.update(1)
                overall_pbar.update(1)
                
                # if we've processed all messages, break
                if end_idx >= total_messages:
                    break
        
        # apply union approach - any message that got votes becomes a session start
        # but prefer continuation (recall is more important, so we're conservative)
        for message in messages_list:
            message_id = message['gpt_stream_id']
            
            if message_id in session_start_votes:
                # default to continue (0) unless we have strong evidence for session start
                self.classifications[message_id] = 1 if session_start_votes[message_id] >= 1 else 0
            else:
                # no votes for session start, so continue
                self.classifications[message_id] = 0
            
            processed_messages.add(message_id)
    
    def process_dataframe(
        self, 
        df: pd.DataFrame,
        additional_examples: str = ""
    ) -> pd.DataFrame:
        """Process entire dataframe and classify session boundaries."""
        # first, filter users with minimum messages to ensure we only consider valid users
        user_message_counts = df.groupby('customer_id').size()
        valid_users = user_message_counts[user_message_counts >= self.config.min_messages_threshold].index
        df_valid_users = df[df['customer_id'].isin(valid_users)].copy()
        
        print(f"Filtered to {len(df_valid_users)} messages from {len(valid_users)} users (minimum {self.config.min_messages_threshold} messages per user)")
        
        # ensure we get AT LEAST the target dataset size (not just close to it)
        if self.config.max_dataset_size and len(df_valid_users) > self.config.max_dataset_size:
            # sort by customer and time to ensure consistent sampling
            df_sorted = df_valid_users.sort_values(['customer_id', 'created_at'])
            
            # get user message counts for optimisation
            user_msg_counts = df_sorted.groupby('customer_id').size().to_dict()
            users_list = list(user_msg_counts.keys())
            
            # find the combination that gets us AT LEAST the target size
            # keep adding users until we reach at least the target
            selected_users = []
            current_count = 0
            
            for user in users_list:
                selected_users.append(user)
                current_count += user_msg_counts[user]
                
                # stop once we have AT LEAST the target number of messages
                if current_count >= self.config.max_dataset_size:
                    break
            
            # if we still don't have enough, add more users
            if current_count < self.config.max_dataset_size:
                remaining_users = [u for u in users_list if u not in selected_users]
                for user in remaining_users:
                    selected_users.append(user)
                    current_count += user_msg_counts[user]
                    if current_count >= self.config.max_dataset_size:
                        break
            
            # filter to only include complete conversations for selected users
            df_filtered = df_sorted[df_sorted['customer_id'].isin(selected_users)]
            
            print(f"Selected {len(df_filtered)} messages from {len(selected_users)} users (target: AT LEAST {self.config.max_dataset_size})")
            print(f"Included complete conversations to ensure minimum target is met")
        else:
            df_filtered = df_valid_users
        
        # sort by customer and time to identify first 4 messages per customer
        df_filtered = df_filtered.sort_values(['customer_id', 'created_at'])
        
        # mark first 4 messages per customer as session starts (1)
        def mark_first_messages(group):
            group = group.copy()
            # mark first 4 messages as session starts
            for i in range(min(4, len(group))):
                self.classifications[group.iloc[i]['gpt_stream_id']] = 1
            return group
        
        # suppress pandas FutureWarning for groupby apply
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df_filtered = df_filtered.groupby('customer_id', group_keys=False).apply(mark_first_messages)
        
        # now process remaining messages (after first 4 per customer) for classification
        messages_to_classify = []
        for customer_id in df_filtered['customer_id'].unique():
            customer_msgs = df_filtered[df_filtered['customer_id'] == customer_id].sort_values('created_at')
            if len(customer_msgs) > 4:
                messages_to_classify.extend(customer_msgs.iloc[4:].index.tolist())
        
        df_to_classify = df_filtered.loc[messages_to_classify] if messages_to_classify else pd.DataFrame()
        
        print(f"Automatically marked first 4 messages per customer as session starts")
        print(f"Processing {len(df_to_classify)} additional messages for classification from {df_filtered['customer_id'].nunique()} users")
        
        # sort by channel and time for classification
        df_to_classify = df_to_classify.sort_values(['gpt_channel_id', 'created_at'])
        
        # classify the remaining messages
        with tqdm(total=len(df_to_classify), desc="Classifying messages", unit="msg") as pbar:
            for i in range(len(df_to_classify)):
                row = df_to_classify.iloc[i]
                
                # skip if already labelled as [START]
                if pd.notna(row.get('is_session_start')) and str(row['is_session_start']) == '[START]':
                    self.classifications[row['gpt_stream_id']] = 1
                    pbar.update(1)
                    continue
                
                # get message window from the full filtered dataset
                # find the position of this message in the filtered dataset
                current_idx_in_filtered = df_filtered[df_filtered['gpt_stream_id'] == row['gpt_stream_id']].index[0]
                filtered_position = df_filtered.index.get_loc(current_idx_in_filtered)
                
                message_window = get_message_window(
                    df_filtered, 
                    filtered_position,
                    row['gpt_channel_id'],
                    self.config.sliding_window_size,
                    self.classifications  # pass classifications so LLM can see previous decisions
                )
                
                # first message in channel is always a new session
                if len(message_window) == 1:
                    self.classifications[row['gpt_stream_id']] = 1
                else:
                    classification = self.classify_message(
                        message_window,
                        message_window[-1],  # current message is last in window
                        additional_examples
                    )
                    self.classifications[row['gpt_stream_id']] = classification
                
                pbar.update(1)
        
        # assign session IDs to the filtered dataset
        result_df = assign_session_ids(df_filtered, self.classifications)
        
        print(f"\nDetected {result_df['session_id'].nunique()} sessions")
        print(f"Average messages per session: {len(result_df) / result_df['session_id'].nunique():.1f}")
        
        # prompt saving summary
        if self.config.save_prompts_count > 0:
            print(f"Saved {self.saved_prompts_count} prompts to {self.prompts_folder}")
        
        return result_df

def main():
    # track total execution time
    start_time = time.time()
    
    # load data
    data_load_start = time.time()
    df = pd.read_excel('../data/labelled_joy_sessions.xlsx')
    df['created_at'] = pd.to_datetime(df['created_at'])
    data_load_time = time.time() - data_load_start
    
    print(f"Loaded {len(df)} messages from {df['customer_id'].nunique()} users")
    
    # initialise detector
    detector = SessionDetector()
    
    # add any additional examples here
    additional_examples = """
    # Add your specific examples here if needed
    """
    
    # process using batch processing (new optimized approach)
    processing_start = time.time()
    result_df = detector.process_dataframe_batch(df, additional_examples)
    processing_time = time.time() - processing_start
    
    # save results
    save_start = time.time()
    output_path = f'../data/session_detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    result_df.to_excel(output_path, index=False)
    save_time = time.time() - save_start
    print(f"\nResults saved to {output_path}")
    
    # if ground truth labels exist, calculate comprehensive metrics
    evaluation_time = 0
    if 'is_session_start' in df.columns:
        eval_start = time.time()
        from evaluation_metrics import SessionBoundaryEvaluator
        
        # now all messages have predictions (no NaN values due to fillna(0))
        # convert ground truth labels: 1.0 or '[START]' -> 1, everything else (including NaN) -> 0
        ground_truth = result_df['is_session_start'].apply(
            lambda x: 1 if (pd.notna(x) and x in [1, 1.0, '[START]']) else 0
        ).values
        
        predictions = result_df['is_session_start_pred'].astype(int).values
        
        print(f"\nEvaluating on {len(predictions)} messages (all messages in sample)")
        print(f"Ground truth distribution: {np.sum(ground_truth)} positives, {len(ground_truth) - np.sum(ground_truth)} negatives")
        print(f"Prediction distribution: {np.sum(predictions)} positives, {len(predictions) - np.sum(predictions)} negatives")
        
        # run comprehensive evaluation
        evaluator = SessionBoundaryEvaluator()
        evaluation_results = evaluator.evaluate_comprehensive(predictions, ground_truth)
        evaluator.print_evaluation_report(evaluation_results, "SessionDetector")
        evaluation_time = time.time() - eval_start
        
        # calculate total execution time
        total_time = time.time() - start_time
        
        # add timing information to evaluation results
        evaluation_results['timing'] = {
            'total_time_seconds': total_time,
            'data_load_time_seconds': data_load_time,
            'processing_time_seconds': processing_time,
            'save_time_seconds': save_time,
            'evaluation_time_seconds': evaluation_time,
            'total_time_formatted': f"{total_time:.2f}s",
            'data_load_time_formatted': f"{data_load_time:.2f}s",
            'processing_time_formatted': f"{processing_time:.2f}s",
            'save_time_formatted': f"{save_time:.2f}s",
            'evaluation_time_formatted': f"{evaluation_time:.2f}s"
        }
        
        # save evaluation results  
        import json
        eval_path = f'../data/corrected_evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"\nDetailed evaluation results saved to {eval_path}")
        
        # print timing information to terminal
        print(f"\n{'='*50}")
        print(f"EXECUTION TIMING")
        print(f"{'='*50}")
        print(f"Data loading:     {data_load_time:.2f}s")
        print(f"Processing:       {processing_time:.2f}s")
        print(f"Saving results:   {save_time:.2f}s")
        print(f"Evaluation:       {evaluation_time:.2f}s")
        print(f"{'='*50}")
        print(f"Total runtime:    {total_time:.2f}s")
        print(f"{'='*50}")
    else:
        # calculate total execution time (no evaluation case)
        total_time = time.time() - start_time
        
        # print timing information to terminal
        print(f"\n{'='*50}")
        print(f"EXECUTION TIMING")
        print(f"{'='*50}")
        print(f"Data loading:     {data_load_time:.2f}s")
        print(f"Processing:       {processing_time:.2f}s")
        print(f"Saving results:   {save_time:.2f}s")
        print(f"{'='*50}")
        print(f"Total runtime:    {total_time:.2f}s")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()