from dataclasses import dataclass

@dataclass
class Config:
    min_messages_threshold: int = 1  # minimum conversation length for batch processing
    sliding_window_size: int = 6  # context window for individual message classification (legacy)
    
    # batch processing parameters
    batch_window_size: int = 100       # messages per batch window
    batch_overlap: int = 5             # overlap between consecutive windows
    auto_session_start_count: int = 4  # first N messages automatically marked as session starts
    
    model: str = "gpt-4.1-2025-04-14"
    # model: str = "gpt-4.1-mini-2025-04-14"
    # model: str = "gpt-4o"
    # model: str = "o4-mini-2025-04-16"
    temperature: float = 0.1
    max_retries: int = 3
    max_dataset_size: int = None  # minimum target dataset size (will include complete conversations to reach at least this many messages)
    save_prompts_count: int = 5  # number of random prompts to save as markdown for debugging