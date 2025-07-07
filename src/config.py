from dataclasses import dataclass

@dataclass
class Config:
    min_messages_threshold: int = 1  # reduced from 10 to include more users
    sliding_window_size: int = 6
    model: str = "gpt-4o"
    # model: str = "o4-mini-2025-04-16"
    temperature: float = 0.1
    max_retries: int = 3
    batch_size: int = 10  # progress bar
    max_dataset_size: int = None  # minimum target dataset size (will include complete conversations to reach at least this many messages)
    save_prompts_count: int = 100  # number of random prompts to save as markdown for debugging