import random
import datasets


def combine_datasets(dataset1, dataset2, dataset2_prob=0.5, shuffle_seed=42):
    """
    Combine two dataset iterators with a given probability.
    
    Args:
        dataset1: Primary dataset iterator
        dataset2: Secondary dataset iterator  
        dataset2_prob: Probability of sampling from dataset2
        shuffle_seed: Random seed for consistent shuffling
        
    Yields:
        Items from combined datasets
    """
    random.seed(shuffle_seed)
    iter_dataset2 = iter(dataset2) if dataset2 is not None else None
    
    for item in dataset1:
        yield item
        
        # Occasionally yield from dataset2
        if random.random() < dataset2_prob and iter_dataset2 is not None:
            try:
                item2 = next(iter_dataset2)
                yield item2
            except StopIteration:
                iter_dataset2 = None

def load_jsonl_dataset(filepath, max_samples=None):
    """
    Load a JSONL file as a dataset iterator.
    
    Args:
        filepath: Path to JSONL file
        max_samples: Maximum number of samples to load
        
    Yields:
        Parsed JSON objects from each line
    """
    import json
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples is not None and count >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                if isinstance(data, dict):
                    for field_names in ['messages', 'conversation', 'prompt', 'text']:
                        if field_names in data:
                            data = data[field_names]
                            break
                # If not a list, make it a list
                if not isinstance(data, list):
                    data = [data] if data else []
                assert isinstance(data, list) and all(isinstance(item, str) for item in data)
                yield data
                count += 1
            except json.JSONDecodeError:
                continue

class CalibrationDataLoader:
    """
    DataLoader for conversations from various sources for calibration.
    Supports HuggingFace datasets and JSONL files.
    """
    
    def __init__(self, 
                 dataset_name='allenai/WildChat-1M',
                 max_samples=None,
                 conversation_length_limit=1500,
                 streaming=False,
                 dataset2_name='HuggingFaceH4/ultrachat_200k',
                 dataset2_prob=0.5,
                 multiplier=1,
                 default_split='train'):
        """
        Initialize conversation dataloader.
        
        Args:
            dataset_name: HuggingFace dataset name or path to .jsonl file
            split: Dataset split to use (ignored for JSONL)
            max_samples: Maximum number of samples to load (None for all)
            conversation_length_limit: Max total character length per conversation
            streaming: Whether to use streaming mode (ignored for JSONL)
            dataset2_name: Secondary dataset name or JSONL path
            dataset2_prob: Probability of sampling from dataset2 in addition to each item from dataset1
            multiplier: Subsample rate for dataset1
        """
        self.dataset_name = dataset_name
        self.dataset2_name = dataset2_name
        self.max_samples = max_samples
        self.conversation_length_limit = conversation_length_limit
        self.streaming = streaming
        self.dataset2_prob = dataset2_prob
        self._dataset = None
        self._dataset2 = None
        self.multiplier = multiplier
        self.default_split = default_split
    
    def _load_single_dataset(self, dataset_name):
        if dataset_name is None:
            return None
        
        if dataset_name.endswith('.jsonl'):
            print(f"Loading JSONL dataset: {dataset_name}")
            return load_jsonl_dataset(dataset_name, self.max_samples)
        else:
            print(f"Loading HuggingFace dataset: {dataset_name}")
            split = self.default_split
            if 'ultrachat' in dataset_name:
                split = 'train_sft'
            elif 'wildchat' in dataset_name:
                split = 'train'
            return datasets.load_dataset(dataset_name, split=split, streaming=self.streaming)

    def _load_dataset(self):
        """Load the datasets if not already loaded."""
        if self._dataset is None:
            self._dataset = self._load_single_dataset(self.dataset_name)
        
        if self._dataset2 is None and self.dataset2_name is not None:
            self._dataset2 = self._load_single_dataset(self.dataset2_name)
        
        return self._dataset, self._dataset2
    
    def _process_conversation(self, conversation_data):
        """
        Process a single conversation.
        
        Args:
            conversation_data: Raw conversation data
            
        Returns:
            list: Processed conversation as list of strings
        """
        # Handle different conversation formats
        if isinstance(conversation_data, list):
            cv = conversation_data
        else:
            cv = conversation_data.get('conversation', conversation_data.get('messages', None))
        
        processed_conversation = []
        total_len = 0
        
        for message in cv:
            budget = self.conversation_length_limit - total_len
            if budget <= 0:
                break
                
            # Extract content from message
            if isinstance(message, dict):
                content = message.get('content', str(message))
            else:
                content = str(message)
                
            content = content[:budget]
            processed_conversation.append(content)
            total_len += len(content)
            
        return processed_conversation
    
    def __iter__(self):
        """Make the dataloader iterable."""
        dataset, dataset2 = self._load_dataset()
        
        # Create combined dataset if dataset2 exists
        if dataset2 is not None:
            combined_dataset = combine_datasets(dataset, dataset2, self.dataset2_prob)
        else:
            combined_dataset = dataset
        
        count = 0
        chunk_size = 2000
        buffer = []
        
        for item in combined_dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break
                
            processed_conv = self._process_conversation(item)
            if len(processed_conv) > 0:  # Only yield non-empty conversations
                buffer.append(processed_conv)
                
                if len(buffer) >= self.multiplier * chunk_size:
                    # Shuffle with fixed seed
                    random.seed(42)
                    random.shuffle(buffer)
                    # Keep only first chunk_size items
                    buffer = buffer[:chunk_size]
                    # Yield all items in buffer
                    for conv in buffer:
                        yield conv
                        count += 1
                        if self.max_samples is not None and count >= self.max_samples:
                            return
                    buffer = []
        
        # Yield remaining items in buffer
        while len(buffer) and (self.max_samples is None or count < self.max_samples):
            yield buffer.pop(0)
            count += 1
    
    def __len__(self):
        """Return length if max_samples is specified, otherwise not implemented."""
        if self.max_samples is not None:
            return self.max_samples
        else:
            raise NotImplementedError("Length not available for streaming datasets without max_samples")
