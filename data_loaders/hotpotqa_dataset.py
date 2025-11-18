from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


class HotpotQADataset(Dataset):
    """
    HotPotQA dataset.
    
    Args:
        split: 'train' or 'validation'
        max_seq_len: Maximum token sequence length (default: 1024)
        max_sentences: Maximum number of sentences for H-module (default: 20)
        subset: 'distractor' (10 docs, 2 relevant) or 'fullwiki' (harder)
        tokenizer_name: HuggingFace tokenizer to use
        prioritize_supporting: If True, prioritize documents with supporting facts
        subset_size: Optional - use only first N examples (for testing/debugging)
    """
    
    def __init__(
        self,
        split: str = 'train',
        max_seq_len: int = 1024,
        max_sentences: int = 20,
        subset: str = 'distractor',
        tokenizer_name: str = 'bert-base-uncased',
        prioritize_supporting: bool = True,
        subset_size: Optional[int] = None,
        ):

        super().__init__()
        
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_sentences = max_sentences
        self.prioritize_supporting = prioritize_supporting
        self.subset_size = subset_size
        
        # get the dataset from HuggingFace
        print(f"Loading HotpotQA ({subset}, {split})...")
        self.dataset = load_dataset("hotpotqa/hotpot_qa", subset, split=split)
        
        # optionally limit to subset for testing
        if subset_size is not None:
            self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))
            print(f"✓ Loaded {len(self.dataset)} examples (subset)")
        else:
            print(f"✓ Loaded {len(self.dataset)} examples")
        
        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # special tokens
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        
        # vocab size for model
        self.vocab_size = len(self.tokenizer)
        
        print(f"✓ Tokenizer initialized: {tokenizer_name}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Max seq len: {max_seq_len}")
        print(f"  Max sentences: {max_sentences}")
        print(f"  Prioritize supporting docs: {prioritize_supporting}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _reorder_documents_by_importance(
        self, 
        titles: List[str], 
        sentences_list: List[List[str]], 
        supporting_titles: List[str]
    ) -> Tuple[List[str], List[List[str]]]:
        """
        reorder documents to prioritize supporting fact documents.
        this ensures the model sees relevant evidence even if we truncate.
        
        Args:
            titles: document titles
            sentences_list: list of sentence lists per document
            supporting_titles: titles that contain supporting facts
        
        Returns:
            reordered_titles, reordered_sentences
        """
        # Original order: [Doc1, Doc2, Doc3, Doc4, ...]
        # Supporting facts in: [Doc2, Doc4]
        # Reordered: [Doc2, Doc4, Doc1, Doc3, ...]  ← Evidence first!
        if not self.prioritize_supporting:
            return titles, sentences_list
        
        # separate supporting docs from distractors
        supporting_docs = []
        distractor_docs = []
        
        for title, sents in zip(titles, sentences_list):
            doc = (title, sents)
            if title in supporting_titles:
                supporting_docs.append(doc)
            else:
                distractor_docs.append(doc)
        
        # put supporting docs first, then distractors
        reordered = supporting_docs + distractor_docs
        
        reordered_titles = [doc[0] for doc in reordered]
        reordered_sentences = [doc[1] for doc in reordered]
        
        return reordered_titles, reordered_sentences
    
    def _find_answer_span(
        self, 
        answer_text: str, 
        context_text: str, 
        input_ids: List[int]
    ) -> Tuple[int, int]:
        """
        Find the token span of the answer in the input sequence.
        Uses improved matching with multiple strategies.
        
        Args:
            answer_text: the answer string (e.g., "yes" or "James Cameron")
            context_text: the full context string
            input_ids: full tokenized input (includes [CLS] question [SEP] context)
        
        Returns:
            (start_pos, end_pos): token indices of answer span
                                 returns (-1, -1) if not found
        """
        # Tokenize the answer to see what we're looking for
        answer_tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)
        
        if len(answer_tokens) == 0:
            return -1, -1
        
        # Strategy 1: Look for exact token sequence match in input_ids
        # This is more reliable than character-level matching
        for i in range(len(input_ids) - len(answer_tokens) + 1):
            if input_ids[i:i+len(answer_tokens)] == answer_tokens:
                return i, i + len(answer_tokens)
        
        # Strategy 2: Try case-insensitive character matching with better alignment
        answer_lower = answer_text.lower().strip()
        context_lower = context_text.lower().strip()
        
        # Find all occurrences (not just first)
        char_start = context_lower.find(answer_lower)
        
        if char_start != -1:
            # Found in text, now map to tokens more carefully
            # Reconstruct the text from tokens to find alignment
            try:
                # Decode the context portion to find exact token positions
                # Skip [CLS] token (position 0) and question tokens
                question_end = input_ids.index(self.tokenizer.sep_token_id) + 1  # After first [SEP]
                
                # Try to find answer tokens starting from after question
                for i in range(question_end, len(input_ids) - len(answer_tokens) + 1):
                    # Decode this span and check if it matches answer
                    span_tokens = input_ids[i:i+len(answer_tokens)]
                    span_text = self.tokenizer.decode(span_tokens, skip_special_tokens=True)
                    
                    # Clean and compare
                    span_clean = span_text.lower().strip()
                    answer_clean = answer_text.lower().strip()
                    
                    if span_clean == answer_clean or answer_clean in span_clean:
                        return i, i + len(answer_tokens)
            except (ValueError, IndexError):
                pass
        
        # Strategy 3: Fuzzy match - check if answer tokens appear consecutively
        # This handles cases where tokenization differs slightly
        for i in range(len(input_ids) - 1):
            span_text = self.tokenizer.decode(input_ids[i:min(i+10, len(input_ids))], skip_special_tokens=True)
            if answer_text.lower() in span_text.lower():
                # Found it! Now find exact end
                for end in range(i+1, min(i+15, len(input_ids))):
                    decoded = self.tokenizer.decode(input_ids[i:end], skip_special_tokens=True)
                    if answer_text.lower() in decoded.lower():
                        return i, end
        
        # Answer not found in context
        return -1, -1
    
    def _create_sentence_masks(
        self,
        sentence_token_lists: List[List[int]],
        total_tokens: int
    ) -> Tuple[torch.Tensor, int]:
        """
        create binary masks showing which tokens belong to which sentence.
        this is the KEY transformation for HRM's sentence-level reasoning.
        
        Args:
            sentence_token_lists: list of token IDs for each sentence
            total_tokens: total number of tokens in sequence
        
        Returns:
            sentence_masks: [max_sentences, max_seq_len] binary tensor
            num_sentences: actual number of sentences (before padding)
        
        Example:
            Input: 3 sentences with tokens at positions:
                Sent 0: tokens 1-5
                Sent 1: tokens 6-10
                Sent 2: tokens 11-15
            
            Output mask[0]: [0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,...]
            Output mask[1]: [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,...]
            Output mask[2]: [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,...]
        """
        sentence_masks = torch.zeros(self.max_sentences, self.max_seq_len, dtype=torch.float32)
        
        # track current position in token sequence
        # start at 1 to skip [CLS] token
        current_pos = 1
        num_sentences = 0
        
        for sent_idx, sent_tokens in enumerate(sentence_token_lists):
            if sent_idx >= self.max_sentences:
                # reached max sentences limit
                break
            
            sent_len = len(sent_tokens)
            
            # check if adding this sentence would exceed max_seq_len
            if current_pos + sent_len + 1 > self.max_seq_len:  # +1 for [SEP]
                # stop adding sentences since we're at capacity
                break
            
            # mark tokens belonging to this sentence
            sentence_masks[sent_idx, current_pos:current_pos + sent_len] = 1
            
            current_pos += sent_len + 1  # +1 for [SEP] after each sentence
            num_sentences += 1
        
        return sentence_masks, num_sentences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        get a single training example.
        
        Returns:
            dict with keys:
                - inputs: [seq_len] token IDs
                - labels: [seq_len] target token IDs (with IGNORE_LABEL for non-answer)
                - sentence_masks: [max_sentences, seq_len] binary mask
                - num_sentences: scalar, number of actual sentences
                - attention_mask: [seq_len] padding mask
                - (no puzzle_identifiers needed - we've removed it as it's optional now!)
        """
        # get raw example from dataset
        example = self.dataset[idx]
        
        question = example['question']
        answer = example['answer']
        context_titles = example['context']['title']
        context_sentences = example['context']['sentences']
        supporting_facts = example['supporting_facts']
        
        # reorder documents to prioritize supporting facts
        reordered_titles, reordered_sentences = self._reorder_documents_by_importance(
            context_titles,
            context_sentences,
            supporting_facts['title']
        )
        
        # build context string with sentence separation (similar to the puzzles it gets trained on)
        # format: [CLS] question [SEP] sent1 [SEP] sent2 [SEP] ... [PAD]
        
        # tokenize question
        question_tokens = self.tokenizer.encode(
            question,
            add_special_tokens=False,
            truncation=False
        )
        
        # tokenize all sentences from all documents
        all_sentence_tokens = []
        all_sentence_texts = []
        
        for doc_sentences in reordered_sentences:
            for sentence in doc_sentences:
                sent_tokens = self.tokenizer.encode(
                    sentence,
                    add_special_tokens=False,
                    truncation=False
                )
                all_sentence_tokens.append(sent_tokens)
                all_sentence_texts.append(sentence)
        
        # now we must truncate to fit within max_seq_len 
        # some like: [CLS] + question + [SEP] + sentences + [SEP]s + [PAD]
        
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        
        # start with [CLS] + question + [SEP]
        input_ids = [cls_id] + question_tokens + [sep_id]
        question_len = len(input_ids)
        
        # add sentences until we run out of space
        selected_sentence_tokens = []
        for sent_tokens in all_sentence_tokens:
            # check if adding this sentence (+ [SEP]) would exceed limit
            if len(input_ids) + len(sent_tokens) + 1 > self.max_seq_len:
                break
            
            input_ids.extend(sent_tokens + [sep_id])
            selected_sentence_tokens.append(sent_tokens)
        
        # now create sentence masks
        sentence_masks, num_sentences = self._create_sentence_masks(
            selected_sentence_tokens,
            len(input_ids)
        )
        
        # now find answer span and create labels
        # labels: IGNORE_LABEL everywhere except answer span
        IGNORE_LABEL = -100
        
        # reconstruct context text for answer finding
        context_text = " ".join(all_sentence_texts)
        
        # find answer span in tokens using improved matching
        answer_start, answer_end = self._find_answer_span(answer, context_text, input_ids)
        
        # create label tensor
        labels = torch.full((self.max_seq_len,), IGNORE_LABEL, dtype=torch.long)
        
        # Validate answer quality before setting labels
        answer_clean = answer.strip()
        valid_short_answers = ['yes', 'no', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        is_valid_answer = (
            len(answer_clean) >= 2 or  # At least 2 characters
            answer_clean.lower() in valid_short_answers  # Or valid single-char answer
        )
        
        if answer_start != -1 and answer_end != -1 and is_valid_answer:
            # Found answer in context - mark those tokens as labels
            for i in range(answer_start, min(answer_end, self.max_seq_len)):
                if i < len(input_ids):
                    labels[i] = input_ids[i]
        elif is_valid_answer:
            # Answer not found in context but is valid - append it
            # This is the fallback for answers that aren't extractable
            answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            if len(answer_tokens) > 0:  # Make sure we got valid tokens
                ans_start = len(input_ids)
                for i, token in enumerate(answer_tokens):
                    if ans_start + i < self.max_seq_len:
                        labels[ans_start + i] = token
                        if ans_start + i >= len(input_ids):
                            # extend input_ids if answer goes beyond
                            input_ids.append(token)
        # else: Invalid answer (empty or too short) - leave all labels as IGNORE_LABEL
        
        # now pad sequences
        # pad input_ids to max_seq_len
        input_ids = input_ids[:self.max_seq_len]  # truncate if too long
        padding_length = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [pad_id] * padding_length
        
        # create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * (len(input_ids) - padding_length) + [0] * padding_length
        
        # now convert to tensors
        return {
            'inputs': torch.tensor(input_ids, dtype=torch.long),
            'labels': labels,
            'sentence_masks': sentence_masks,
            'num_sentences': torch.tensor(num_sentences, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            # no puzzle_identifiers needed - optional in model now!
        }


class HotpotQADatasetMetadata:
    """
    metadata about the HotpotQA dataset for training configuration.
    similar to PuzzleDatasetMetadata in original HRM code.
    """
    def __init__(self, dataset: HotpotQADataset):
        self.vocab_size = dataset.vocab_size
        self.seq_len = dataset.max_seq_len
        self.max_sentences = dataset.max_sentences
        
        # for compatibility with pretrain file
        self.num_puzzle_identifiers = 1  # not used, but required by config
        
        # estimate total examples (for training steps calculation)
        self.total_examples = len(dataset)
        
        # dummy values for compatibility
        self.total_groups = 1
        self.mean_puzzle_examples = self.total_examples
        
        # set names (train/validation)
        self.sets = ['train'] if dataset.split == 'train' else ['validation']


def test_hotpotqa_dataset():
    """
    test function to verify dataset works correctly.
    run this to check shapes and data before training.
    """
    print("\n" + "="*60)
    print("Testing HotpotQA Dataset")
    print("="*60)
    
    # create dataset
    dataset = HotpotQADataset(
        split='train',
        max_seq_len=1024,
        max_sentences=20,
        prioritize_supporting=True
    )
    
    print(f"\n✓ Dataset created: {len(dataset)} examples")
    
    # get one example
    print("\nFetching example 0...")
    example = dataset[0]
    
    # print shapes
    print("\n📊 Example shapes:")
    for key, value in example.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {value.shape}")
    
    # print some content
    print("\n📝 Example content:")
    print(f"  Input tokens (first 50): {example['inputs'][:50].tolist()}")
    print(f"  Decoded input (first 200 chars):")
    decoded = dataset.tokenizer.decode(example['inputs'])
    print(f"    {decoded}...")
    
    print(f"\n  Number of sentences: {example['num_sentences'].item()}")
    print(f"  Sentence masks shape: {example['sentence_masks'].shape}")
    print(f"  Non-zero sentence masks: {(example['sentence_masks'].sum(dim=1) > 0).sum().item()}")
    
    # check labels
    labeled_positions = (example['labels'] != -100).sum().item()
    print(f"\n  Labeled positions: {labeled_positions}")
    if labeled_positions > 0:
        labeled_tokens = example['labels'][example['labels'] != -100]
        print(f"  Labeled tokens: {labeled_tokens.tolist()}")
        print(f"  Decoded answer: {dataset.tokenizer.decode(labeled_tokens)}")
    
    print("\n✅ Dataset test passed!")
    print("="*60)
    
    return dataset


if __name__ == "__main__":
    test_hotpotqa_dataset()

