import pickle
from typing import Any, Optional
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class ReturnStruct:
    """
    WARNING: only supports single batch sizes.  
    Args:
        logits: shape `(batch_size, sequence_length, vocab_size)`.  
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
        hidden_states: returned when output_hidden_states=True `(num_layers+1, sequence_length, hidden_size)`.  
        heads: returned when output_heads=True `(num_layers, num_heads, head_dim)`.  

    """
    logits: Optional[Tensor] = None
    past_key_values: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    heads: Optional[Tensor] = None

def pickle_rw(path : str, mode : str = 'r', obj : Any = None) -> Any:
    if mode not in 'rw': raise
    if mode == 'w' and obj is None: raise
    if mode == 'r' and obj is not None: raise
    with open(path, f"{mode}b") as f:
        if mode == 'r':
            return pickle.load(f)
        else:
            pickle.dump(obj, f)
