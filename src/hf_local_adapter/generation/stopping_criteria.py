"""
Custom stopping criteria for text generation.
"""

from typing import List, Any

import torch
from transformers import StoppingCriteria


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for specific token IDs."""

    def __init__(self, stop_ids: List[int]):
        self.stop_ids = stop_ids

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs: Any,
    ) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
