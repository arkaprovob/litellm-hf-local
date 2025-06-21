import sys
import os

from ..src import ModelConfig, HuggingFaceLocalAdapterV2
config = ModelConfig(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    device="cuda:0",
    load_in_4bit=False,
    trust_remote_code=False
)

# Create adapter
adapter = HuggingFaceLocalAdapterV2(
    model_config=config,
    context_window=4096,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    do_sample=True,
    max_new_tokens=512,
) 