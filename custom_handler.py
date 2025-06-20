import sys
import os

# Ensure the src directory is in the Python path
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from src import ModelConfig, HuggingFaceLocalAdapterV2

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