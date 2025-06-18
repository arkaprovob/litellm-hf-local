# Create configuration
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ModelConfig, HuggingFaceLocalAdapterV2

config = ModelConfig(
    model_id="microsoft/Phi-4-reasoning",
    device="cuda:0",
    load_in_4bit=True,
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
    max_new_tokens=32768,
)

# Use with LiteLLM
import litellm

litellm.custom_provider_map = [
    {"provider": "huggingface-local", "custom_handler": adapter}
]

response = litellm.completion(
    model="huggingface-local/Phi-4-reasoning",
    messages=[
        {"role": "user", "content": "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"},
        {"role": "user", "content": "Write a story on kubernetes in anime style"},
],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")