"""
Chat template definitions for different model families.
"""

from typing import Dict, List


class ChatTemplates:
    """Collection of chat template formatters for different model families."""

    @staticmethod
    def llama2_formatter(messages: List[Dict[str, str]]) -> str:
        """Format messages in Llama-2 chat style."""
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        prompt = ""

        for i, message in enumerate(messages):
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                if i == 0:  # Only use first system message
                    prompt = f"{B_INST} {B_SYS}{content}{E_SYS}"
            elif role == "user":
                if i == 0 and not prompt:  # No system message
                    prompt = f"{B_INST} {content} {E_INST}"
                else:
                    prompt += f"{B_INST} {content} {E_INST}"
            elif role == "assistant":
                prompt += f" {content} "

        return prompt

    @staticmethod
    def mistral_formatter(messages: List[Dict[str, str]]) -> str:
        """Format messages in Mistral instruction style."""
        prompt = ""

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "user":
                prompt += f"[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content}</s>"
            elif role == "system":
                # Mistral doesn't have explicit system role, prepend to first user message
                if not prompt:
                    prompt = f"[INST] {content}\n\n"

        return prompt

    @staticmethod
    def falcon_formatter(messages: List[Dict[str, str]]) -> str:
        """Format messages in Falcon style."""
        prompt = ""

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += "Assistant:"
        return prompt

    @staticmethod
    def alpaca_formatter(messages: List[Dict[str, str]]) -> str:
        """Format messages in Alpaca instruction style."""
        instruction = ""
        input_text = ""

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                instruction = content
            elif role == "user":
                if instruction:
                    input_text = content
                else:
                    instruction = content

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        return prompt

    @staticmethod
    def chatml_formatter(messages: List[Dict[str, str]]) -> str:
        """Format messages in ChatML style."""
        prompt = ""

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"
        return prompt

    @staticmethod
    def generic_formatter(messages: List[Dict[str, str]], default_system_prompt: str = None) -> str:
        """Generic message formatting as fallback."""
        prompt = ""

        # Add system message if present
        system_messages = [m for m in messages if m.get("role") == "system"]
        if system_messages:
            prompt += f"System: {system_messages[0].get('content', '')}\n\n"
        elif default_system_prompt:
            prompt += f"System: {default_system_prompt}\n\n"

        # Add conversation history
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        # Add final assistant prompt
        prompt += "Assistant: "
        return prompt
