"""
Message formatter class for handling various chat templates.
"""

import logging
from typing import Dict, List, Callable, Optional

from transformers import PreTrainedTokenizer

from .templates import ChatTemplates

logger = logging.getLogger(__name__)


class MessageFormatter:
    """
    Handles conversion of messages to prompts with support for various chat templates.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            default_system_prompt: Optional[str] = None,
            messages_to_prompt: Optional[Callable[[List[Dict[str, str]]], str]] = None,
            completion_to_prompt: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the message formatter.
        
        Args:
            tokenizer: The tokenizer to use for chat template application
            default_system_prompt: Default system prompt if none provided
            messages_to_prompt: Custom function to convert messages to prompt
            completion_to_prompt: Custom function to convert completion to prompt
        """
        self.tokenizer = tokenizer
        self.default_system_prompt = default_system_prompt
        self._messages_to_prompt = messages_to_prompt
        self._completion_to_prompt = completion_to_prompt

        # Check if tokenizer supports chat templates
        self.supports_chat_template = hasattr(tokenizer, 'apply_chat_template')

        logger.info(
            f"MessageFormatter initialized. "
            f"Chat template support: {self.supports_chat_template}"
        )

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of messages to a prompt string.
        
        This method attempts multiple strategies:
        1. Use custom messages_to_prompt if provided
        2. Use tokenizer's apply_chat_template if available
        3. Use model-specific formatting based on known patterns
        4. Fall back to generic formatting
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        # Strategy 1: Use custom formatter if provided
        if self._messages_to_prompt:
            return self._messages_to_prompt(messages)

        # Strategy 2: Use tokenizer's chat template if available
        if self.supports_chat_template:
            try:
                # Ensure messages are in the correct format
                formatted_messages = [
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                    for msg in messages
                ]

                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(
                    f"Failed to apply chat template: {e}. "
                    f"Falling back to model-specific formatting."
                )

        # Strategy 3: Model-specific formatting
        model_name_lower = self.tokenizer.name_or_path.lower()

        # Llama-2 style formatting
        if "llama-2" in model_name_lower or "llama2" in model_name_lower:
            return ChatTemplates.llama2_formatter(messages)

        # Mistral/Mixtral style formatting
        elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
            return ChatTemplates.mistral_formatter(messages)

        # Falcon style formatting
        elif "falcon" in model_name_lower:
            return ChatTemplates.falcon_formatter(messages)

        # Alpaca style formatting
        elif "alpaca" in model_name_lower:
            return ChatTemplates.alpaca_formatter(messages)

        # ChatML style (used by many models including Qwen, Yi, etc.)
        elif any(name in model_name_lower for name in ["qwen", "yi", "deepseek"]):
            return ChatTemplates.chatml_formatter(messages)

        # Strategy 4: Generic formatting
        return ChatTemplates.generic_formatter(messages, self.default_system_prompt)

    def format_completion(self, text: str) -> str:
        """
        Format a completion prompt.
        
        Args:
            text: The completion text
            
        Returns:
            Formatted prompt string
        """
        if self._completion_to_prompt:
            return self._completion_to_prompt(text)

        # Add system prompt if available
        if self.default_system_prompt:
            return f"{self.default_system_prompt}\n\n{text}"

        return text
