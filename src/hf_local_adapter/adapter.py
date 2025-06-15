"""
Main HuggingFace Local Adapter class for LiteLLM integration.
"""

import asyncio
import time
from threading import Thread
from typing import Iterator, AsyncIterator, Optional, Dict, List, Callable

import torch
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse, Message, Choices, Usage
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    PreTrainedTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig
)

from .config import ModelConfig
from .formatting import MessageFormatter
from .generation import GenerationParameterManager
from .utils import TokenizationUtils, setup_logging

logger = setup_logging()


class HuggingFaceLocalAdapterV2(CustomLLM):
    """
    Advanced HuggingFace Local Adapter for LiteLLM.
    
    This adapter provides a robust interface for running HuggingFace models locally
    through LiteLLM's unified API with clean separation of concerns.
    """

    def __init__(
            self,
            model_config: Optional[ModelConfig] = None,
            model_id: Optional[str] = None,
            device: Optional[str] = None,
            cache_dir: Optional[str] = None,
            # Generation parameters
            context_window: int = 4096,
            max_new_tokens: int = 512,
            temperature: float = 1.0,
            top_p: float = 1.0,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            do_sample: bool = True,
            # Stopping criteria
            stopping_ids: Optional[List[int]] = None,
            eos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            # Custom formatters
            messages_to_prompt: Optional[Callable[[List[Dict[str, str]]], str]] = None,
            completion_to_prompt: Optional[Callable[[str], str]] = None,
            default_system_prompt: Optional[str] = None,
            # Model and tokenizer instances (optional)
            model: Optional[PreTrainedModel] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            **kwargs
    ):
        """Initialize the HuggingFace Local Adapter with clean separation of concerns."""
        super().__init__()

        # Initialize model config
        if model_config is None:
            if model_id is None:
                raise ValueError("Either model_config or model_id must be provided")

            model_config = ModelConfig(
                model_id=model_id,
                device=device or "cuda:0",
                cache_dir=cache_dir,
                **kwargs
            )

        self.model_config = model_config
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens

        # Initialize model and tokenizer
        self._initialize_model_and_tokenizer(model, tokenizer)

        # Initialize utilities
        self.tokenization_utils = TokenizationUtils(
            self.tokenizer,
            context_window
        )

        # Validate context window
        self.context_window = self.tokenization_utils.validate_context_window()

        # Initialize generation parameter manager
        default_generation_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        }

        self.generation_manager = GenerationParameterManager(
            tokenizer=self.tokenizer,
            default_params=default_generation_params,
            stopping_ids=stopping_ids,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )

        # Initialize message formatter
        self.message_formatter = MessageFormatter(
            tokenizer=self.tokenizer,
            default_system_prompt=default_system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt
        )

        logger.info(
            f"HuggingFaceLocalAdapterV2 initialized with model: {self.model_config.model_id}"
        )

    def _initialize_model_and_tokenizer(
            self,
            model: Optional[PreTrainedModel] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """Initialize the model and tokenizer with proper configuration."""

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            logger.info(f"Loading tokenizer: {self.model_config.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_id,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code,
                revision=self.model_config.revision,
                **self.model_config.tokenizer_kwargs
            )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Initialize model
        if model is not None:
            self.model = model
        else:
            logger.info(f"Loading model: {self.model_config.model_id}")

            # Prepare model kwargs
            model_kwargs = self.model_config.model_kwargs.copy()

            # Set up quantization if requested
            if self.model_config.load_in_4bit or self.model_config.load_in_8bit:
                if self.model_config.quantization_config:
                    quantization_config = BitsAndBytesConfig(
                        **self.model_config.quantization_config
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=self.model_config.load_in_4bit,
                        load_in_8bit=self.model_config.load_in_8bit,
                        bnb_4bit_compute_dtype=self.model_config.torch_dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                model_kwargs["quantization_config"] = quantization_config

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_id,
                cache_dir=self.model_config.cache_dir,
                trust_remote_code=self.model_config.trust_remote_code,
                device_map=self.model_config.device_map,
                max_memory=self.model_config.max_memory,
                offload_folder=self.model_config.offload_folder,
                revision=self.model_config.revision,
                torch_dtype=self.model_config.torch_dtype,
                **model_kwargs
            )

    def completion(self, *args, **kwargs) -> ModelResponse:
        """Generate a completion for the given prompt."""
        # Extract messages or prompt
        messages = kwargs.get("messages", [])
        prompt = kwargs.get("prompt", "")

        # Format messages into prompt
        if messages:
            prompt = self.message_formatter.format_messages(messages)
        elif not prompt:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        # Tokenize input
        inputs = self.tokenization_utils.tokenize_prompt(
            prompt,
            device=self.model.device if hasattr(self.model, 'device') else None
        )
        prompt_tokens = inputs["input_ids"].shape[1]

        # Prepare generation kwargs
        gen_kwargs = self.generation_manager.prepare_generation_kwargs(**kwargs)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        generation_time = time.time() - start_time

        # Extract generated tokens (excluding prompt)
        generated_ids = outputs[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Count tokens
        completion_tokens = len(generated_ids)
        total_tokens = prompt_tokens + completion_tokens

        # Create response
        response = ModelResponse(
            id=f"chatcmpl-{int(time.time())}",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=generated_text,
                        role="assistant"
                    )
                )
            ],
            created=int(time.time()),
            model=self.model_config.model_id,
            object="chat.completion",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )

        logger.debug(
            f"Generated {completion_tokens} tokens in {generation_time:.2f}s "
            f"({completion_tokens / generation_time:.2f} tokens/s)"
        )

        return response

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        """Stream tokens as they are generated."""
        # Extract messages or prompt
        messages = kwargs.get("messages", [])
        prompt = kwargs.get("prompt", "")

        # Format messages into prompt
        if messages:
            prompt = self.message_formatter.format_messages(messages)
        elif not prompt:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        # Tokenize input
        inputs = self.tokenization_utils.tokenize_prompt(
            prompt,
            device=self.model.device if hasattr(self.model, 'device') else None
        )
        prompt_tokens = inputs["input_ids"].shape[1]

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Prepare generation kwargs
        gen_kwargs = self.generation_manager.prepare_generation_kwargs(**kwargs)
        gen_kwargs["streamer"] = streamer

        # Start generation in separate thread
        generation_thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs}
        )
        generation_thread.start()

        # Stream tokens
        generated_text = ""
        token_count = 0

        for new_text in streamer:
            generated_text += new_text
            token_count += self.tokenization_utils.count_tokens(new_text)

            chunk: GenericStreamingChunk = {
                "text": new_text,
                "is_finished": False,
                "finish_reason": None,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": token_count,
                    "total_tokens": prompt_tokens + token_count
                },
                "index": 0,
                "tool_use": None
            }
            yield chunk

        # Wait for generation to complete
        generation_thread.join()

        # Send final chunk
        final_chunk: GenericStreamingChunk = {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count
            },
            "index": 0,
            "tool_use": None
        }
        yield final_chunk

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """Async version of completion."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.completion, *args, **kwargs)

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming completion."""
        # Extract messages or prompt
        messages = kwargs.get("messages", [])
        prompt = kwargs.get("prompt", "")

        # Format messages into prompt
        if messages:
            prompt = self.message_formatter.format_messages(messages)
        elif not prompt:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        # Tokenize input
        inputs = self.tokenization_utils.tokenize_prompt(
            prompt,
            device=self.model.device if hasattr(self.model, 'device') else None
        )
        prompt_tokens = inputs["input_ids"].shape[1]

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Prepare generation kwargs
        gen_kwargs = self.generation_manager.prepare_generation_kwargs(**kwargs)
        gen_kwargs["streamer"] = streamer

        # Start generation in separate thread
        generation_thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **gen_kwargs}
        )
        generation_thread.start()

        # Stream tokens asynchronously
        generated_text = ""
        token_count = 0
        loop = asyncio.get_event_loop()

        while True:
            try:
                # Get next token with timeout
                new_text = await loop.run_in_executor(
                    None,
                    lambda: next(iter(streamer), None)
                )

                if new_text is None:
                    break

                generated_text += new_text
                token_count += self.tokenization_utils.count_tokens(new_text)

                chunk: GenericStreamingChunk = {
                    "text": new_text,
                    "is_finished": False,
                    "finish_reason": None,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": token_count,
                        "total_tokens": prompt_tokens + token_count
                    },
                    "index": 0,
                    "tool_use": None
                }
                yield chunk

            except StopIteration:
                break

        # Wait for generation to complete
        await loop.run_in_executor(None, generation_thread.join)

        # Send final chunk
        final_chunk: GenericStreamingChunk = {
            "text": "",
            "is_finished": True,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count
            },
            "index": 0,
            "tool_use": None
        }
        yield final_chunk
