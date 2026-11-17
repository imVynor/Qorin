from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass
class ModelMetadata:
    id: str
    name: str
    provider: str  # "qorin" | "ollama"
    generation: str  # "16M" | "50M" | "100M" | "500M" | "external"
    parameters: int
    context_length: int
    architecture: str
    status: str  # "ready" | "loading" | "unloaded" | "training_in_progress" | "unavailable"
    checkpoint_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    supported_params: List[str] = field(default_factory=lambda: [
        "temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty", "system_prompt"
    ])
    description: str = ""
    is_native: bool = True
    vram_estimate_mb: int = 0


@dataclass
class GenerationConfig:
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 200
    repetition_penalty: float = 1.15
    system_prompt: Optional[str] = None
    stop_sequences: List[str] = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])


@dataclass
class StreamChunk:
    text: str
    is_finished: bool = False
    finish_reason: Optional[str] = None
    token_count: int = 0
    ttft_ms: Optional[float] = None
    elapsed_ms: Optional[float] = None
    tok_per_sec: Optional[float] = None


class ModelProvider(ABC):
    @abstractmethod
    def list_models(self) -> List[ModelMetadata]:
        """List all discoverable models under this provider."""
        pass

    @abstractmethod
    def load_model(self, model_id: str, device: Optional[str] = None) -> bool:
        """Load the model into memory/VRAM."""
        pass

    @abstractmethod
    def unload_model(self, model_id: str) -> bool:
        """Unload the model and reclaim memory."""
        pass

    @abstractmethod
    def is_loaded(self, model_id: str) -> bool:
        """Check whether the model is currently active."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generation tokens."""
        pass

    @abstractmethod
    def get_runtime_status(self) -> Dict[str, Any]:
        """Return runtime health, active memory, and device stats."""
        pass

