import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
from backend.models.base import GenerationConfig, ModelMetadata, ModelProvider, StreamChunk
from backend.models.ollama_provider import OllamaModelProvider
from backend.models.qorin_provider import QorinModelProvider


class ModelManager:
    def __init__(self):
        self.qorin_provider = QorinModelProvider()
        self.ollama_provider = OllamaModelProvider()
        self.active_provider_name: Optional[str] = None
        self.active_model_id: Optional[str] = None

    def list_models(self) -> List[ModelMetadata]:
        models: List[ModelMetadata] = []
        # Native QORIN models always listed first
        models.extend(self.qorin_provider.list_models())
        # External Ollama test models listed second
        models.extend(self.ollama_provider.list_models())
        return models

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        for m in self.list_models():
            if m.id == model_id:
                return m
        return None

    def get_provider_for_model(self, model_id: str) -> ModelProvider:
        if model_id.startswith("ollama-"):
            return self.ollama_provider
        return self.qorin_provider

    def is_loaded(self, model_id: str) -> bool:
        provider = self.get_provider_for_model(model_id)
        return provider.is_loaded(model_id)

    def select_model(self, model_id: str, device: Optional[str] = None) -> bool:
        provider = self.get_provider_for_model(model_id)

        # If switching from QORIN to Ollama, unload QORIN GPU memory
        if provider == self.ollama_provider and self.qorin_provider.active_model is not None:
            self.qorin_provider.unload_model()

        success = provider.load_model(model_id, device=device)
        if success:
            self.active_model_id = model_id
            self.active_provider_name = "ollama" if provider == self.ollama_provider else "qorin"
        return success

    def unload_active(self) -> bool:
        q_unloaded = self.qorin_provider.unload_model()
        o_unloaded = self.ollama_provider.unload_model()
        self.active_model_id = None
        self.active_provider_name = None
        return q_unloaded or o_unloaded

    async def generate_stream(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamChunk]:
        provider = self.get_provider_for_model(model_id)
        if not provider.is_loaded(model_id):
            self.select_model(model_id)

        async for chunk in provider.generate_stream(model_id, messages, config):
            yield chunk

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "active_model_id": self.active_model_id,
            "active_provider": self.active_provider_name,
            "qorin": self.qorin_provider.get_runtime_status(),
            "ollama": self.ollama_provider.get_runtime_status(),
        }

