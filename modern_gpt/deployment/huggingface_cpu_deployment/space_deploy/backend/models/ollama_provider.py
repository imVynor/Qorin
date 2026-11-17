import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import urllib.error
import urllib.request
import urllib.parse

from backend.models.base import GenerationConfig, ModelMetadata, ModelProvider, StreamChunk


class OllamaModelProvider(ModelProvider):
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")
        self.active_model_id: Optional[str] = None
        self._is_server_reachable = False
        self._last_check_time = 0.0

    def _check_connectivity(self) -> bool:
        now = time.time()
        # Cache check result for 5 seconds to prevent spamming
        if now - self._last_check_time < 5.0:
            return self._is_server_reachable

        self._last_check_time = now
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                self._is_server_reachable = resp.status == 200
        except Exception:
            self._is_server_reachable = False
        return self._is_server_reachable

    def list_models(self) -> List[ModelMetadata]:
        if not self._check_connectivity():
            return []

        results: List[ModelMetadata] = []
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                for m in models:
                    name = m.get("name", "unknown")
                    mid = f"ollama-{name.replace(':', '-')}"
                    size_bytes = m.get("size", 0)
                    params_est = size_bytes // 2 if size_bytes > 0 else 0
                    results.append(
                        ModelMetadata(
                            id=mid,
                            name=f"{name} (External Test Model)",
                            provider="ollama",
                            generation="external",
                            parameters=params_est,
                            context_length=m.get("details", {}).get("context_length", 4096),
                            architecture=m.get("details", {}).get("family", "external"),
                            status="ready",
                            description=f"External temporary benchmark model served via local Ollama daemon ({name}).",
                            is_native=False,
                            supported_params=[
                                "temperature", "top_p", "top_k", "max_new_tokens", "system_prompt"
                            ],
                            vram_estimate_mb=int(size_bytes / (1024 * 1024)) if size_bytes else 0,
                        )
                    )
        except Exception:
            pass

        return results

    def is_loaded(self, model_id: str) -> bool:
        return self.active_model_id == model_id

    def load_model(self, model_id: str, device: Optional[str] = None) -> bool:
        if not self._check_connectivity():
            raise ConnectionError("Ollama daemon is not reachable at 127.0.0.1:11434.")
        self.active_model_id = model_id
        return True

    def unload_model(self, model_id: Optional[str] = None) -> bool:
        self.active_model_id = None
        return True

    async def generate_stream(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamChunk]:
        if not self._check_connectivity():
            raise ConnectionError("Ollama daemon is not reachable. Ensure Ollama service is active.")

        # Map back from ollama-name to raw ollama name
        raw_name = model_id.replace("ollama-", "").replace("-", ":")
        # Find exact name if in list
        for m in self.list_models():
            if m.id == model_id:
                raw_name = m.name.split(" ")[0]
                break

        payload = {
            "model": raw_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_new_tokens,
            },
        }

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        token_count = 0

        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        err_text = await resp.aread()
                        raise RuntimeError(f"Ollama error ({resp.status_code}): {err_text.decode('utf-8', errors='ignore')}")

                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk_data = json.loads(line)
                        msg_chunk = chunk_data.get("message", {}).get("content", "")
                        done = chunk_data.get("done", False)

                        if msg_chunk:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                            now = time.perf_counter()
                            yield StreamChunk(
                                text=msg_chunk,
                                is_finished=False,
                                token_count=token_count,
                                ttft_ms=(first_token_time - start_time) * 1000.0 if first_token_time else 0.0,
                                elapsed_ms=(now - start_time) * 1000.0,
                                tok_per_sec=token_count / (now - start_time) if (now - start_time) > 0 else 0.0,
                            )

                        if done:
                            now = time.perf_counter()
                            yield StreamChunk(
                                text="",
                                is_finished=True,
                                finish_reason="stop",
                                token_count=token_count,
                                ttft_ms=(first_token_time - start_time) * 1000.0 if first_token_time else 0.0,
                                elapsed_ms=(now - start_time) * 1000.0,
                                tok_per_sec=token_count / (now - start_time) if (now - start_time) > 0 else 0.0,
                            )
                            break
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def get_runtime_status(self) -> Dict[str, Any]:
        return {
            "is_reachable": self._check_connectivity(),
            "base_url": self.base_url,
            "active_model_id": self.active_model_id,
        }

