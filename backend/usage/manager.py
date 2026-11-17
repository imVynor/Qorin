from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.usage.base import ProviderUsageReport, UsageSummaryResponse
from backend.usage.huggingface import HuggingFaceUsageProvider


class UsageManager:
    """
    Extensible provider registry managing multiple inference and resource providers.
    Supports Hugging Face, future OpenAI or custom providers cleanly.
    """
    def __init__(self):
        self._providers: Dict[str, Any] = {
            "huggingface": HuggingFaceUsageProvider(),
        }

    def register_provider(self, provider_id: str, provider_instance: Any):
        self._providers[provider_id] = provider_instance

    async def get_all_usage(self, force_refresh: bool = False) -> UsageSummaryResponse:
        reports: List[ProviderUsageReport] = []
        for pid, provider in self._providers.items():
            report = await provider.get_usage(force_refresh=force_refresh)
            reports.append(report)

        return UsageSummaryResponse(
            providers=reports,
            timestamp=datetime.now(timezone.utc).isoformat(),
            backend_mode="connected",
        )

    async def get_provider_usage(self, provider_id: str, force_refresh: bool = False) -> Optional[ProviderUsageReport]:
        provider = self._providers.get(provider_id)
        if not provider:
            return None
        return await provider.get_usage(force_refresh=force_refresh)


usage_manager = UsageManager()
