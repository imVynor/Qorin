import os
import time
from datetime import datetime, timezone
import urllib.request
import urllib.error
import json
from typing import Optional

from backend.usage.base import (
    ProviderUsageReport,
    ProviderStatus,
    InferenceCreditInfo,
    ResourceCategory,
)


class HuggingFaceUsageProvider:
    """
    Extensible, secure provider for retrieving verified Hugging Face
    account status and infrastructure usage metadata via official APIs.
    """
    PROVIDER_ID = "huggingface"
    PROVIDER_NAME = "Hugging Face"
    DASHBOARD_URL = "https://huggingface.co/settings/billing"
    DOCS_URL = "https://huggingface.co/docs/hub/billing"

    CACHE_TTL_SECONDS = 60.0

    def __init__(self):
        self._cached_report: Optional[ProviderUsageReport] = None
        self._cache_timestamp: float = 0.0

    def _get_api_token(self) -> Optional[str]:
        """
        Retrieves HF token strictly from server-side environment variables.
        Tokens are NEVER returned in any public response or logged.
        """
        return (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

    async def get_usage(self, force_refresh: bool = False) -> ProviderUsageReport:
        now = time.time()
        if not force_refresh and self._cached_report is not None:
            if now - self._cache_timestamp < self.CACHE_TTL_SECONDS:
                # Return cached report with is_cached=True
                cached = self._cached_report.model_copy()
                cached.is_cached = True
                return cached

        token = self._get_api_token()
        iso_now = datetime.now(timezone.utc).isoformat()

        if not token or not token.strip():
            report = ProviderUsageReport(
                provider_id=self.PROVIDER_ID,
                provider_name=self.PROVIDER_NAME,
                status=ProviderStatus.NOT_CONNECTED,
                account_name=None,
                plan_tier="Unconfigured",
                is_live=False,
                is_cached=False,
                last_synchronized=None,
                inference_credits=InferenceCreditInfo(
                    available_via_api=False,
                    status="not_connected",
                    remaining=None,
                    used=None,
                    limitation_notice="HF_TOKEN environment variable is not configured on the backend server. Configure HF_TOKEN to synchronize account status.",
                    billing_dashboard_url=self.DASHBOARD_URL,
                ),
                other_resources=[
                    ResourceCategory(
                        category_id="hub_storage",
                        name="Hub Storage & Bandwidth",
                        status="unavailable",
                        note="Connect HF_TOKEN to inspect authorized Hub namespaces.",
                        dashboard_url="https://huggingface.co/settings/billing",
                    ),
                    ResourceCategory(
                        category_id="spaces_hardware",
                        name="Spaces Hardware Accelerators",
                        status="unavailable",
                        note="Connect HF_TOKEN to query Spaces compute configurations.",
                        dashboard_url="https://huggingface.co/settings/billing",
                    ),
                ],
                dashboard_url=self.DASHBOARD_URL,
                documentation_url=self.DOCS_URL,
                error_message="HF_TOKEN environment variable is missing on the inference server.",
            )
            self._cached_report = report
            self._cache_timestamp = now
            return report

        # Attempt querying official HF whoami endpoint
        url = "https://huggingface.co/api/whoami-v2"
        headers = {
            "Authorization": f"Bearer {token.strip()}",
            "User-Agent": "QORIN-Research-Showcase/1.0",
            "Accept": "application/json",
        }

        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                status_code = resp.getcode()
                raw_body = resp.read().decode("utf-8")
                user_info = json.loads(raw_body)

            # Successfully verified identity
            account_name = user_info.get("name") or user_info.get("id") or "Verified Account"
            is_pro = user_info.get("isPro", False)
            raw_plan = str(user_info.get("plan", "")).upper()
            if is_pro or "PRO" in raw_plan:
                plan_tier = "Hugging Face Pro"
            elif "ENTERPRISE" in raw_plan:
                plan_tier = "Hugging Face Enterprise"
            else:
                plan_tier = "Community (Free Tier)"

            can_pay = user_info.get("canPay", False)
            auth_info = user_info.get("auth", {})
            token_role = auth_info.get("accessToken", {}).get("role", "read") if isinstance(auth_info, dict) else "read"

            report = ProviderUsageReport(
                provider_id=self.PROVIDER_ID,
                provider_name=self.PROVIDER_NAME,
                status=ProviderStatus.CONNECTED,
                account_name=account_name,
                account_type=user_info.get("type", "user"),
                plan_tier=plan_tier,
                is_live=True,
                is_cached=False,
                last_synchronized=iso_now,
                inference_credits=InferenceCreditInfo(
                    available_via_api=False,
                    status="connected_via_account",
                    remaining=None,
                    used=None,
                    currency="USD",
                    billing_period="Monthly Cycle" if can_pay else "Free Tier (Rate-Limited)",
                    last_synchronized=iso_now,
                    limitation_notice="Hugging Face does not expose real-time dollar credit balances via REST API. Serverless inference compute is tracked via account monthly invoices.",
                    billing_dashboard_url=self.DASHBOARD_URL,
                ),
                other_resources=[
                    ResourceCategory(
                        category_id="inference_providers",
                        name="Serverless Inference Providers",
                        status="active" if can_pay or is_pro else "rate_limited_free_tier",
                        details=f"Token Permission: {token_role.upper()}",
                        note="Pay-as-you-go serverless routing across official HF Inference Providers.",
                        dashboard_url=self.DASHBOARD_URL,
                    ),
                    ResourceCategory(
                        category_id="hub_storage",
                        name="Model Hub Storage & LFS",
                        status="available",
                        details=f"Account: {account_name}",
                        note="Unlimited public model repositories; private repository quotas per plan tier.",
                        dashboard_url="https://huggingface.co/settings/billing",
                    ),
                    ResourceCategory(
                        category_id="spaces_hardware",
                        name="Spaces Compute & ZeroGPU",
                        status="available",
                        details="CPU Basic (Free) / GPU (Hourly Pay-as-you-go)",
                        note="Hardware grants tied to active Hugging Face Spaces instances.",
                        dashboard_url="https://huggingface.co/settings/billing",
                    ),
                ],
                dashboard_url=self.DASHBOARD_URL,
                documentation_url=self.DOCS_URL,
                error_message=None,
            )
            self._cached_report = report
            self._cache_timestamp = now
            return report

        except urllib.error.HTTPError as http_err:
            if http_err.code == 401:
                status = ProviderStatus.UNAUTHORIZED
                err_msg = "Invalid or expired Hugging Face token (HTTP 401). Verify HF_TOKEN permissions."
            elif http_err.code == 429:
                status = ProviderStatus.RATE_LIMITED
                err_msg = "Hugging Face API rate limit reached (HTTP 429). Please retry shortly."
            else:
                status = ProviderStatus.ERROR
                err_msg = f"Hugging Face API returned error HTTP {http_err.code}."

            report = ProviderUsageReport(
                provider_id=self.PROVIDER_ID,
                provider_name=self.PROVIDER_NAME,
                status=status,
                account_name=None,
                plan_tier="Authentication Failed",
                is_live=False,
                is_cached=False,
                last_synchronized=iso_now,
                inference_credits=InferenceCreditInfo(
                    available_via_api=False,
                    status="error",
                    limitation_notice=err_msg,
                    billing_dashboard_url=self.DASHBOARD_URL,
                ),
                other_resources=[],
                dashboard_url=self.DASHBOARD_URL,
                documentation_url=self.DOCS_URL,
                error_message=err_msg,
            )
            self._cached_report = report
            self._cache_timestamp = now
            return report

        except Exception as e:
            err_msg = f"Network failure connecting to Hugging Face: {type(e).__name__}"
            report = ProviderUsageReport(
                provider_id=self.PROVIDER_ID,
                provider_name=self.PROVIDER_NAME,
                status=ProviderStatus.ERROR,
                account_name=None,
                plan_tier="Network Error",
                is_live=False,
                is_cached=False,
                last_synchronized=iso_now,
                inference_credits=InferenceCreditInfo(
                    available_via_api=False,
                    status="error",
                    limitation_notice=err_msg,
                    billing_dashboard_url=self.DASHBOARD_URL,
                ),
                other_resources=[],
                dashboard_url=self.DASHBOARD_URL,
                documentation_url=self.DOCS_URL,
                error_message=err_msg,
            )
            self._cached_report = report
            self._cache_timestamp = now
            return report
