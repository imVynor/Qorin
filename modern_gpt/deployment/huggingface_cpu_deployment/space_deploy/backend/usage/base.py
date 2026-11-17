from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProviderStatus(str, Enum):
    CONNECTED = "connected"
    NOT_CONNECTED = "not_connected"
    ERROR = "error"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"


class ResourceCategory(BaseModel):
    category_id: str
    name: str
    status: str = Field(..., description="available, unavailable, unsupported_by_api, or active")
    details: Optional[str] = None
    metric_value: Optional[str] = None
    note: Optional[str] = None
    dashboard_url: Optional[str] = None


class InferenceCreditInfo(BaseModel):
    available_via_api: bool = False
    status: str = "unavailable"
    remaining: Optional[float] = None
    used: Optional[float] = None
    currency: Optional[str] = None
    billing_period: Optional[str] = None
    last_synchronized: Optional[str] = None
    limitation_notice: Optional[str] = None
    billing_dashboard_url: str = "https://huggingface.co/settings/billing"


class ProviderUsageReport(BaseModel):
    provider_id: str
    provider_name: str
    status: ProviderStatus
    account_name: Optional[str] = None
    account_type: Optional[str] = None  # user, org
    plan_tier: Optional[str] = None     # Free, Pro, Enterprise
    is_live: bool = False
    is_cached: bool = False
    last_synchronized: Optional[str] = None
    inference_credits: InferenceCreditInfo
    other_resources: List[ResourceCategory] = []
    dashboard_url: str
    documentation_url: str
    error_message: Optional[str] = None


class UsageSummaryResponse(BaseModel):
    providers: List[ProviderUsageReport]
    timestamp: str
    backend_mode: str = "connected"
