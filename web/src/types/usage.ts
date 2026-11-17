export type ProviderStatusType = 'connected' | 'not_connected' | 'error' | 'unauthorized' | 'rate_limited';

export interface ResourceCategory {
  category_id: string;
  name: string;
  status: string;
  details?: string | null;
  metric_value?: string | null;
  note?: string | null;
  dashboard_url?: string | null;
}

export interface InferenceCreditInfo {
  available_via_api: boolean;
  status: string;
  remaining?: number | null;
  used?: number | null;
  currency?: string | null;
  billing_period?: string | null;
  last_synchronized?: string | null;
  limitation_notice?: string | null;
  billing_dashboard_url: string;
}

export interface ProviderUsageReport {
  provider_id: string;
  provider_name: string;
  status: ProviderStatusType;
  account_name?: string | null;
  account_type?: string | null;
  plan_tier?: string | null;
  is_live: boolean;
  is_cached: boolean;
  last_synchronized?: string | null;
  inference_credits: InferenceCreditInfo;
  other_resources: ResourceCategory[];
  dashboard_url: string;
  documentation_url: string;
  error_message?: string | null;
}

export interface UsageSummaryResponse {
  providers: ProviderUsageReport[];
  timestamp: string;
  backend_mode: 'connected' | 'offline' | 'static_showcase';
}
