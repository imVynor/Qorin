import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
  X,
  CreditCard,
  RefreshCw,
  ExternalLink,
  ShieldCheck,
  Server,
  Layers,
  KeyRound,
  Info,
} from 'lucide-react';
import type { ProviderUsageReport, UsageSummaryResponse } from '../../types/usage';
import { getProviderUsage } from '../../services/api';
import styles from './UsageCreditsModal.module.css';

interface UsageCreditsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const UsageCreditsModal: React.FC<UsageCreditsModalProps> = ({ isOpen, onClose }) => {
  const [usageData, setUsageData] = useState<UsageSummaryResponse | null>(null);
  const [selectedProviderId, setSelectedProviderId] = useState<string>('huggingface');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [showConfigHelp, setShowConfigHelp] = useState<boolean>(false);

  useEffect(() => {
    if (isOpen) {
      loadUsage(false);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const loadUsage = async (forceRefresh = false) => {
    if (forceRefresh) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }

    try {
      const data = await getProviderUsage(forceRefresh);
      setUsageData(data);
    } catch (err) {
      console.error('Failed to load usage data:', err);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  if (!isOpen) return null;

  const currentProvider: ProviderUsageReport | undefined =
    usageData?.providers.find((p) => p.provider_id === selectedProviderId) ||
    usageData?.providers[0];

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case 'connected':
        return (
          <span className={`${styles.statusPill} ${styles.statusConnected}`}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#16a34a' }} />
            Connected (Verified via API)
          </span>
        );
      case 'unauthorized':
        return (
          <span className={`${styles.statusPill} ${styles.statusError}`}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#dc2626' }} />
            Invalid Token (401)
          </span>
        );
      case 'rate_limited':
        return (
          <span className={`${styles.statusPill} ${styles.statusError}`}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#eab308' }} />
            Rate Limited (429)
          </span>
        );
      default:
        return (
          <span className={`${styles.statusPill} ${styles.statusNotConnected}`}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#94a3b8' }} />
            Not Connected
          </span>
        );
    }
  };

  return createPortal(
    <div className={styles.overlay} onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="usage-modal-title">
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.titleGroup}>
            <div className={styles.iconBadge}>
              <CreditCard size={18} />
            </div>
            <div>
              <h2 id="usage-modal-title" className={styles.title}>
                Usage & Credits Accounting
              </h2>
              <p className={styles.subtitle}>
                Provider Infrastructure & Quota Inspection
              </p>
            </div>
          </div>
          <button
            type="button"
            className={styles.closeButton}
            onClick={onClose}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className={styles.body}>
          {/* Provider Tab Bar */}
          <div className={styles.providerTabBar} role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={selectedProviderId === 'huggingface'}
              className={`${styles.providerTab} ${
                selectedProviderId === 'huggingface' ? styles.providerTabActive : ''
              }`}
              onClick={() => setSelectedProviderId('huggingface')}
            >
              <span>🤗 Hugging Face</span>
              {currentProvider?.status === 'connected' && (
                <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#16a34a' }} />
              )}
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={false}
              disabled
              className={`${styles.providerTab} ${styles.providerTabDisabled}`}
              title="OpenAI provider integration planned for future release"
            >
              <span>OpenAI (Extensible)</span>
            </button>
          </div>

          {isLoading ? (
            <div style={{ textAlign: 'center', padding: 'var(--space-8)', color: 'var(--color-ink-muted)' }}>
              <RefreshCw size={24} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
              <p style={{ marginTop: 'var(--space-2)', fontSize: '0.82rem' }}>Querying provider infrastructure status...</p>
            </div>
          ) : currentProvider ? (
            <>
              {/* Identity Card */}
              <div className={styles.identityCard}>
                <div className={styles.identityHeader}>
                  <div className={styles.providerName}>
                    <span>{currentProvider.provider_name}</span>
                  </div>
                  {getStatusBadge(currentProvider.status)}
                </div>

                <div className={styles.metaGrid}>
                  <div className={styles.metaItem}>
                    <span className={styles.metaLabel}>Account</span>
                    <span className={styles.metaValue}>
                      {currentProvider.account_name ? `@${currentProvider.account_name}` : 'Not Configured'}
                    </span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaLabel}>Plan Tier</span>
                    <span className={styles.metaValue}>
                      {currentProvider.plan_tier || 'Free / Community'}
                    </span>
                  </div>
                  <div className={styles.metaItem}>
                    <span className={styles.metaLabel}>Data Mode</span>
                    <span className={styles.metaValue}>
                      {usageData?.backend_mode === 'static_showcase'
                        ? 'Static Showcase'
                        : currentProvider.is_cached
                        ? 'Cached (60s TTL)'
                        : 'Live Synchronized'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Inference Credits Section */}
              <div className={styles.sectionBlock}>
                <div className={styles.sectionHeading}>
                  <Server size={14} /> Inference Provider Credits
                </div>

                <div className={styles.limitationNotice}>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
                    <Info size={15} style={{ flexShrink: 0, marginTop: 2, color: 'var(--color-accent-amber)' }} />
                    <div>
                      <strong>Official Accounting Notice:</strong>{' '}
                      {currentProvider.inference_credits.limitation_notice ||
                        'Hugging Face processes serverless inference usage on a monthly billing cycle. Real-time dollar credit balances are not exposed via the public REST API.'}
                    </div>
                  </div>
                </div>

                <div className={styles.metricsGrid}>
                  <div className={styles.metricCard}>
                    <span className={styles.metricTitle}>Remaining Credits</span>
                    <div className={styles.metricValueLarge}>Account Managed</div>
                    <div className={styles.metricFootnote}>
                      Tracked via Hugging Face Billing Dashboard
                    </div>
                  </div>
                  <div className={styles.metricCard}>
                    <span className={styles.metricTitle}>Billing Period</span>
                    <div className={styles.metricValueLarge}>
                      {currentProvider.inference_credits.billing_period || 'Monthly Cycle'}
                    </div>
                    <div className={styles.metricFootnote}>
                      {currentProvider.status === 'connected' ? 'Authenticated Account' : 'Requires HF_TOKEN'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Other Resources Section */}
              <div className={styles.sectionBlock}>
                <div className={styles.sectionHeading}>
                  <Layers size={14} /> Other Provider Resources
                </div>

                <div className={styles.resourceList}>
                  {currentProvider.other_resources.map((res) => (
                    <div key={res.category_id} className={styles.resourceRow}>
                      <div className={styles.resourceInfo}>
                        <span className={styles.resourceName}>{res.name}</span>
                        {res.note && <span className={styles.resourceNote}>{res.note}</span>}
                        {res.details && (
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--color-accent-indigo)' }}>
                            {res.details}
                          </span>
                        )}
                      </div>
                      <span className={styles.resourceBadge}>{res.status.toUpperCase()}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Security & Configuration Notice */}
              <div style={{ borderTop: '1px solid var(--color-rule-subtle)', paddingTop: varSpace(3) }}>
                <button
                  type="button"
                  onClick={() => setShowConfigHelp(!showConfigHelp)}
                  style={{
                    background: 'none',
                    border: 'none',
                    padding: 0,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.72rem',
                    color: 'var(--color-ink-muted)',
                  }}
                >
                  <KeyRound size={12} />
                  <span>{showConfigHelp ? 'Hide Connection Details' : 'Configure HF_TOKEN on Backend'}</span>
                </button>

                {showConfigHelp && (
                  <div
                    style={{
                      marginTop: 'var(--space-2)',
                      padding: 'var(--space-3)',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--color-paper-subtle)',
                      border: '1px solid var(--color-rule)',
                      fontSize: '0.75rem',
                      lineHeight: 1.5,
                      color: 'var(--color-ink-secondary)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                      <ShieldCheck size={14} color="#16a34a" />
                      <strong>Security Standard:</strong>
                    </div>
                    Tokens are never exposed in frontend code or client bundles. Store your token on the backend:
                    <pre
                      style={{
                        margin: '6px 0',
                        padding: '6px 8px',
                        background: 'var(--color-paper)',
                        border: '1px solid var(--color-rule-subtle)',
                        borderRadius: 'var(--radius-xs)',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.72rem',
                        overflowX: 'auto',
                      }}
                    >
                      {'$env:HF_TOKEN="hf_your_token_here"  # PowerShell\nexport HF_TOKEN="hf_your_token_here"   # Bash'}
                    </pre>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', padding: 'var(--space-6)', color: 'var(--color-ink-muted)' }}>
              No provider report available.
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className={styles.footerActions}>
          <span className={styles.syncInfo}>
            {currentProvider?.last_synchronized
              ? `Last Sync: ${new Date(currentProvider.last_synchronized).toLocaleTimeString()}`
              : 'Status: Unsynchronized'}
          </span>

          <div className={styles.actionButtonGroup}>
            <button
              type="button"
              className={styles.actionButton}
              onClick={() => loadUsage(true)}
              disabled={isRefreshing}
            >
              <RefreshCw size={12} className={isRefreshing ? 'spin' : ''} />
              <span>{isRefreshing ? 'Refreshing...' : 'Refresh Usage'}</span>
            </button>

            <a
              href={currentProvider?.dashboard_url || 'https://huggingface.co/settings/billing'}
              target="_blank"
              rel="noopener noreferrer"
              className={`${styles.actionButton} ${styles.actionButtonPrimary}`}
            >
              <span>HF Billing Dashboard</span>
              <ExternalLink size={12} />
            </a>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
};

function varSpace(num: number): string {
  return `var(--space-${num})`;
}
