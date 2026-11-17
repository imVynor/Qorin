import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
  Lock,
  Unlock,
  AlertCircle,
  CheckCircle2,
  X,
  Loader2,
  ShieldCheck,
  ShieldAlert,
} from 'lucide-react';
import {
  isComputeAuthorized,
  verifyComputeAccess,
  clearComputeToken,
} from '../../services/api';
import styles from './ComputeAuthModal.module.css';

interface ComputeAuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAuthorized?: () => void;
}

export const ComputeAuthModal: React.FC<ComputeAuthModalProps> = ({
  isOpen,
  onClose,
  onAuthorized,
}) => {
  const [password, setPassword] = useState('');
  const [isAuthorized, setIsAuthorized] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setIsAuthorized(isComputeAuthorized());
      setPassword('');
      setErrorMessage(null);
      setSuccessMessage(null);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password.trim()) return;

    setIsVerifying(true);
    setErrorMessage(null);
    setSuccessMessage(null);

    try {
      const res = await verifyComputeAccess(password);
      setIsAuthorized(true);
      setSuccessMessage(res.message || 'Compute access authorized: Welcome, Prof. Udit Bhatia!');
      setPassword('');
      if (onAuthorized) onAuthorized();
    } catch (err: any) {
      setErrorMessage(err.message || "Invalid compute access password. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)");
    } finally {
      setIsVerifying(false);
    }
  };

  const handleRevoke = () => {
    clearComputeToken();
    setIsAuthorized(false);
    setSuccessMessage(null);
    setErrorMessage(null);
  };

  return createPortal(
    <div
      className={styles.modalOverlay}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="compute-auth-title"
    >
      <div className={styles.modalCard} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className={styles.modalHeader}>
          <div className={styles.headerTitleGroup}>
            <div className={styles.kickerRow}>
              <span className={styles.kickerBadge}>Access Control</span>
              {isAuthorized ? (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontSize: '11px',
                    fontFamily: 'var(--font-mono)',
                    color: '#22c55e',
                    fontWeight: 600,
                  }}
                >
                  <ShieldCheck size={13} /> Unlocked
                </span>
              ) : (
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontSize: '11px',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--color-ink-muted)',
                  }}
                >
                  <ShieldAlert size={13} /> Protected
                </span>
              )}
            </div>
            <h2 id="compute-auth-title" className={styles.title}>
              Personal Compute Authorization
            </h2>
            <p className={styles.subtitle}>
              Gated execution access for course evaluation &amp; research inspection.
            </p>
          </div>

          <button
            type="button"
            className={styles.closeButton}
            onClick={onClose}
            aria-label="Close dialog"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className={styles.modalBody}>
          {isAuthorized ? (
            <div className={styles.statusCardUnlocked}>
              <CheckCircle2 size={18} className={styles.statusIconUnlocked} />
              <div className={styles.statusContent}>
                <div className={styles.statusHeading}>Compute Access Authorized</div>
                <div className={styles.statusDescription}>
                  {successMessage || 'Welcome, Prof. Udit Bhatia! Personal NVIDIA RTX GPU inference, weight loading, and Python sandbox execution are unlocked.'}
                </div>
                <div style={{ marginTop: '8px' }}>
                  <button
                    type="button"
                    onClick={handleRevoke}
                    className={styles.revokeBtn}
                  >
                    <Lock size={12} /> Lock / Revoke Access
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className={styles.statusCardLocked}>
                <Lock size={18} className={styles.statusIconLocked} />
                <div className={styles.statusContent}>
                  <div className={styles.statusHeading}>Workstation GPU Gate Active</div>
                  <div className={styles.statusDescription}>
                    To prevent unauthenticated public compute abuse on the author&apos;s personal hardware, live PyTorch model inference and sandbox execution require authorization.
                  </div>
                </div>
              </div>

              {/* Password Hint Box with visual blanks */}
              <div className={styles.hintBox}>
                <div className={styles.hintHeader}>
                  <span className={styles.hintLabel}>Password Hint</span>
                  <span className={styles.hintPattern}>Pattern: XXXX XXXXXX</span>
                </div>

                <div className={styles.blanksContainer}>
                  <div className={styles.blanksWord} title="First name: 4 letters">____</div>
                  <div className={styles.blanksWord} title="Last name: 6 letters">______</div>
                </div>

                <div className={styles.hintNote}>
                  Hint: Professor&apos;s name in ALL CAPS (4 letters, space, 6 letters).
                </div>
              </div>

              {/* Input Form */}
              <form onSubmit={handleSubmit} className={styles.authForm}>
                <div className={styles.inputGroup}>
                  <label htmlFor="compute-password-input" className={styles.inputLabel}>
                    Authorization Password
                  </label>
                  <input
                    id="compute-password-input"
                    type="text"
                    className={styles.inputField}
                    placeholder="ENTER PASSWORD (ALL CAPS)..."
                    value={password}
                    onChange={(e) => setPassword(e.target.value.toUpperCase())}
                    autoComplete="off"
                    autoFocus
                  />
                </div>

                {errorMessage && (
                  <div className={styles.errorMessage} role="alert">
                    <AlertCircle size={15} style={{ flexShrink: 0 }} />
                    <span>{errorMessage}</span>
                  </div>
                )}

                <div className={styles.actionsRow}>
                  <button
                    type="button"
                    onClick={onClose}
                    className={styles.revokeBtn}
                    style={{ color: 'var(--color-ink-muted)', borderColor: 'var(--color-border)' }}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className={styles.submitBtn}
                    disabled={isVerifying || !password.trim()}
                  >
                    {isVerifying ? (
                      <>
                        <Loader2 size={13} className="spin" /> Verifying...
                      </>
                    ) : (
                      <>
                        <Unlock size={13} /> Authorize Compute
                      </>
                    )}
                  </button>
                </div>
              </form>
            </>
          )}
        </div>

        {/* Footer */}
        <div className={styles.modalFooter}>
          <span className={styles.footerNote}>
            Static showcase, model specs, and research logs remain accessible without credentials.
          </span>
          <button
            type="button"
            onClick={onClose}
            className={styles.submitBtn}
            style={{
              padding: '5px 12px',
              fontSize: '11px',
              background: 'transparent',
              color: 'var(--color-ink)',
              border: '1px solid var(--color-border)',
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
};
