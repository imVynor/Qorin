import React, { useState, useRef, useEffect } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import {
  Menu,
  X,
  SlidersHorizontal,
  CreditCard,
  Cpu,
  Layers,
  FlaskConical,
  ExternalLink,
  Code2,
  ChevronRight,
  Lock,
  Unlock,
} from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';
import { UsageCreditsModal } from '../usage/UsageCreditsModal';
import { ComputeAuthModal } from '../auth/ComputeAuthModal';
import { isComputeAuthorized } from '../../services/api';
import styles from './Navbar.module.css';

export const Navbar: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [usageModalOpen, setUsageModalOpen] = useState(false);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [isAuthorized, setIsAuthorized] = useState(false);
  const optionsRef = useRef<HTMLDivElement>(null);
  const location = useLocation();

  useEffect(() => {
    const checkAuth = () => setIsAuthorized(isComputeAuthorized());
    checkAuth();
    window.addEventListener('qorin-compute-auth-changed', checkAuth);
    return () => window.removeEventListener('qorin-compute-auth-changed', checkAuth);
  }, []);

  // Close popups on route change
  useEffect(() => {
    setMobileOpen(false);
    setOptionsOpen(false);
  }, [location.pathname]);

  // Handle outside click for options popup
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent | TouchEvent) => {
      if (optionsRef.current && !optionsRef.current.contains(e.target as Node)) {
        setOptionsOpen(false);
      }
    };

    if (optionsOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
      document.addEventListener('touchstart', handleOutsideClick);
    }
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('touchstart', handleOutsideClick);
    };
  }, [optionsOpen]);

  // Handle Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOptionsOpen(false);
        setMobileOpen(false);
      }
    };
    if (optionsOpen || mobileOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [optionsOpen, mobileOpen]);

  const navItems = [
    { label: 'Overview', path: '/' },
    { label: 'Journey', path: '/journey' },
    { label: 'Research', path: '/research' },
    { label: 'Experiments', path: '/experiments' },
    { label: 'Architecture', path: '/model' },
    { label: 'Harness', path: '/harness' },
    { label: 'Lab', path: '/learn' },
    { label: 'About', path: '/about' },
  ];

  return (
    <header className={styles.header}>
      <div className={styles.container}>
        {/* Brand Link */}
        <Link to="/" className={styles.brandLink}>
          <img
            src={`${import.meta.env.BASE_URL}qorin.svg`}
            alt="QORIN Logo"
            className={styles.brandLogo}
            width="22"
            height="22"
          />
          <span className={styles.brandWordmark}>QORIN</span>
          <span className={styles.brandDivider}>/</span>
          <span className={styles.brandSubtitle}>Language Model Research</span>
        </Link>

        {/* Main Desktop Navigation */}
        <nav className={styles.navContainer} aria-label="Main Navigation">
          <ul className={styles.navMenu}>
            {navItems.map((item) => (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  end={item.path === '/'}
                  className={({ isActive }) =>
                    isActive
                      ? `${styles.navLink} ${styles.navLinkActive}`
                      : styles.navLink
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Actions & Options */}
        <div className={styles.actionsGroup}>
          <ThemeToggle />

          <Link
            to="/experiments"
            className={styles.statusIndicator}
            title="QORIN-514M Dual-Tool Orchestrator Active"
          >
            <span className={styles.statusDot} />
            <span>514M Active</span>
          </Link>

          {/* Options Dropdown Trigger */}
          <div className={styles.optionsWrapper} ref={optionsRef}>
            <button
              type="button"
              className={`${styles.optionsTrigger} ${optionsOpen ? styles.optionsTriggerActive : ''}`}
              onClick={() => setOptionsOpen(!optionsOpen)}
              aria-label="Platform options and provider usage"
              aria-haspopup="true"
              aria-expanded={optionsOpen}
            >
              <SlidersHorizontal size={14} />
              <span className={styles.optionsTriggerText}>Options</span>
            </button>

            {/* Redesigned Scientific Options Popup */}
            {optionsOpen && (
              <div className={styles.optionsPopup} role="menu" aria-label="Platform options">
                <div className={styles.popupHeader}>
                  <div className={styles.popupHeaderTitle}>Platform Utilities</div>
                  <div className={styles.popupHeaderBadge}>v1.0 • Showcase</div>
                </div>

                <div className={styles.popupSection}>
                  <div className={styles.popupSectionTitle}>Provider & Access</div>

                  {/* Compute Access Protection Item */}
                  <button
                    type="button"
                    className={styles.popupItemButton}
                    role="menuitem"
                    onClick={() => {
                      setOptionsOpen(false);
                      setAuthModalOpen(true);
                    }}
                  >
                    <div className={styles.popupItemIcon}>
                      {isAuthorized ? <Unlock size={15} style={{ color: '#22c55e' }} /> : <Lock size={15} />}
                    </div>
                    <div className={styles.popupItemContent}>
                      <div className={styles.popupItemLabelRow}>
                        <span className={styles.popupItemLabel}>Compute Access</span>
                        <span
                          className={styles.popupItemTag}
                          style={isAuthorized ? { background: 'rgba(34,197,94,0.15)', color: '#22c55e' } : undefined}
                        >
                          {isAuthorized ? 'Authorized' : 'Protected'}
                        </span>
                      </div>
                      <span className={styles.popupItemDesc}>
                        {isAuthorized ? 'Prof. Udit Bhatia (Unlocked)' : 'Restricted Compute (Password Protected)'}
                      </span>
                    </div>
                    <ChevronRight size={13} className={styles.popupItemArrow} />
                  </button>

                  {/* Usage & Credits Item */}
                  <button
                    type="button"
                    className={styles.popupItemButton}
                    role="menuitem"
                    onClick={() => {
                      setOptionsOpen(false);
                      setUsageModalOpen(true);
                    }}
                  >
                    <div className={styles.popupItemIcon}>
                      <CreditCard size={15} />
                    </div>
                    <div className={styles.popupItemContent}>
                      <div className={styles.popupItemLabelRow}>
                        <span className={styles.popupItemLabel}>Usage & Credits</span>
                        <span className={styles.popupItemTag}>Quota</span>
                      </div>
                      <span className={styles.popupItemDesc}>Hugging Face & Provider Accounting</span>
                    </div>
                    <ChevronRight size={13} className={styles.popupItemArrow} />
                  </button>

                  {/* Inference Harness Link */}
                  <Link
                    to="/harness"
                    className={styles.popupItemLink}
                    role="menuitem"
                    onClick={() => setOptionsOpen(false)}
                  >
                    <div className={styles.popupItemIcon}>
                      <Cpu size={15} />
                    </div>
                    <div className={styles.popupItemContent}>
                      <div className={styles.popupItemLabelRow}>
                        <span className={styles.popupItemLabel}>Inference Harness</span>
                      </div>
                      <span className={styles.popupItemDesc}>Interactive Sandbox & Dual-Tool Mode</span>
                    </div>
                    <ChevronRight size={13} className={styles.popupItemArrow} />
                  </Link>
                </div>

                <div className={styles.popupDivider} />

                <div className={styles.popupSection}>
                  <div className={styles.popupSectionTitle}>Architecture & Research</div>

                  <Link
                    to="/model"
                    className={styles.popupItemLink}
                    role="menuitem"
                    onClick={() => setOptionsOpen(false)}
                  >
                    <div className={styles.popupItemIcon}>
                      <Layers size={15} />
                    </div>
                    <div className={styles.popupItemContent}>
                      <span className={styles.popupItemLabel}>Model Lineage</span>
                      <span className={styles.popupItemDesc}>16M to 514M Parameter Reconciliations</span>
                    </div>
                    <ChevronRight size={13} className={styles.popupItemArrow} />
                  </Link>

                  <Link
                    to="/experiments"
                    className={styles.popupItemLink}
                    role="menuitem"
                    onClick={() => setOptionsOpen(false)}
                  >
                    <div className={styles.popupItemIcon}>
                      <FlaskConical size={15} />
                    </div>
                    <div className={styles.popupItemContent}>
                      <span className={styles.popupItemLabel}>Experiments Archive</span>
                      <span className={styles.popupItemDesc}>14 Formal Benchmark & Loss Records</span>
                    </div>
                    <ChevronRight size={13} className={styles.popupItemArrow} />
                  </Link>
                </div>

                <div className={styles.popupDivider} />

                <div className={styles.popupFooter}>
                  <a
                    href="https://github.com/imVynor/Qorin"
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.popupFooterLink}
                    role="menuitem"
                  >
                    <Code2 size={13} />
                    <span>GitHub: imVynor/Qorin</span>
                    <ExternalLink size={11} style={{ marginLeft: 'auto' }} />
                  </a>
                </div>
              </div>
            )}
          </div>

          {/* Mobile Menu Toggle */}
          <button
            type="button"
            className={styles.mobileToggle}
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle navigation menu"
            aria-expanded={mobileOpen}
          >
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Redesigned Mobile Drawer */}
      {mobileOpen && (
        <div className={styles.mobileDrawer} aria-label="Mobile navigation">
          <div className={styles.mobileDrawerHeader}>
            <div className={styles.mobileThemeGroup}>
              <span className={styles.mobileThemeLabel}>Appearance</span>
              <ThemeToggle />
            </div>

            <button
              type="button"
              className={styles.mobileUsageButton}
              onClick={() => {
                setMobileOpen(false);
                setUsageModalOpen(true);
              }}
            >
              <CreditCard size={14} />
              <span>Usage & Credits</span>
            </button>

            <button
              type="button"
              className={styles.mobileUsageButton}
              onClick={() => {
                setMobileOpen(false);
                setAuthModalOpen(true);
              }}
              style={{ marginTop: '8px' }}
            >
              {isAuthorized ? <Unlock size={14} style={{ color: '#22c55e' }} /> : <Lock size={14} />}
              <span>Compute: {isAuthorized ? 'Authorized' : 'Protected'}</span>
            </button>
          </div>

          <div className={styles.mobileNavList}>
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  isActive
                    ? `${styles.mobileNavLink} ${styles.mobileNavLinkActive}`
                    : styles.mobileNavLink
                }
                onClick={() => setMobileOpen(false)}
              >
                <span>{item.label}</span>
                <ChevronRight size={14} className={styles.mobileNavArrow} />
              </NavLink>
            ))}
          </div>

          <div className={styles.mobileDrawerFooter}>
            <a
              href="https://github.com/imVynor/Qorin"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.mobileGithubLink}
            >
              <Code2 size={14} />
              <span>Source: imVynor/Qorin</span>
              <ExternalLink size={12} style={{ marginLeft: 'auto' }} />
            </a>
          </div>
        </div>
      )}

      {/* Dedicated Usage & Credits Modal Panel */}
      <UsageCreditsModal
        isOpen={usageModalOpen}
        onClose={() => setUsageModalOpen(false)}
      />

      {/* Compute Access Protection Modal (Prof. Udit Bhatia) */}
      <ComputeAuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onAuthorized={() => setIsAuthorized(true)}
      />
    </header>
  );
};
