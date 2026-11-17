import React, { useState } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';
import styles from './Navbar.module.css';

export const Navbar: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);

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

      {mobileOpen && (
        <div className={styles.mobileDrawer}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-2)', paddingBottom: 'var(--space-2)', borderBottom: '1px solid var(--color-rule-subtle)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-ink-muted)' }}>Appearance</span>
            <ThemeToggle />
          </div>
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
              {item.label}
            </NavLink>
          ))}
        </div>
      )}
    </header>
  );
};
