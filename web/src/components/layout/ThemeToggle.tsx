import React from 'react';
import { Sun, Moon, Laptop } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import styles from './ThemeToggle.module.css';

export const ThemeToggle: React.FC = () => {
  const { theme, resolvedTheme, toggleTheme } = useTheme();

  const getIcon = () => {
    if (theme === 'system') {
      return <Laptop size={13} aria-hidden="true" />;
    }
    return resolvedTheme === 'dark' ? (
      <Moon size={13} aria-hidden="true" />
    ) : (
      <Sun size={13} aria-hidden="true" />
    );
  };

  const getLabel = () => {
    if (theme === 'system') return 'Auto';
    return theme === 'dark' ? 'Dark' : 'Light';
  };

  return (
    <button
      type="button"
      className={styles.toggleBtn}
      onClick={toggleTheme}
      title={`Current theme: ${theme} (resolved to ${resolvedTheme}). Click to cycle themes.`}
      aria-label={`Current theme: ${theme}. Click to switch theme.`}
    >
      {getIcon()}
      <span className={styles.toggleLabel}>{getLabel()}</span>
    </button>
  );
};

