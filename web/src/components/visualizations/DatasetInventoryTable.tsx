import React, { useState, useMemo } from 'react';
import { Search, ChevronDown, ChevronUp, ExternalLink, ShieldCheck, FileCode } from 'lucide-react';
import { datasetsData } from '../../data/datasetsData';
import type { DatasetEntry } from '../../data/datasetsData';
import styles from './DatasetInventoryTable.module.css';

export const DatasetInventoryTable: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Dynamically compute category counts from datasetsData array
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {
      all: datasetsData.length,
      pretraining: 0,
      validation: 0,
      sft: 0,
      evaluation: 0,
      retrieval: 0,
    };

    datasetsData.forEach((d) => {
      if (d.category === 'pretraining') counts.pretraining += 1;
      else if (d.category === 'validation') counts.validation += 1;
      else if (d.category === 'sft') counts.sft += 1;
      else if (d.category === 'evaluation') counts.evaluation += 1;
      else if (d.category === 'retrieval') counts.retrieval += 1;
    });

    return counts;
  }, []);

  const categories = useMemo(() => [
    { label: `All Sources (${categoryCounts.all})`, value: 'all' },
    { label: `Pretraining (${categoryCounts.pretraining})`, value: 'pretraining' },
    { label: `Validation Corpora (${categoryCounts.validation})`, value: 'validation' },
    { label: `Supervised Tuning (${categoryCounts.sft})`, value: 'sft' },
    { label: `Evaluation Benchmarks (${categoryCounts.evaluation})`, value: 'evaluation' },
    { label: `External Retrieval (${categoryCounts.retrieval})`, value: 'retrieval' },
  ], [categoryCounts]);

  const filtered = useMemo(() => {
    return datasetsData.filter((d) => {
      const matchesCat = selectedCategory === 'all' || d.category === selectedCategory;
      const q = searchQuery.toLowerCase();
      const matchesSearch =
        !q ||
        d.name.toLowerCase().includes(q) ||
        d.source.toLowerCase().includes(q) ||
        d.description.toLowerCase().includes(q) ||
        d.license.toLowerCase().includes(q) ||
        (d.externalUrl && d.externalUrl.toLowerCase().includes(q)) ||
        (d.internalProvenance && d.internalProvenance.toLowerCase().includes(q)) ||
        d.modelUsage.some((m) => m.toLowerCase().includes(q));
      return matchesCat && matchesSearch;
    });
  }, [selectedCategory, searchQuery]);

  const toggleExpand = (id: string) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  const getLicenseBadge = (entry: DatasetEntry) => {
    let className = styles.licensePermissive;
    const label = entry.license;

    if (entry.licenseStatus === 'non_commercial') {
      className = styles.licenseNonCommercial;
    } else if (entry.licenseStatus === 'open_copyleft') {
      className = styles.licenseCopyleft;
    } else if (entry.licenseStatus === 'uncertain') {
      className = styles.licenseUncertain;
    }

    return <span className={`${styles.licenseBadge} ${className}`}>{label}</span>;
  };

  const getCategoryLabel = (entry: DatasetEntry) => {
    switch (entry.category) {
      case 'pretraining':
        return 'Pretraining';
      case 'validation':
        return 'Val Corpus';
      case 'sft':
        return 'SFT Mixture';
      case 'evaluation':
        return 'Eval Suite';
      case 'retrieval':
        return 'RAG Corpus';
      default:
        return entry.category;
    }
  };

  return (
    <div className={styles.wrap}>
      <div className={styles.controlsBar}>
        <div className={styles.searchWrap}>
          <Search size={14} className={styles.searchIcon} />
          <input
            type="text"
            placeholder="Search datasets, licenses, models, paths..."
            className={styles.searchInput}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className={styles.categoryGroup}>
          {categories.map((c) => (
            <button
              key={c.value}
              type="button"
              className={`${styles.categoryBtn} ${selectedCategory === c.value ? styles.categoryBtnActive : ''}`}
              onClick={() => setSelectedCategory(c.value)}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Dataset &amp; Provenance</th>
              <th>Category &amp; Split</th>
              <th>Volume / Size</th>
              <th>Share &amp; Denominator Context</th>
              <th>License Status</th>
              <th>Inspect</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((d) => {
              const isExpanded = expandedId === d.id;
              return (
                <React.Fragment key={d.id}>
                  <tr>
                    <td>
                      <span className={styles.datasetName}>{d.name}</span>
                      <span className={styles.datasetSource}>{d.source}</span>
                    </td>
                    <td>
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        textTransform: 'uppercase',
                        fontSize: '9px',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '1px 5px',
                        borderRadius: '2px',
                        background: d.isValidation ? 'var(--color-accent-indigo-subtle)' : 'var(--color-paper-subtle)',
                        color: d.isValidation ? 'var(--color-accent-indigo)' : 'var(--color-ink-secondary)',
                        border: '1px solid var(--color-rule-subtle)'
                      }}>
                        {d.isValidation && <ShieldCheck size={10} />}
                        {getCategoryLabel(d)}
                      </span>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>{d.size}</td>
                    <td>
                      <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ink-primary)' }}>
                        {d.contribution}
                      </strong>
                      <span className={styles.denominatorNote}>{d.denominatorContext}</span>
                    </td>
                    <td>{getLicenseBadge(d)}</td>
                    <td>
                      <button
                        type="button"
                        className={styles.detailsBtn}
                        onClick={() => toggleExpand(d.id)}
                        aria-expanded={isExpanded}
                      >
                        {isExpanded ? (
                          <>
                            Close <ChevronUp size={11} />
                          </>
                        ) : (
                          <>
                            Details <ChevronDown size={11} />
                          </>
                        )}
                      </button>
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr className={styles.drawerRow}>
                      <td colSpan={6}>
                        <div className={styles.drawerContent}>
                          <div className={styles.drawerItem}>
                            <strong>Research Purpose &amp; Motivation</strong>
                            <p>{d.purpose}</p>
                          </div>
                          <div className={styles.drawerItem}>
                            <strong>Processing &amp; Quality Filtering</strong>
                            <p>{d.processing}</p>
                          </div>
                          <div className={styles.drawerItem}>
                            <strong>Partition &amp; Model Targets</strong>
                            <p>
                              Split: <code>{d.split}</code> {d.isValidation ? '(Dedicated Ground Truth)' : '(Training Data)'} &bull; Targets:{' '}
                              <strong>{d.modelUsage.join(', ')}</strong>
                            </p>
                          </div>
                          <div className={styles.drawerItem}>
                            <strong>Source Verification &amp; Provenance</strong>
                            <p>
                              {d.externalUrl ? (
                                <a
                                  href={d.externalUrl}
                                  target="_blank"
                                  rel="noreferrer"
                                  style={{ color: 'var(--color-accent-indigo)', display: 'inline-flex', alignItems: 'center', gap: '4px', wordBreak: 'break-all' }}
                                >
                                  {d.externalUrl} <ExternalLink size={11} />
                                </a>
                              ) : d.internalProvenance ? (
                                <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--color-ink-primary)' }}>
                                  <FileCode size={12} color="var(--color-accent-sage)" />
                                  <span>Internal Repository Provenance: <code>{d.internalProvenance}</code></span>
                                </span>
                              ) : (
                                <span style={{ color: 'var(--color-ink-muted)' }}>Verified Internal Specification</span>
                              )}
                            </p>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
