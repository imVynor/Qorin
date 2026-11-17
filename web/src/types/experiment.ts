export type ExperimentCategory =
  | 'pretraining'
  | 'architecture_ablation'
  | 'context_extension'
  | 'supervised_fine_tuning'
  | 'diagnostic_audit'
  | 'algorithm_verification'
  | 'production_preparation'
  | 'evaluation'
  | 'harness_engineering';

export type ExperimentStatus =
  | 'completed'
  | 'in_progress'
  | 'planned'
  | 'archived';

export interface ExperimentRecord {
  experiment_id: string;
  name: string;
  codename: string;
  category: ExperimentCategory;
  date: string;
  objective: string;
  model_id: string;
  parameter_count: number | null;
  parameter_count_display?: string;
  historical_metadata_parameters?: number | null;
  parameter_reconciliation_note?: string;
  context_length: number | null;
  tokenizer: string;
  tokenizer_fingerprint: string;
  dataset: string | null;
  dataset_label?: string;
  training_tokens: number | null;
  steps: number | null;
  metrics: Record<string, number | string | null>;
  outcome: string;
  failed_or_succeeded?: string;
  consequent_decision?: string;
  status: ExperimentStatus;
  checkpoint: string | null;
  evidence_file?: string;
  evidence_line?: string;
}
