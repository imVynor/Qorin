export type ModelStatus =
  | 'archived'
  | 'active_baseline'
  | 'in_progress'
  | 'frozen_archive'
  | 'active_pretraining'
  | 'production_sft';

export interface ModelConfigRecord {
  model_id: string;
  name: string;
  historical_name: string;
  architecture_family: string;
  total_parameters: number;
  historical_metadata_parameters?: number;
  parameter_reconciliation_note?: string;
  non_embedding_parameters: number;
  layers: number;
  hidden_dim: number;
  query_heads: number;
  kv_heads: number;
  head_dim: number;
  gqa_ratio: number;
  ffn_hidden_dim: number;
  ffn_type: string;
  ffn_expansion_ratio: number;
  normalization: string;
  norm_eps: number;
  positional_encoding: string;
  rope_theta: number;
  rope_scale: number;
  attention_backend: string;
  context_length: number;
  vocab_size: number;
  tied_embeddings: boolean;
  linear_bias: boolean;
  out_proj_bias?: boolean;
  qkv_bias?: boolean;
  mlp_bias?: boolean;
  tokenizer_id: string;
  tokenizer_fingerprint: string;
  status: ModelStatus;
  checkpoints: Record<string, string>;
}
