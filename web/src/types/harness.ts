export interface ModelInfo {
  id: string;
  name: string;
  provider: 'qorin' | 'ollama';
  generation: string;
  parameters: number;
  context_length: number;
  architecture: string;
  status: string;
  checkpoint_path?: string;
  description: string;
  is_native: boolean;
  supported_params: string[];
  vram_estimate_mb: number;
}

export interface RetrievedChunk {
  chunk_id: string;
  doc_title: string;
  doc_source: string;
  section: string;
  similarity_score: number;
  snippet: string;
  token_count: number;
}

export interface RetrievalTrace {
  triggered: boolean;
  mode: string;
  decision_reason: string;
  query: string;
  retrieved_chunks: RetrievedChunk[];
  injected_context: string;
  retrieval_latency_ms: number;
}

export interface ToolTrace {
  tool_name: string;
  requested_by: string;
  status: 'completed' | 'failed' | 'disabled' | 'timeout' | 'started';
  input_payload: Record<string, any>;
  output_payload: Record<string, any>;
  latency_ms: number;
  error?: string;
  call_id?: string;
  execution_id?: string;
  request_id?: string;
  conversation_id?: string;
  stdout?: string;
  stderr?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  retrieval_trace?: RetrievalTrace;
  tool_traces?: ToolTrace[];
  ttft_ms?: number;
  tok_per_sec?: number;
  latency_ms?: number;
  token_count?: number;
}

export interface GenerationSettings {
  temperature: number;
  top_p: number;
  top_k: number;
  max_new_tokens: number;
  repetition_penalty: number;
  system_prompt: string;
  retrieval_mode: 'off' | 'always' | 'auto';
  retrieval_top_k: number;
  retrieval_threshold: number;
  web_research_enabled: boolean;
  python_sandbox_enabled: boolean;
}

export interface TelemetryInfo {
  active_model_id: string | null;
  active_provider: string | null;
  qorin: {
    active_model_id: string | null;
    device: string;
    is_loaded: boolean;
    cuda_available: boolean;
    device_name?: string;
    vram_allocated_mb?: number;
    vram_reserved_mb?: number;
    vram_total_mb?: number;
  };
  ollama: {
    is_reachable: boolean;
    base_url: string;
    active_model_id: string | null;
  };
}

export interface TrainingRunSummary {
  id: string;
  name: string;
  generation: string;
  parameters: number;
  status: string;
  description: string;
}

export interface MetricSeriesPoint {
  step: number;
  tokens: number;
  loss: number;
  lr: number;
  val_loss?: number;
  val_perplexity?: number;
  gpu_temp_c?: number;
  gpu_power_w?: number;
  tok_per_sec?: number;
  timestamp?: string;
}

export interface TrainingRunDetail {
  id: string;
  name: string;
  generation: string;
  parameters: number;
  architecture: string;
  context_length: number;
  status: string;
  total_tokens_target: number;
  tokens_processed: number;
  current_step: number;
  target_steps: number;
  progress_pct: number;
  best_val_loss?: number;
  best_val_perplexity?: number;
  last_train_loss?: number;
  checkpoint_name: string;
  learning_rate: number;
  duration_hours: number;
  series: MetricSeriesPoint[];
  duty_cycle_telemetry?: {
    completed_cycles: number;
    active_window_min: number;
    cooldown_window_min: number;
    latest_temp_c: number;
    latest_power_w: number;
    recent_cycles: any[];
  };
}

export interface ExperimentTraceRecord {
  id: string;
  timestamp: string;
  model_id: string;
  provider: string;
  checkpoint: string;
  prompt: string;
  generation_config: Record<string, any>;
  retrieval_trace?: RetrievalTrace;
  tool_traces: ToolTrace[];
  response_text: string;
  token_count: number;
  ttft_ms: number;
  total_latency_ms: number;
  tokens_per_second: number;
}

