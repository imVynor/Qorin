# Phase 1B.5: Architecture Decisions & Risk Register

## Decisions
1. **Decision D-1B5-01**: Maintain honest `PARTIALLY_COMPLETE_BLOCKED_ON_BANDWIDTH_LIMITS` status.
   - *Rationale*: Never declare 1B pretraining readiness when 758.3M tokens remain to be streamed. Integrity takes precedence over superficial completion.
2. **Decision D-1B5-02**: Enforce strict isolation between training and validation.
   - *Rationale*: Validation tokens are permanently excluded from training calculations and added to the quarantine index.

## Risk Register
1. **Risk R-1B5-01 (Remote Socket Throttling)**: High-volume streaming from Hugging Face may hit rate limits or socket drops.
   - *Mitigation*: Implemented automatic retry loop (up to 5 retries with exponential backoff) and persistent JSONL checkpointing every 250 records.
2. **Risk R-1B5-02 (Domain Deficit Skew)**: Web and Edu have much higher local availability than Books, Math, and Dialogue.
   - *Mitigation*: Strictly prohibit domain borrowing or synthetic inflation. Each domain must be acquired from its authorized source repository.
