# Phase 1B.6: Architectural Decisions & Risk Register

## Key Decisions

1. **Decision D-1B6-01 (Scaling Target to 5B)**: Formally expanded the target from the 1B intermediate validation milestone to the full 5B pretraining target.
2. **Decision D-1B6-02 (Proportional Allocation Preservation)**: Maintained exact seven-domain distribution (40% Web, 15% Edu, 15% Code, 7.5% Wiki, 7.5% Math, 10% Books, 5% Dialogue).
3. **Decision D-1B6-03 (Strict Status Integrity)**: Recorded `PARTIALLY_COMPLETE_BLOCKED_ON_BANDWIDTH_LIMITS` rather than falsely declaring completion.

## Risk Register

1. **Risk R-1B6-01 (Streaming Runtime Bottleneck)**: Acquiring ~18.5 GB of text across 7 domains requires ~8 to 10 hours of continuous streaming.
   - *Mitigation*: Hardened resumable architecture with atomic writes, sync checkpoints every 250 records, and cross-session deduplication allows unattended execution.
2. **Risk R-1B6-02 (Long-Tail Domain Scarcity)**: Books, Math, and Dialogue require dedicated high-capacity repositories.
   - *Mitigation*: Authoritative repositories (LoC-PD-Books, OpenWebMath, UltraChat) have verified multi-billion token upstream capacity.
