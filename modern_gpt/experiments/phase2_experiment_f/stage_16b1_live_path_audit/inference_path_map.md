# End-to-End Inference Path Map

## Overview

This document maps the complete execution path traversed by a live user request from the React frontend through the backend inference engine, policy classification, tool orchestration, grounding validation, and client streaming.

---

## 1. Sequence Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Browser
    participant FE as React Frontend (ChatArea.tsx)
    participant API as FastAPI Endpoint (/api/chat/stream)
    participant Policy as ToolPolicyClassifier
    participant Model as ModelManager (500M SFT)
    participant Registry as CanonicalToolRegistry
    participant Web as UnifiedWebResearcher
    participant Val as EvidenceGroundingValidator

    User->>FE: Types "Who is Jensen Huang?"
    FE->>API: POST /api/chat/stream {messages, conversation_id, ...}
    API->>Policy: classify_intent("Who is Jensen Huang?")
    Policy-->>API: UserIntent.FACTUAL_INQUIRY
    
    alt Model did not emit tool call or emitted errant Python
        API->>Registry: Route to canonical web_search
        Registry->>Web: search("Who is Jensen Huang?")
        Web-->>Registry: EvidenceBundle (NVIDIA, Wikipedia, Britannica)
        Registry-->>API: CanonicalToolObservation + EvidenceBundle
        API-->>FE: Yield SSE "tool_start" & "tool_result"
    end

    API->>Model: Synthesis Turn (Prompt + Strict Grounding Rules + Evidence)
    Model-->>API: Generated text response
    
    API->>Val: validate(generated_text, EvidenceBundle, query)
    alt Response is ungrounded or contaminated
        Val-->>API: is_grounded = False (missing entities / proposition mismatch)
        API->>Val: synthesize_conservative_grounded_answer(EvidenceBundle, query)
        Val-->>API: Verified standalone declarative answer with [SOURCE_X] citations
        API->>Val: Independent Fallback Check validate(fallback_answer)
        Val-->>API: is_grounded = True
    else Response is verified
        Val-->>API: is_grounded = True
    end

    API-->>FE: Yield SSE token-by-token stream
    FE-->>User: Renders verified response and source cards
```

---

## 2. Component Responsibilities

1. **Frontend (`web/src/components/harness/ChatArea.tsx`)**:
   - Manages message history with stable IDs (`msg-${Date.now()}`).
   - Manages conversation scoping via `conversationIdRef.current`.
   - Sends user prompt and prior conversation context to `/api/chat/stream`.

2. **Policy Classification (`backend/tools/policy.py`)**:
   - Classifies user message into `CODE_GENERATION`, `SIMPLE_ARITHMETIC`, `EXPLICIT_EXECUTION`, or `FACTUAL_INQUIRY`.
   - Enforces execution boundaries (prevents Python calls on general factual queries).

3. **Tool Orchestration (`backend/app.py`)**:
   - Routes `FACTUAL_INQUIRY` to `web_search` when `web_research_enabled` is active.
   - Converts errant Python calls on entity queries to `web_search`.
   - Emits structured `tool_start` and `tool_result` events.

4. **Structured Evidence Handling (`backend/tools/evidence.py`)**:
   - Serializes retrieved sources into `EvidenceBundle` and `EvidenceRecord` instances with ISO 8601 timestamps, URLs, snippets, and entity tags.

5. **Deterministic Grounding Validator (`backend/tools/grounding_validator.py`)**:
   - Sentence-level co-occurrence checking.
   - Contradiction and self-referential impossibility detection.
   - Independent verification of fallback synthesis.
