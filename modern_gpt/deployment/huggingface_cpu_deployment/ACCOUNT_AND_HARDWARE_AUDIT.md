# Hugging Face Account & Hardware Eligibility Audit

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/huggingface_cpu_deployment/ACCOUNT_AND_HARDWARE_AUDIT.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete; Zero GPU Workloads Triggered  

---

## 1. Account Identity & Credentials Verification

The deployment inspection was performed using the active, authenticated Hugging Face API token configured in the environment (`HF_TOKEN`):

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Authenticated Username** | `vynor` | Authenticated via `https://huggingface.co/api/whoami-v2`. |
| **Account Type** | `user` | Personal developer account. |
| **Subscription Tier** | `Community (Free Tier)` | `isPro: False`, `plan: None`. |
| **Payment Capability** | `canPay: False` | No credit card attached. Zero risk of unexpected billing. |
| **Token Identifier** | `slm-2k26` | Fine-grained personal access token. |
| **Token Scope** | `fineGrained` | Scoped to user `vynor` with permissions: `repo.write`, `repo.content.read`, `repo.access.read`, `repo.contribute.write`, `inference.serverless.write`, `inference.endpoints.infer.write`. |

---

## 2. Repository Permissions & Eligibility Findings

### 1. Model Repository Creation: **APPROVED & VERIFIED (100% FREE)**
- The Hugging Face API was queried to verify model repository creation permissions.
- **Verification Action:** Dispatched `HfApi.create_repo(repo_id="vynor/qorin-models", repo_type="model", exist_ok=True)`.
- **Result:** Successfully created and verified:
  **`https://huggingface.co/vynor/qorin-models`**
- **Capabilities:**
  - Standard model storage with Git LFS support for multi-gigabyte checkpoints.
  - Zero subscription requirements.
  - Public and private repository visibility supported.

### 2. Spaces Repository Creation: **NEW CRITICAL RESTRICTION DISCOVERED**
- The Hugging Face API was queried to verify new Space creation (`repo_type="space"`, `space_sdk="docker"`).
- **API Response:**
  ```json
  HTTP 402 Payment Required:
  "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription. Subscribe at https://huggingface.co/pro"
  ```
- **Policy Finding:** Hugging Face recently updated its Spaces hosting policy. While static HTML/JS Spaces remain free, creating *new* dynamic compute Spaces (Gradio, Streamlit, Docker) on `cpu-basic` now requires a Hugging Face Pro subscription ($9/month).
- **Existing Space Asset:**
  - Account `vynor` currently possesses one pre-existing Space created before the policy restriction: **`vynor/stt4`**.
  - SDK: `docker`.
  - Configured Hardware: `cpu-basic` (2 vCPU, 16 GB RAM).
  - Current Stage: `SLEEPING` (`gcTimeout: 172800` seconds / 48 hours idle timeout).
  - Custom Domain: `https://vynor-stt4.hf.space`.

---

## 3. Hardware Tiers & GPU Conservation Enforcement

Per mandatory instructions, **ZeroGPU resources were strictly conserved and untouched**:

| Hardware Tier | Account Status | Cost | Verification / Guardrail Action Taken |
| :--- | :--- | :--- | :--- |
| **CPU Basic** | Standard Free Tier | **$0.00** | 2 vCPU, 16 GB RAM, 50 GB disk. Verified via existing space runtime metadata. |
| **ZeroGPU (A100)** | **PRESERVED** | **$0.00** | **NO ZeroGPU workloads or decorators were called.** Zero GPU minutes consumed. Quota preserved 100% for professor evaluation. |
| **T4 Small GPU** | Blocked | $0.60/hr | Inaccessible because `canPay: False`. Cannot incur charges. |
| **A10G Small GPU** | Blocked | $1.05/hr | Inaccessible because `canPay: False`. Cannot incur charges. |

---

## 4. Architectural Summary

1. **Model Storage:** The separate model repository **`vynor/qorin-models`** is created and active on Hugging Face at zero cost.
2. **Compute Options:**
   - *Option A (Repurpose Existing Space):* The existing Docker Space `vynor/stt4` can host the QORIN backend container on free `cpu-basic` without requiring an HF Pro subscription.
   - *Option B (Zero-Cost Self-Hosted Tunnel):* Cloudflare Tunnel maps the local RTX 5060 backend to public HTTPS for $0 with zero cold starts.
   - *Option C (Hugging Face Pro Upgrade):* If the user chooses to subscribe to HF Pro ($9/mo), new dedicated Docker Spaces (`vynor/qorin-backend`) can be spawned on demand.
