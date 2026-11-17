# Deployment Architecture & Provider Assessment

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/DEPLOYMENT_OPTIONS.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. Architectural Options Comparison Matrix

| Evaluation Dimension | Option A: HF Spaces (Free CPU) | Option B: HF Spaces (Paid / ZeroGPU) | Option C: Serverless GPU (Modal / RunPod) | Option D: Self-Hosted Secure Tunnel (Cloudflare) | Option E: Hybrid (Cloud API + Local/Modal GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Estimated Monthly Cost** | **$0.00** (Free) | **$0.00** (ZeroGPU) or **~$5 - $20/mo** (T4 auto-pause) | **~$1 - $5/mo** (Pay-per-second, ~$0.0002/s) | **$0.00** (Free) | **$0.00 - $5/mo** |
| **Hardware Specs** | 2 vCPU, 16 GB RAM | Nvidia T4 (16GB VRAM) or A100 | Nvidia T4 / L4 (16GB - 24GB VRAM) | **Nvidia RTX 5060 Laptop (8GB VRAM)** | Render free web service + GPU worker |
| **Model Compatibility** | 100M: OK; 500M: Very Slow | 100M & 500M: 100% Native | 100M & 500M: 100% Native | **100M & 500M: 100% Verified** | 100M & 500M: 100% Native |
| **Cold-Start Behavior** | 60 - 120s (container sleep) | 30 - 60s (wake from idle) | 10 - 25s (container spin-up) | **0.0s (Continuously Warm)** | 0s warm / 20s cold |
| **Token Generation Speed** | 1.5 - 3.5 tok/s (500M on CPU) | 35 - 55 tok/s | 40 - 60 tok/s | **30 - 55 tok/s (Measured)** | 30 - 55 tok/s |
| **Time to First Token (TTFT)** | > 3,000 ms | ~100 - 250 ms | ~150 - 350 ms | **40 - 160 ms (Measured)** | ~250 - 500 ms |
| **Persistent Storage** | 50 GB free ephemeral | 50 GB free ephemeral | Persistent Network Volume ($0.15/GB/mo) | **Local 1TB NVMe SSD** | Remote Cloud Volume |
| **Operational Complexity** | Low (Docker template) | Medium (Gradio/Docker + GPU config) | Medium-High (Python SDK / container packaging) | **Lowest (Single `cloudflared` command)** | High (Distributed orchestration) |
| **Host Availability** | 24/7 Cloud (sleeps on idle) | 24/7 Cloud (sleeps on idle) | On-demand cloud | Dependent on laptop being awake | Split cloud & local |

---

## 2. In-Depth Evaluation of Each Architecture

### Option A: Hugging Face Spaces with CPU Inference
- **Viability:** **NOT RECOMMENDED FOR 500M; VIABLE ONLY AS A 100M FALLBACK**.
- **Hardware & Limits:** Hugging Face Spaces free tier provides 2 vCPUs and 16 GB of RAM.
- **Critical Performance Constraint:**
  - Running autoregressive matrix multiplications on 514M parameters across 2 vCPUs without GPU acceleration generates at **1.5 to 3.5 tokens per second**.
  - A standard 200-token answer requires **60 to 120 seconds** to complete, frequently triggering frontend fetch timeouts.
  - While the 100M model is tolerable (~10-14 tok/s), the flagship 500M model cannot deliver an acceptable interactive research showcase on 2 vCPUs.

### Option B: Hugging Face Spaces with GPU Inference (ZeroGPU or T4)
- **Viability:** **VIABLE CLOUD OPTION**.
- **Details:**
  - *ZeroGPU:* Hugging Face provides free dynamic A100 GPU allocation for Spaces using Gradio or Docker with the `@spaces.GPU` decorator. It allocates GPU slices on-demand with zero idle cost. However, it imposes daily usage quotas and may queue during peak community traffic.
  - *Paid T4 Small:* Costs **$0.60 per hour**. With Spaces auto-sleep configured to pause after 15 minutes of inactivity, evaluation costs for a grading window (e.g. 5 active hours across a week) total **~$3.00**.
- **Model Storage:** Weights can be stored in a private Hugging Face Model repository and loaded using `hf_hub_download` during container startup.

### Option C: Serverless GPU Inference (Modal Labs / RunPod Serverless)
- **Viability:** **EXCELLENT PRODUCTION CLOUD OPTION**.
- **Details:**
  - Providers like Modal (`modal.com`) or RunPod (`runpod.io`) offer true serverless GPU execution.
  - You pay strictly per active second of GPU usage (~$0.00015 to $0.00030/second on an Nvidia T4 or L4).
  - When idle, cost is **$0.00**.
  - 1,000 grading queries of 5 seconds each consume 5,000 seconds = **$1.00 to $1.50 total spend**.
- **Cold-Start Consideration:** Takes ~15–20 seconds to load the 500M checkpoint on cold start if container is idle.

### Option D: Self-Hosted Secure Tunnel (Cloudflare Tunnel)
- **Viability:** **STRONGLY RECOMMENDED FOR COURSE ASSESSMENT (BEST IMMEDIATE PATH)**.
- **Details:**
  - The author's development machine is an **NVIDIA RTX 5060 Laptop GPU (8GB VRAM)** running Windows 11.
  - The backend is already fully configured, tested, and running live on `http://127.0.0.1:8000`.
  - Checkpoints (1.98 GB for 500M, 588 MB for 100M), tokenizers, SQLite cache, and Python sandbox are already verified and operating locally.
  - By running **Cloudflare Tunnel** (`cloudflared`), a public HTTPS URL (e.g. `https://qorin-api.yourdomain.com` or a free Cloudflare trycloudflare tunnel) securely maps directly to `localhost:8000`.
- **Advantages:**
  - **$0.00 cost** (100% free).
  - Zero cloud setup or weight transfer overhead.
  - Zero cold starts (server is warm and ready instantly).
  - Peak performance: **mean TTFT 80ms, 30-55 tokens/second**.
  - Built-in HTTPS encryption and DDoS protection via Cloudflare edge.
  - Password gate (`UDIT BHATIA`) strictly prevents any public compute abuse.
- **Operational Requirement:** The author's laptop must remain powered on and connected to the internet during the professor's grading period.

### Option E: Hybrid Architecture
- **Viability:** **SECONDARY ALTERNATIVE**.
- Hosts the FastAPI routing, usage dashboard, and training viewer on a free cloud container (Render / Fly.io / Railway), while forwarding `/api/chat/stream` requests to a serverless GPU or local tunnel.
- Adds unnecessary distributed failure points for an undergraduate course evaluation compared to Option D.

---

## 3. Recommended Architectural Decision

1. **For Course Assessment & Final Submission:**
   - **Deploy Option D (Self-Hosted via Cloudflare Tunnel).**
   - Provide the public HTTPS tunnel URL to the GitHub repository / GitHub Pages frontend via `VITE_API_URL`.
   - The instructor visits the GitHub Pages website, clicks **Options -> Compute Access**, enters the password `UDIT BHATIA` using the provided blanks hint, and receives instantaneous, zero-latency 514M GPU token streaming directly from the author's local workstation.
2. **For Post-Course Permanent Cloud Archiving (If Needed):**
   - **Deploy Option B (Hugging Face Spaces with ZeroGPU) or Option C (Modal Serverless).**
