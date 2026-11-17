# Phase 1B.5: Dataset Source Register

## Upstream Repository Registry

The following approved Hugging Face Hub repositories provide the authoritative corpus across all seven domains:

| Domain | Source Repository | Config / Subset | Split | Primary Language | Permitted Use |
|---|---|---|---|---|---|
| **Quality Web** | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | `train` | English | Pretraining |
| **Educational** | `HuggingFaceTB/smollm-corpus` | `cosmopedia-v2` | `train` | English | Pretraining |
| **Source Code** | `codeparrot/codeparrot-clean-valid` | `default` | `train` | Python / Multi | Pretraining |
| **Wikipedia / Reference** | `wikimedia/wikipedia` | `20231101.en` | `train` | English | Pretraining |
| **Mathematics / Technical** | `open-web-math/open-web-math` | `default` | `train` | English / LaTeX | Pretraining |
| **Books / Prose** | `storytracer/LoC-PD-Books` | `default` | `train` | English | Pretraining |
| **Curated Dialogue** | `HuggingFaceH4/ultrachat_200k` | `default` | `train_sft` | English | Pretraining |

All repositories were accessed via authenticated streaming (`streaming=True`) to avoid multi-gigabyte monolithic archive downloads and maximize network throughput.
