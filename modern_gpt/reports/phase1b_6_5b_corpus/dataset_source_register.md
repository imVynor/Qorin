# Phase 1B.6: Dataset Source Register

## Upstream Hugging Face Source Registry

| Domain | Repository ID | Config / Subset | Split | Primary Language | Permitted Use |
|---|---|---|---|---|---|
| **Quality Web** | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | `train` | English | Pretraining |
| **Educational** | `HuggingFaceTB/smollm-corpus` | `cosmopedia-v2` | `train` | English | Pretraining |
| **Source Code** | `codeparrot/codeparrot-clean-valid` | `default` | `train` | Python / Polyglot | Pretraining |
| **Wikipedia / Reference** | `wikimedia/wikipedia` | `20231101.en` | `train` | English | Pretraining |
| **Mathematics / Technical** | `open-web-math/open-web-math` | `default` | `train` | English / LaTeX | Pretraining |
| **Books / Prose** | `storytracer/LoC-PD-Books` | `default` | `train` | English | Pretraining |
| **Curated Dialogue** | `HuggingFaceH4/ultrachat_200k` | `default` | `train_sft` | English | Pretraining |

All data was streamed directly via authenticated HTTP row groups.
