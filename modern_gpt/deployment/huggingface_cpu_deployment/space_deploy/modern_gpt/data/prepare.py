"""Data preparation module and CLI for ModernGPT-S.

Provides backward-compatible `read_split` function alongside modern
dataset and tokenizer pipeline subcommands:
- verify-sources
- prepare-corpus
- train-tokenizer
- evaluate-tokenizer
- create-shards
- compare-report
- run-pipeline
"""

import argparse
import sys
from pathlib import Path

# Ensure package root is in sys.path if invoked directly
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


# Preserve backward compatibility for existing smoke tests and training scripts
def read_split(path, train_ratio=0.9):
    text = Path(path).read_text(encoding="utf-8")
    split = int(len(text) * train_ratio)
    return text[:split], text[split:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ModernGPT-S Dataset and Tokenizer Preparation Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available pipeline subcommands")

    # verify-sources
    p_verify = subparsers.add_parser("verify-sources", help="Verify HuggingFace connectivity and dataset schema")
    p_verify.add_argument("--sample-size", type=int, default=2, help="Number of sample records per source")

    # prepare-corpus
    p_prep = subparsers.add_parser("prepare-corpus", help="Stream bounded corpus, deduplicate, and split")
    p_prep.add_argument("--output-dir", type=str, default="artifacts/corpus", help="Directory to save raw jsonl and manifest")
    p_prep.add_argument("--total-docs", type=int, default=None, help="Total number of documents to stream")
    p_prep.add_argument("--target-tokens", type=int, default=None, help="Target training token budget (e.g. 5000000) for measured token collection")
    p_prep.add_argument("--approx-tokens-per-doc", type=int, default=715, help="Estimated average tokens per document (fallback when tokenizer not available)")
    p_prep.add_argument("--tokenizer-dir", type=str, default="artifacts/tokenizer_16k_v2", help="Path to tokenizer directory for measuring streaming tokens")
    p_prep.add_argument("--fineweb-ratio", type=float, default=0.70, help="FineWeb-Edu target proportion")
    p_prep.add_argument("--cosmo-ratio", type=float, default=0.20, help="Cosmopedia v2 target proportion")
    p_prep.add_argument("--code-ratio", type=float, default=0.10, help="CodeParrot clean target proportion")
    p_prep.add_argument("--val-ratio", type=float, default=0.10, help="Validation split proportion")
    p_prep.add_argument("--seed", type=int, default=123, help="Deterministic random seed")
    p_prep.add_argument("--min-chars", type=int, default=50, help="Minimum character length filter")
    p_prep.add_argument("--overwrite", action="store_true", help="Overwrite existing corpus artifacts if present")

    # train-tokenizer
    p_train_tok = subparsers.add_parser("train-tokenizer", help="Train 16K BPE tokenizer on training split")
    p_train_tok.add_argument("--corpus-dir", type=str, default="artifacts/corpus", help="Path to corpus directory with train_documents.jsonl")
    p_train_tok.add_argument("--output-dir", type=str, default="artifacts/tokenizer_16k_v2", help="Directory to save trained tokenizer")
    p_train_tok.add_argument("--name", type=str, default="bpe_16k_v2", help="Tokenizer identifier")
    p_train_tok.add_argument("--vocab-size", type=int, default=16000, help="Target vocabulary size")

    # evaluate-tokenizer
    p_eval_tok = subparsers.add_parser("evaluate-tokenizer", help="Evaluate and compare GPT-2 vs 16K BPE")
    p_eval_tok.add_argument("--corpus-dir", type=str, default="artifacts/corpus", help="Path to corpus directory with val_documents.jsonl")
    p_eval_tok.add_argument("--tokenizer-dir", type=str, default="artifacts/tokenizer_16k_v2", help="Path to trained tokenizer directory")
    p_eval_tok.add_argument("--legacy-dir", type=str, default="artifacts/tokenizer_16k", help="Optional path to legacy tokenizer directory for 3-way comparison")
    p_eval_tok.add_argument("--output-file", type=str, default="artifacts/reports/evaluation_metrics.json", help="Path to save evaluation metrics")

    # create-shards
    p_shards = subparsers.add_parser("create-shards", help="Produce binary uint16 memmap shards for training")
    p_shards.add_argument("--corpus-dir", type=str, default="artifacts/corpus", help="Path to corpus directory")
    p_shards.add_argument("--tokenizer-dir", type=str, default="artifacts/tokenizer_16k_v2", help="Path to trained tokenizer")
    p_shards.add_argument("--output-dir", type=str, default="artifacts/shards/bpe_16k_v2", help="Directory to save binary shards")
    p_shards.add_argument("--context-length", type=int, default=512, help="Sequence context length")
    p_shards.add_argument("--seed", type=int, default=123, help="Seed")
    p_shards.add_argument("--overwrite", action="store_true", help="Overwrite existing binary shards if present")

    # compare-report
    p_report = subparsers.add_parser("compare-report", help="Generate final JSON and Markdown comparison reports")
    p_report.add_argument("--corpus-manifest", type=str, default="artifacts/corpus/corpus_manifest.json", help="Path to corpus manifest")
    p_report.add_argument("--eval-file", type=str, default="artifacts/reports/evaluation_metrics.json", help="Path to evaluation metrics")
    p_report.add_argument("--output-dir", type=str, default="artifacts/reports", help="Directory to save final comparison reports")

    # run-pipeline
    p_all = subparsers.add_parser("run-pipeline", help="Run end-to-end dataset and tokenizer preparation pipeline")
    p_all.add_argument("--output-base", type=str, default="artifacts", help="Base artifacts directory")
    p_all.add_argument("--total-docs", type=int, default=500, help="Total number of documents")
    p_all.add_argument("--fineweb-ratio", type=float, default=0.8, help="FineWeb-Edu proportion")
    p_all.add_argument("--val-ratio", type=float, default=0.1, help="Validation split proportion")
    p_all.add_argument("--vocab-size", type=int, default=16000, help="Vocabulary size for BPE")
    p_all.add_argument("--context-length", type=int, default=512, help="Context length")
    p_all.add_argument("--seed", type=int, default=123, help="Seed")

    return parser


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    from data.pipeline import (
        cmd_compare_report,
        cmd_create_shards,
        cmd_evaluate_tokenizer,
        cmd_prepare_corpus,
        cmd_train_tokenizer,
        cmd_verify_sources,
    )

    if args.command == "verify-sources":
        cmd_verify_sources(sample_size=args.sample_size)

    elif args.command == "prepare-corpus":
        cmd_prepare_corpus(
            output_dir=Path(args.output_dir),
            total_docs=args.total_docs,
            target_tokens=args.target_tokens,
            approx_tokens_per_doc=args.approx_tokens_per_doc,
            tokenizer_dir=Path(args.tokenizer_dir) if args.tokenizer_dir else None,
            fineweb_ratio=args.fineweb_ratio,
            cosmo_ratio=args.cosmo_ratio,
            code_ratio=args.code_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            min_chars=args.min_chars,
            overwrite=args.overwrite,
        )

    elif args.command == "train-tokenizer":
        cmd_train_tokenizer(
            corpus_dir=Path(args.corpus_dir),
            output_dir=Path(args.output_dir),
            vocab_size=args.vocab_size,
            name=args.name,
        )

    elif args.command == "evaluate-tokenizer":
        cmd_evaluate_tokenizer(
            corpus_dir=Path(args.corpus_dir),
            tokenizer_dir=Path(args.tokenizer_dir),
            legacy_tokenizer_dir=Path(args.legacy_dir) if args.legacy_dir else None,
            output_file=Path(args.output_file) if args.output_file else None,
        )

    elif args.command == "create-shards":
        cmd_create_shards(
            corpus_dir=Path(args.corpus_dir),
            tokenizer_dir=Path(args.tokenizer_dir),
            output_dir=Path(args.output_dir),
            context_length=args.context_length,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    elif args.command == "compare-report":
        import json
        with Path(args.eval_file).open("r", encoding="utf-8") as f:
            eval_metrics = json.load(f)
        cmd_compare_report(
            eval_metrics=eval_metrics,
            corpus_manifest_path=Path(args.corpus_manifest),
            output_dir=Path(args.output_dir),
        )

    elif args.command == "run-pipeline":
        base_dir = Path(args.output_base)
        corpus_dir = base_dir / "corpus"
        tokenizer_dir = base_dir / "tokenizer_16k"
        shards_dir = base_dir / "shards"
        reports_dir = base_dir / "reports"

        print("=== Step 1: Verify Sources ===")
        cmd_verify_sources(sample_size=1)

        print("\n=== Step 2: Prepare Corpus ===")
        manifest_file = cmd_prepare_corpus(
            output_dir=corpus_dir,
            total_docs=args.total_docs,
            fineweb_ratio=args.fineweb_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        print("\n=== Step 3: Train 16K BPE Tokenizer ===")
        cmd_train_tokenizer(
            corpus_dir=corpus_dir,
            output_dir=tokenizer_dir,
            vocab_size=args.vocab_size,
        )

        print("\n=== Step 4: Evaluate Tokenizers ===")
        eval_metrics = cmd_evaluate_tokenizer(
            corpus_dir=corpus_dir,
            tokenizer_dir=tokenizer_dir,
            output_file=reports_dir / "evaluation_metrics.json",
        )

        print("\n=== Step 5: Create Shards ===")
        cmd_create_shards(
            corpus_dir=corpus_dir,
            tokenizer_dir=tokenizer_dir,
            output_dir=shards_dir,
            context_length=args.context_length,
            seed=args.seed,
        )

        print("\n=== Step 6: Generate Comparison Report ===")
        json_rep, md_rep = cmd_compare_report(
            eval_metrics=eval_metrics,
            corpus_manifest_path=manifest_file,
            output_dir=reports_dir,
        )
        print(f"\nPipeline successfully completed! Reports generated:\n- {json_rep}\n- {md_rep}")


if __name__ == "__main__":
    main()
