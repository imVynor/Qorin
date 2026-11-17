# Interleaved Domain Sampling Design

## Domain Mixture
- Quality Web (`fineweb-edu`): 40.0%
- Educational (`smollm-corpus`): 15.0%
- Source Code (`codeparrot-clean`): 15.0%
- Wikipedia (`wikipedia:en`): 7.5%
- Mathematics (`open-web-math`): 7.5%
- Books / Prose (`gutenberg`): 10.0%
- Curated Dialogue (`ultrachat_200k`): 5.0%

## Zero-Copy Memory-Mapped Implementation
- Each domain's binary shards are mapped into virtual memory using `np.memmap(offset=80, dtype=np.uint16)`.
- Eliminates RAM overhead and startup loading lag.
- Remainder carryover tokens are seamlessly concatenated across shard boundaries with zero token drops.
- Sampler state (RNG seed, per-domain token counters, shard pointers, token offsets) is saved into every checkpoint.
