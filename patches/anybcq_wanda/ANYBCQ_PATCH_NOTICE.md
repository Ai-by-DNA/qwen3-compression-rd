# AnyBCQ + Wanda Patch — Third-Party Attribution

This notice applies to the `anybcq_wanda.patch` file distributed in the
`feature/anybcq-wanda` branch of the Ai by DNA Qwen3 Compression R&D repository.

## Upstream project

AnyBCQ
Copyright (c) 2025-present NAVER Cloud Corp.
Licensed under the Apache License, Version 2.0.

Official source: https://github.com/naver-aics/anybcq
Upstream revision: `c316a4d06fc7714aa6b500b4d607a08ec6fe0b24`
Upstream license: https://github.com/naver-aics/anybcq/blob/c316a4d06fc7714aa6b500b4d607a08ec6fe0b24/LICENSE
Upstream NOTICE: https://github.com/naver-aics/anybcq/blob/c316a4d06fc7714aa6b500b4d607a08ec6fe0b24/NOTICE

## Modifications

The Ai by DNA research project distributes a compatibility patch for the
combined Wanda 50% pruning and AnyBCQ 4-bit quantization experiment.
The patch changes these upstream files:

- `run_clm.py` — makes token-embedding resizing conditional.
- `data_utils.py` — loads previously cached C4 Arrow data for the experiment.
- `anybcq/evaluate/eval.py` — preserves the model checkpoint vocabulary size.
- `anybcq/inference/AnyBCQForCausalLM.py` — avoids overwriting the vocabulary size when not requested.

The patch records changes relative to the upstream revision above; it does
not redistribute the complete AnyBCQ source repository. Original copyright,
license, and third-party notices remain applicable. In particular, the
upstream `data_utils.py` retains copyright statements for NAVER Cloud Corp.
and the Hugging Face team.

## Repository licensing documentation

The original code developed by Ai by DNA is licensed under Apache 2.0.
See the central licensing documentation in the `main` branch:

- https://github.com/Ai-by-DNA/qwen3-compression-rd/blob/main/LICENSE
- https://github.com/Ai-by-DNA/qwen3-compression-rd/blob/main/NOTICE
- https://github.com/Ai-by-DNA/qwen3-compression-rd/blob/main/THIRD_PARTY_LICENSES.md

This supplemental attribution notice does not replace the original upstream
NOTICE, license texts, or other notices required when redistributing upstream
source code or its substantial portions.
