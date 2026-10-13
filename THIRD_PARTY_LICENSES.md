# Third-Party Licenses

This repository accompanies the study **“Evaluating Weight Quantization, Pruning, and KV-Cache Compression for Qwen3-8B on a Consumer GPU.”** The project code is distributed under the [Apache License, Version 2.0](LICENSE). Third-party components retain their own licenses and copyright notices.

## Compression implementations and model

| Component | Version used in the study | Upstream repository / model | License | Use in this repository |
| --- | --- | --- | --- | --- |
| AnyBCQ | Commit `c316a4d06fc7714aa6b500b4d607a08ec6fe0b24` | [naver-aics/anybcq](https://github.com/naver-aics/anybcq) | Apache-2.0 | Official quantization implementation; a compatibility patch is distributed for the combined experiment. |
| AWQ and Wanda via `llmcompressor` | `0.13.0` | [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) | Apache-2.0 | AWQ quantization and Wanda pruning through the library APIs. |
| TurboQuant-KV (third-party implementation) | Commit `12c38c796a7199c93d232ef197cf213809f91afe` | [hackimov/turboquant-kv](https://github.com/hackimov/turboquant-kv) | Apache-2.0 | KV-cache compression; compatibility changes are applied using a repository script. |
| Qwen3-8B | Revision `b968826d9c46dd6066d109eabc6255188de91218` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 | Base model, downloaded separately; model weights are not distributed here. |

The upstream AnyBCQ implementation has its own [NOTICE](https://github.com/naver-aics/anybcq/blob/main/NOTICE), which identifies third-party components and their licenses. That notice must continue to be respected when reusing or distributing upstream code.

## Principal Python dependencies

Versions below are taken from the committed `requirements-lock.txt`, `requirements-awq.txt`, `requirements-wanda.txt`, and method-specific setup scripts. **Base** means the common benchmarking environment; **AWQ** and **Wanda** are separate environments.

| Package | Version(s) recorded | Upstream repository | License |
| --- | --- | --- | --- |
| PyTorch (`torch`) | Base `2.8.0+cu128`; AWQ/Wanda `2.13.0` | [pytorch/pytorch](https://github.com/pytorch/pytorch) | BSD-3-Clause (core; bundled third-party components have additional licenses) |
| Transformers | Base `4.55.4`; AnyBCQ official runtime `4.51.0`; AWQ/Wanda `5.14.1`; TurboQuant setup `5.15.1` | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| Datasets | Base/Wanda `4.8.5`; AWQ/AnyBCQ `5.0.1` | [huggingface/datasets](https://github.com/huggingface/datasets) | Apache-2.0 |
| lm-evaluation-harness (`lm_eval`) | `0.4.12` (common benchmark setup; included explicitly in TurboQuant setup) | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | MIT |
| Triton | Base `3.4.0`; AWQ/Wanda `3.7.1` | [triton-lang/triton](https://github.com/triton-lang/triton) | MIT (core; check bundled components if redistributed) |
| Accelerate | `1.14.0` | [huggingface/accelerate](https://github.com/huggingface/accelerate) | Apache-2.0 |
| safetensors | `0.8.0` | [huggingface/safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 |
| tokenizers | Base `0.21.4`; AWQ/Wanda `0.22.2` | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 |
| PEFT | `0.20.0` | [huggingface/peft](https://github.com/huggingface/peft) | Apache-2.0 |
| NumPy | Base `2.5.2`; AWQ `2.3.5`; Wanda `2.4.6` | [numpy/numpy](https://github.com/numpy/numpy) | BSD-3-Clause (core; additional bundled component licenses) |
| pandas | `3.0.5` | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | BSD-3-Clause |
| SciPy | Base/Wanda `1.18.0`; AWQ `1.18.1` | [scipy/scipy](https://github.com/scipy/scipy) | BSD-3-Clause (core; additional bundled component licenses) |
| scikit-learn | `1.9.0` | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD-3-Clause |
| psutil | `7.2.2` | [giampaolo/psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause |
| vLLM | AWQ requirements `0.27.1` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 |
| Matplotlib | Base `3.10.7` (plot generation) | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) | Matplotlib License (PSF-based, BSD-compatible) |

The requirements files record the specified package versions for their respective environments; they do not necessarily lock every transitive dependency. In particular, Matplotlib dependencies may be resolved separately during installation. Package dependencies are installed separately and are **not** bundled with this source repository. Binary redistributions, including NVIDIA CUDA packages appearing in some requirements files, are subject to their respective vendor licenses and additional terms.

## Evaluation datasets

| Dataset | Dataset/configuration used | Source | Stated license |
| --- | --- | --- | --- |
| WikiText-2 | `wikitext-2-raw-v1` | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) | CC BY-SA 3.0 / GFDL (as indicated in dataset metadata) |
| C4 | English C4 calibration data | [allenai/c4](https://huggingface.co/datasets/allenai/c4) | ODC-BY, subject also to Common Crawl terms for source content |
| MMLU | Full 57-subject benchmark | [hendrycks/test](https://github.com/hendrycks/test) | MIT |
| GSM8K | Full test benchmark | [openai/grade-school-math](https://github.com/openai/grade-school-math) | MIT |

Benchmark datasets are downloaded or accessed separately. Some committed GSM8K result files include original questions and reference answers, so the source attribution and applicable license notices must be preserved. See [NOTICE](NOTICE).

## Modifications and attribution

- The `feature/anybcq-wanda` branch contains `patches/anybcq_wanda/anybcq_wanda.patch`, applied against the AnyBCQ commit identified above. The patch modifies `run_clm.py`, `data_utils.py`, `anybcq/evaluate/eval.py`, and `anybcq/inference/AnyBCQForCausalLM.py`. The patch and its accompanying README identify these modifications; upstream copyright and license notices remain applicable.
- The `feature/turboquant` branch contains `scripts/turboquant/patch_transformers5_compat.py`, which changes the separately cloned TurboQuant-KV implementation to support the tested Transformers interfaces. Upstream license terms remain applicable.
- AWQ and Wanda are used through `llmcompressor`; the original standalone research implementations are not bundled with this repository.

The repository's Apache-2.0 license does not replace the licenses of the projects, dependencies, models, or datasets listed above. For any redistribution of third-party source code or binaries, preserve the applicable upstream license texts, notices, and required modification statements.
