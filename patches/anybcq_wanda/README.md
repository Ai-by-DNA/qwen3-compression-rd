# AnyBCQ + Wanda compatibility patch

The AnyBCQ+Wanda experiment used the official AnyBCQ repository at:

c316a4d06fc7714aa6b500b4d607a08ec6fe0b24

The experiment required local modifications to:

- run_clm.py
- data_utils.py
- anybcq/evaluate/eval.py
- anybcq/inference/AnyBCQForCausalLM.py

These exact modifications are stored in:

anybcq_wanda.patch

To reproduce the modified AnyBCQ source:

    git clone https://github.com/naver-aics/anybcq.git
    cd anybcq
    git checkout c316a4d06fc7714aa6b500b4d607a08ec6fe0b24
    git apply /path/to/anybcq_wanda.patch

The patch was verified against the source used for the reported AnyBCQ+Wanda experiment.
