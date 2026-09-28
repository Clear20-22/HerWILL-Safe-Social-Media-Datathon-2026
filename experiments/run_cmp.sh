#!/bin/bash
# Sequentially compare encoders on fold 0.
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
unset TRANSFORMERS_CACHE
for spec in "xlm-roberta-base 2e-5 3" "csebuetnlp/banglabert 3e-5 3" "microsoft/mdeberta-v3-base 3e-5 3" "cardiffnlp/twitter-xlm-roberta-base 2e-5 3" "google/muril-base-cased 3e-5 3" "xlm-roberta-large 1e-5 3"; do
  set -- $spec
  echo "===== $1"
  .venv/bin/python -u experiments/03_transformer_cmp.py $1 $2 $3 2>&1 | grep -v -i warn
done
