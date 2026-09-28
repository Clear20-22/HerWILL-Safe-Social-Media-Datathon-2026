#!/bin/bash
# Overnight chain: wait for the Qwen3-4B CV run (tmux session herwill_qwen) to finish, then run XLM-R XL CV.
# Safe to run in tmux; survives the laptop disconnecting. Every fold is cached, so re-running resumes.
cd "$(dirname "$0")/../.."
unset TRANSFORMERS_CACHE
LOG=xlmr_xl/logs/xlmr_xl_cv.txt
QLOG=qwen/logs/qwen3_4b_lora_cv.txt
FILTER='warn|newly initialized|TRAIN this model|Fetching|Loading checkpoint|use_cache'

echo "[$(date '+%F %T')] waiting for Qwen run to finish (DONE in $QLOG)" | tee -a $LOG
until grep -q '^DONE' "$QLOG" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] Qwen finished: $(grep -E 'OOF|saved' $QLOG | tr '\n' ' ')" | tee -a $LOG
sleep 30   # let the GPU memory be released

.venv/bin/python -u xlmr_xl/scripts/xlmr_xl_cv.py 2>&1 | grep --line-buffered -v -i -E "$FILTER" >> $LOG
if grep -q -i "out of memory" $LOG && ! grep -q "XL PIPELINE FINISHED" $LOG; then
  echo "[$(date '+%F %T')] OOM -> retrying with batch 8 x accum 4 (cached folds are kept)" | tee -a $LOG
  BATCH=8 ACCUM=4 .venv/bin/python -u xlmr_xl/scripts/xlmr_xl_cv.py 2>&1 | grep --line-buffered -v -i -E "$FILTER" >> $LOG
fi
echo "[$(date '+%F %T')] DONE" >> $LOG
