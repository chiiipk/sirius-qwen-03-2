# sirius-qwen-03-2
bash run_grid_search.sh

bash run_final_evaluation.sh

Can run like this:
python3 -u train.py \
  --task_name "FewRel" \
  --lambda_1 1 \
  --lambda_2 1 \
  --lambda_3 0.25 \
  --lambda_4 0.5 \
  --temperature 0.01 \
  --distance_threshold 0.1
