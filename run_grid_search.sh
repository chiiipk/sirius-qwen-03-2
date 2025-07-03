#!/bin/bash

# =================================================================
# SCRIPT ĐỂ CHẠY GRID SEARCH TÌM SIÊU THAM SỐ TỐT NHẤT
# - Chạy với một seed cố định (--seed 42) để có kết quả nhanh.
# - Lưu log của mỗi lần chạy vào một file riêng trong thư mục logs.
# =================================================================

# --- CẤU HÌNH THÍ NGHIỆM (CHỈNH SỬA Ở ĐÂY) ---
TASK_NAME="FewRel" # <<< THAY ĐỔI DATASET Ở ĐÂY (FewRel hoặc TACRED)
# -----------------------------------------------

# Các tham số cố định
LAMBDA_1=1
LAMBDA_2=1
LAMBDA_3=0.25
TEMPERATURE=0.01
DISTANCE_THRESHOLD=0.1
GRID_SEARCH_SEED=42 # Dùng một seed cố định để tìm siêu tham số nhanh

# Các giá trị lambda_4 để thử
LAMBDA_4_VALUES=(0.1 0.25 0.5 0.75 1.0)

# Tạo thư mục lưu log nếu chưa có
LOG_DIR="grid_search_logs_${TASK_NAME}"
mkdir -p ${LOG_DIR}

echo "Bắt đầu Grid Search cho lambda_4 trên dataset ${TASK_NAME}..."

for LAMBDA_4 in "${LAMBDA_4_VALUES[@]}"
do
  LOG_FILE="${LOG_DIR}/lambda4_${LAMBDA_4}.log"
  echo "----------------------------------------------------"
  echo "Đang chạy với lambda_4=${LAMBDA_4}. Log được lưu tại: ${LOG_FILE}"
  echo "----------------------------------------------------"
  
  # Chạy train.py với một SEED CỐ ĐỊNH (--seed)
  python3 train.py \
    --task_name "${TASK_NAME}" \
    --lambda_1 ${LAMBDA_1} \
    --lambda_2 ${LAMBDA_2} \
    --lambda_3 ${LAMBDA_3} \
    --lambda_4 ${LAMBDA_4} \
    --temperature ${TEMPERATURE} \
    --distance_threshold ${DISTANCE_THRESHOLD} \
    --seed ${GRID_SEARCH_SEED} > "${LOG_FILE}" 2>&1
done

echo "Grid search hoàn tất. Vui lòng kiểm tra các file log trong thư mục '${LOG_DIR}'."
echo "Sau khi tìm được lambda_4 tốt nhất, hãy cập nhật nó vào file 'run_final_evaluation.sh'."
