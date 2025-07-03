#!/bin/bash

# =================================================================
# SCRIPT ĐỂ CHẠY ĐÁNH GIÁ CUỐI CÙNG VỚI 5 SEED
# - Sử dụng bộ siêu tham số tốt nhất đã tìm được từ grid search.
# - Chạy với danh sách 5 seed trong config.ini để có kết quả ổn định.
# =================================================================

# --- CẤU HÌNH THÍ NGHIỆM (CHỈNH SỬA Ở ĐÂY) ---
TASK_NAME="FewRel"      # <<< THAY ĐỔI DATASET Ở ĐÂY (FewRel hoặc TACRED)
BEST_LAMBDA_4=0.5       # <<< THAY ĐỔI GIÁ TRỊ LAMBDA_4 TỐT NHẤT BẠN TÌM ĐƯỢC
# -----------------------------------------------

# Các tham số cố định
LAMBDA_1=1
LAMBDA_2=1
LAMBDA_3=0.25
TEMPERATURE=0.01
DISTANCE_THRESHOLD=0.1

LOG_FILE="final_evaluation_${TASK_NAME}_lambda4_${BEST_LAMBDA_4}.log"

echo "Bắt đầu chạy đánh giá cuối cùng cho ${TASK_NAME} với 5 seed..."
echo "Các siêu tham số:"
echo "  lambda_4 = ${BEST_LAMBDA_4}"
echo "Log sẽ được lưu tại: ${LOG_FILE}"

# Chạy train.py mà KHÔNG có tham số --seed, để nó tự đọc 5 seed từ config.ini
# Chạy trong nền với `nohup` và `&` để không bị ngắt khi đóng terminal
nohup python3 -u train.py \
  --task_name "${TASK_NAME}" \
  --lambda_1 ${LAMBDA_1} \
  --lambda_2 ${LAMBDA_2} \
  --lambda_3 ${LAMBDA_3} \
  --lambda_4 ${BEST_LAMBDA_4} \
  --temperature ${TEMPERATURE} \
  --distance_threshold ${DISTANCE_THRESHOLD} > "${LOG_FILE}" 2>&1 &

echo "Đã bắt đầu tiến trình chạy trong nền. PID: $!"
echo "Để theo dõi tiến trình, sử dụng lệnh: tail -f ${LOG_FILE}"
