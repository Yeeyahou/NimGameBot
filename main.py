from ultralytics import YOLO
import cv2
import time

# Grundy수 구하기
def get_grundy_number(coin_arr):
    # Sub Grundy값 구하기
    sub_grundy = []
    for row in coin_arr:
        sub_grundy.append(len(row))

    # XOR 연산으로 Grundy 값 구하기
    grundy = 0
    for value in sub_grundy:
        grundy = grundy ^ value  # 누적 XOR

    return grundy

# 필승 전략 계산(제거할 동전의 수, 행 리턴)
def find_optimal_move(coin_arr):
    current_grundy = get_grundy_number(coin_arr)
    for row_index, row in enumerate(coin_arr):
        target = len(row) ^ current_grundy
        if target < len(row):
            return row_index, len(row) - target
    return None, None

model = YOLO("runs/detect/train/weights/best.pt") #욜로모델 불러오기
cap = cv2.VideoCapture(0)

last_detection_time = 0
results = None #검출 결과
num_coins = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_detection_time > 0.2: #성능 최적화를 위해 0.2초마다 동전 검출
        results = model.predict(frame, conf=0.3, imgsz=700)
        num_coins = len(results[0].boxes) # 동전 총 수량
        last_detection_time = current_time
    display_frame = frame.copy()

    if results is not None and results[0].boxes: # 동전이 검출 됐을 때 알고리즘 수행
        centers = [] # 좌표 정보(중앙, 최소 최대에 대한 x,y값)저장 배열
        avg_width = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centers.append((cx, cy, x1, y1, x2, y2))
            avg_width += x2 - x1

        avg_width = avg_width / num_coins if num_coins > 0 else 0 #동전 평균 변의 길이
        coin_arr = [] # 게임 판 배열
        scan_line = min(c[1] for c in centers) # 스캔라인을 가장 위 동전부터 잡기
        offset = 0.5 # 검사 오프셋

        while scan_line < frame.shape[0]:
            row = []
            ok = False
            for c in centers:
                if abs(c[1] - scan_line) < offset * (c[5]-c[3]): #현재 검사하는 동전이 오프셋 범위 내인지 판별
                    row.append(c)
                    if ok == False:
                        ok = True
                        scan_line = c[1] #스캔라인을 해당 동전중앙으로 확정

            #각 동전의 행을 x좌표기준 오름차순 정렬(제거해야 할 동전을 오른쪽 동전부터 추천 해야하기 때문이다)
            if row:
                row.sort(key=lambda x: x[0])
                coin_arr.append(row)
            scan_line += avg_width #다음줄 스캔

        optimal_row, remove_count = find_optimal_move(coin_arr) #각 행을 검사하며 그런디수를 구해 제가할 동전 수, 행을 구한다
        overlay = display_frame.copy() #제거할 동전 표시할 프레임 하나 생성
        for row_index, row in enumerate(coin_arr):
            for i, (cx, cy, x1, y1, x2, y2) in enumerate(row):
                if row_index == optimal_row and i >= len(row) - remove_count:
                    # 가져가야할 동전에 빨강 박스 그리기
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

        # 원본 프레임에 overlay를 알파 블렌딩
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

        #제거 해야할 행이 없으면 즉, 이기는 수가 없으면 문구 출력
        if optimal_row is None:
            cv2.putText(display_frame, "No winning move exists", (370, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #전체 동전 수 출력
    cv2.putText(display_frame, f"Total Coins: {num_coins}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("NimGameBot", display_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
