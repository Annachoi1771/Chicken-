import cv2
import numpy as np
import pandas as pd

def save_contour_as_csv(image_path, output_csv="drumstick_contour.csv"):
    # 이미지 로드
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용 (노이즈 제거)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 이진화 (Threshold)
    _, binary_mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # 모폴로지 연산 (노이즈 제거)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 윤곽선 찾기 (닭다리 모양일 가능성이 높음)
    largest_contour = max(contours, key=cv2.contourArea)

    # 윤곽선 좌표 데이터 추출
    contour_points = largest_contour.reshape(-1, 2)  # (x, y) 형식 변환

    # Pandas DataFrame 생성
    df_contour = pd.DataFrame(contour_points, columns=["X", "Y"])

    # CSV 파일로 저장
    df_contour.to_csv(output_csv, index=False)

    print(f"CSV 파일이 저장되었습니다: {output_csv}")

# 코드 실행 예시
image_path = "your_image.jpg"  # 닭다리 이미지 경로
save_contour_as_csv(image_path)
