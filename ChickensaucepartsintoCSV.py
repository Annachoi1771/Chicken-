import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_black_contours_to_csv(image_path, output_csv="black_contours.csv"):
    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 검정색 및 어두운 부분을 포함하는 RGB 범위 설정
    lower_rgb = np.array([25, 20, 5])  # 어두운 갈색-검정 포함
    upper_rgb = np.array([50, 45, 20])  # 색상 변화를 포함

    # 검정색 부분만 마스킹
    color_mask = cv2.inRange(image_rgb, lower_rgb, upper_rgb)

    # 모폴로지 연산을 사용하여 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # 윤곽선 감지
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선 데이터 저장을 위한 리스트
    contour_data = []

    # 윤곽선 좌표 데이터 저장
    for contour in contours:
        for point in contour:
            x, y = point[0]
            contour_data.append([x, y])

    # Pandas DataFrame 생성
    df_contours = pd.DataFrame(contour_data, columns=["X", "Y"])

    # CSV 파일로 저장
    df_contours.to_csv(output_csv, index=False)

    print(f"CSV 파일이 저장되었습니다: {output_csv}")

    return output_csv

# 코드 실행 예시
image_path = "your_image.jpg"  # 이미지 파일 경로
detect_black_contours_to_csv(image_path)
