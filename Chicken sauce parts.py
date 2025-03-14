import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_black_contours(image_path, output_path="black_contours.png"):
    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # RGB 색상 범위 설정 (약간의 검정 포함)
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

    # 윤곽선만 남기기 위한 검은색 배경 생성
    contour_image = np.zeros_like(image_rgb)

    # 윤곽선 그리기 (하얀색, 두께 2)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

    # 결과 저장
    cv2.imwrite(output_path, cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))

    # 결과 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(contour_image)
    plt.axis("off")
    plt.title("Contours of Slightly Dark Regions")
    plt.show()

    return output_path

# 코드 실행 예시
image_path = "your_image.jpg"  # 이미지 파일 경로
detect_black_contours(image_path)
