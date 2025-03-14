import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_drumstick_contour(image_path, output_path="drumstick_contour.png"):
    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용 (노이즈 감소)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 임계값 적용 (이진화) - 닭다리만 남기기
    _, binary_mask = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # 모폴로지 연산 (노이즈 제거 및 형태 보정)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 윤곽선 찾기 (닭다리 모양일 가능성이 높음)
    largest_contour = max(contours, key=cv2.contourArea)

    # 윤곽선만 남기기 위한 검은색 배경 생성
    contour_image = np.zeros_like(image_rgb)

    # 윤곽선 그리기 (초록색, 두께 2)
    cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)

    # 결과 저장
    cv2.imwrite(output_path, cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))

    # 결과 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(contour_image)
    plt.axis("off")
    plt.title("Detected Chicken Drumstick Contour")
    plt.show()

    return output_path

# 코드 실행 예시
image_path = "your_image.jpg"  # 닭
