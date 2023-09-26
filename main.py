import cv2
import numpy as np
import enum
import scipy.interpolate


def apply_fisheye_distortion(image_path, mask=False):
    # 이미지 불러오기
    if mask == True:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image[image == 255] = 12
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지 크기 가져오기
    height, width = image.shape[:2]

    # 카메라 매트릭스 생성
    focal_length = width / 4
    center_x = width / 2
    center_y = height / 2
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=np.float32)

    # 왜곡 계수 생성
    dist_coeffs = np.array([0, 0.2, 0, 0], dtype=np.float32)

    # 왜곡 보정
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    print(undistorted_image.shape)
    undistorted_image = undistorted_image[40:height,340:1740,:]
    print(undistorted_image.shape)
    return (undistorted_image)

# 이미지 로드
image_path = './train_source_image/TRAIN_SOURCE_0032.png'  # 이미지 경로를 적절하게 변경해주세요


# fisheye distortion 적용
distorted_image = apply_fisheye_distortion(image_path)

# 결과 이미지 시각화
output_path = './distorted_image.jpg'
cv2.imwrite(output_path, distorted_image)
