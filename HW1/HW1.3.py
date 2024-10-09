import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Size of the chessboard and each square's size
chessboard_size = (7, 9)
square_size = 0.019  # Size of each square in meters

# Prepare the 3D world coordinates of the calibration points (z=0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

def calibrate_camera(image_folder):
    obj_points = []  # 3D world coordinates
    img_points = []  # 2D image coordinates
    images = glob.glob(f'{image_folder}/*.jpg')  # Read all images in the folder
    
    for image_file in images:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
    
    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return camera_matrix, dist_coeffs

# Read images from the 1 meter and 2-3 meter folders and perform calibration
camera_matrix_1m, dist_coeffs_1m = calibrate_camera('1m_images')
camera_matrix_2m_3m, dist_coeffs_2m_3m = calibrate_camera('2m_3m_images')

# Output results
print("Calibration results within 1 meter:")
print("Camera Matrix (K matrix):\n", camera_matrix_1m)
print("Distortion Coefficients (k1, k2):\n", dist_coeffs_1m[:2])

print("\nCalibration results between 2 to 3 meters:")
print("Camera Matrix (K matrix):\n", camera_matrix_2m_3m)
print("Distortion Coefficients (k1, k2):\n", dist_coeffs_2m_3m[:2])

# Compare results
print("\nK matrix change:\n", camera_matrix_2m_3m - camera_matrix_1m)
print("Distortion Coefficients change (k1, k2):\n", dist_coeffs_2m_3m[:2] - dist_coeffs_1m[:2])

# Generate visualization charts for differences
k_matrix_change = camera_matrix_2m_3m - camera_matrix_1m
dist_change = dist_coeffs_2m_3m - dist_coeffs_1m

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the K matrix changes
axs[0].imshow(k_matrix_change, cmap='coolwarm', interpolation='none')
axs[0].set_title('K Matrix Change (2-3m vs 1m)')
axs[0].set_xticks([0, 1, 2])
axs[0].set_yticks([0, 1, 2])
axs[0].set_xticklabels(['fx', '0', 'cx'])
axs[0].set_yticklabels(['fy', '0', 'cy'])
for i in range(3):
    for j in range(3):
        axs[0].text(j, i, f'{k_matrix_change[i, j]:.2f}', ha='center', va='center', color='black')

# Plot the distortion coefficients changes
axs[1].bar(['k1', 'k2', 'p1', 'p2', 'k3'], dist_change.flatten(), color='blue')
axs[1].set_title('Distortion Coefficients Change (2-3m vs 1m)')
axs[1].set_ylabel('Coefficient Change')

plt.tight_layout()
plt.show()



                     
