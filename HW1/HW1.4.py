import cv2
import numpy as np

# Define camera intrinsic matrix K and distortion coefficients D
K = np.array([[495.367, 0, 326.802], 
              [0, 495.561, 245.859], 
              [0, 0, 1]], dtype=float)
D = np.array([-0.00468472, 0.13135612, 0.00521043, 0.0033172, -0.31450576], dtype=float)

# Define the 3D coordinates of the cube
cube_size = 0.142  # Cube edge length in meters, matching the size of the marker
marker_length = cube_size  # Set the marker size to be the same as the cube's edge length

# Redefine the 3D coordinates of the cube so that its bottom four vertices coincide with the corners of the marker
# The cube's bottom coincides with the marker, and the top extends upwards
object_points = np.array([
    [-0.5*marker_length, -0.5*marker_length, 0],  # Four corners of the marker
    [0.5*marker_length, -0.5*marker_length, 0],
    [0.5*marker_length, 0.5*marker_length, 0],
    [-0.5*marker_length, 0.5*marker_length, 0],
    [-0.5*marker_length, -0.5*marker_length, marker_length],  # Four corners of the marker
    [0.5*marker_length, -0.5*marker_length, marker_length],
    [0.5*marker_length, 0.5*marker_length, marker_length],
    [-0.5*marker_length, 0.5*marker_length, marker_length]
], dtype=np.float32)

# Load ArUco dictionary and detection parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# List of image files to process
image_files = ['TAGS/image_1.jpg','TAGS/image_2.jpg','TAGS/image_3.jpg']  # Replace with your image filenames

for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Unable to load image {image_file}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        # Draw detected marker boundaries
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # Estimate pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, K, D)

        for rvec, tvec in zip(rvecs, tvecs):
            # Project the 3D points of the cube onto the image plane
            imgpts, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Draw the edges of the hollow cube
            img = image.copy()

            # Bottom quadrilateral (red lines, coinciding with the marker)
            cv2.drawContours(img, [imgpts[:4]], -1, (0, 0, 255), 3)
            
            # Four vertical edges (red lines)
            for i, j in zip(range(4), range(4, 8)):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 0, 255), 3)
            
            # Top quadrilateral (red lines)
            cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

            # Display the result
            cv2.imshow('Image with Red Cube', img)
            cv2.waitKey(0)
    else:
        print("No ArUco markers detected.")

cv2.destroyAllWindows()

