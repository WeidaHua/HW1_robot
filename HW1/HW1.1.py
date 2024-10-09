import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('IMAGE.png')

# Check if the image is successfully read
if image is None:
    print("Failed to read the image, please check the path.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Manually divide the image into three parts
    height = gray_image.shape[0]
    part1 = gray_image[0:height//3, :]
    part2 = gray_image[height//3:2*height//3, :]
    part3 = gray_image[2*height//3:, :]
    
    # Adjust the contrast and brightness for each part
    enhanced_part1 = cv2.convertScaleAbs(part1, alpha=4.1, beta=60)
    enhanced_part2 = cv2.convertScaleAbs(part2, alpha=1.03, beta=30)
    enhanced_part3 = cv2.convertScaleAbs(part3, alpha=1.71, beta=50)
    
    # Concatenate the three parts into one complete image
    enhanced_image = cv2.vconcat([enhanced_part1, enhanced_part2, enhanced_part3])

    # Display the original image and the processed image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title("Enhanced Image")

    plt.show()




