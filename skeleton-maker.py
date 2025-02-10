import cv2
import numpy as np
from skimage.morphology import skeletonize

def extract_skeleton(road_mask):
    # Convert to grayscale if needed
    if len(road_mask.shape) == 3:
        road_mask = cv2.cvtColor(road_mask, cv2.COLOR_BGR2GRAY)
    
    # Binarize the mask
    _, binary_mask = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Skeletonize the road mask
    skeleton = skeletonize(binary_mask > 0).astype(np.uint8) * 255  # Convert boolean to 0-255 image
    
    return skeleton

# Load road mask (example)
road_mask = cv2.imread("Test Files/24779275_15.tif", cv2.IMREAD_GRAYSCALE)

# Extract skeleton
skeleton_image = extract_skeleton(road_mask)

# Save the skeleton image
cv2.imwrite("skeleton_output.png", skeleton_image)

# Optional: Display the result
cv2.imshow("Skeletonized Road", skeleton_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
