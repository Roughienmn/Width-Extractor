#every n pixels, find white pixel that comes immediately before or after black pixel

#use white pixels to find nearest white pixel that is also next to a black pixel via breadth first search

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist

def extract_road_width(road_mask):
    # Convert to grayscale if needed
    if len(road_mask.shape) == 3:
        road_mask = cv2.cvtColor(road_mask, cv2.COLOR_BGR2GRAY)
    
    # Binarize the mask
    _, binary_mask = cv2.threshold(road_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Skeletonize the road to get centerline
    skeleton = skeletonize(binary_mask > 0).astype(np.uint8) * 255
    
    # Get contours of the road mask (road boundaries)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    road_boundary = np.zeros_like(binary_mask)
    cv2.drawContours(road_boundary, contours, -1, 255, 1)
    
    # Find centerline points
    centerline_points = np.column_stack(np.where(skeleton > 0))
    
    # Find boundary points
    boundary_points = np.column_stack(np.where(road_boundary > 0))
    
    widths = []
    for point in centerline_points:
        # Compute distances from centerline point to all boundary points
        distances = cdist([point], boundary_points)
        min_dist = np.min(distances)  # Closest boundary distance
        widths.append(min_dist * 2)  # Full width

    avg_width = np.mean(widths) if widths else 0  # Compute mean width

    cv2.imwrite("skeleton_output.png", skeleton)

    # Optional: Display the result
    cv2.imshow("Skeletonized Road", skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return avg_width, widths  # Return avg width and local width array

# Load road mask (example)
road_mask = cv2.imread("Test Files/24779275_15.tif", cv2.IMREAD_GRAYSCALE)
avg_width, width_distribution = extract_road_width(road_mask)
print(f"Estimated Road Width: {avg_width:.2f} pixels")