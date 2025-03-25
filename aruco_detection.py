import cv2
import numpy as np

def normalize_coordinates(aruco_points, led_position, board_size=(800, 800)):
    """
    Normalize coordinates from camera view to standardized dartboard coordinates
    
    Args:
    - aruco_points: Dictionary of ArUco marker centers {marker_id: (x, y)}
    - led_position: (x, y) of the LED/dart position
    - board_size: Size of the normalized coordinate system
    
    Returns:
    - Normalized coordinates on the dartboard
    """
    # Order points in a consistent manner (top-left, top-right, bottom-right, bottom-left)
    # This assumes you know which marker corresponds to which corner
    # You might need to adjust the order based on your specific marker placement
    src_points = np.float32([
        aruco_points[marker_id] for marker_id in sorted(aruco_points.keys())
    ])
    
    # Destination points (normalized board coordinates)
    dst_points = np.float32([
        [0, 0],  # Top-left
        [board_size[0], 0],  # Top-right
        [board_size[0], board_size[1]],  # Bottom-right
        [0, board_size[1]]  # Bottom-left
    ])
    
    # Compute perspective transform
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Transform LED position
    led_homography = np.array([led_position[0], led_position[1], 1])
    normalized_point = perspective_matrix @ led_homography
    
    # Normalize the point
    normalized_point /= normalized_point[2]
    print(int(normalized_point[0]), int(normalized_point[1]))
    return (int(normalized_point[0]), int(normalized_point[1]))

def calculate_dart_score(normalized_point, board_size=(800, 800)):
    """
    Calculate dart score based on normalized coordinates
    
    Args:
    - normalized_point: (x, y) of normalized dart position
    - board_size: Size of the normalized coordinate system
    
    Returns:
    - Dart score
    """
    # Calculate distance from center
    center_x, center_y = board_size[0] // 2, board_size[1] // 2
    distance = np.sqrt(
        ((normalized_point[0] - center_x) ** 2) + 
        ((normalized_point[1] - center_y) ** 2)
    )
    
    # Define scoring zones (these are approximate and should be calibrated)
    # Zones are proportional to board radius
    radius = board_size[0] // 2
    scoring_zones = [
        (radius * 0.1, 50),   # Bullseye
        (radius * 0.3, 25),   # Inner bullseye
        (radius * 0.5, 20),   # Triples ring
        (radius * 0.7, 1),    # Outer rings scoring 1-20
        (radius, 0)           # Miss
    ]
    
    # Determine score based on distance
    for zone_radius, score in scoring_zones:
        if distance <= zone_radius:
            return score
    
    return 0  # Miss

# Example usage in your main script
def process_dart_throw(image, aruco_reference_points, led_position):
    # Normalize LED position
    normalized_point = normalize_coordinates(aruco_reference_points, led_position)
    
    # Calculate score
    dart_score = calculate_dart_score(normalized_point)
    
    print(f"Normalized Point: {normalized_point}")
    print(f"Dart Score: {dart_score}")
    
    return normalized_point, dart_score


# Load image
image = cv2.imread("newimage.JPEG")
height, width, _ = image.shape

# Make a copy of the original image for visualization
output_image = image.copy()

# ------------ ARUCO DETECTION (REFERENCE POINTS) ------------
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create dictionary - DICT_ARUCO_ORIGINAL
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# Create parameters with adaptiveThreshConstant = 3
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshConstant = 3

# Create detector
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Detect markers
corners, ids, rejected = detector.detectMarkers(gray)

# Create a coordinate system from ArUco markers
aruco_reference_points = {}
aruco_corner_points = []  # Store all corner points for boundary calculation

if ids is not None and len(ids) > 0:
    print(f"Successfully detected {len(ids)} ArUco markers")
    
    # Draw markers on the output image
    cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
    
    # Store the center and corners of each marker
    for i, corner in enumerate(corners):
        # Calculate center point (for visualization and reference)
        center_x = int(corner[0][:, 0].mean())
        center_y = int(corner[0][:, 1].mean())
        marker_id = ids[i][0]
        
        # Store marker centers in dictionary (for reference)
        aruco_reference_points[marker_id] = (center_x, center_y)
        
        # Add ID text at marker center
        cv2.putText(output_image, f"ID:{marker_id}", (center_x-20, center_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Extract all four corner points of each marker
        for corner_point in corner[0]:
            aruco_corner_points.append(corner_point)
    print(aruco_reference_points)
    # If we have detected ArUco markers, compute dartboard boundary using outer corners
    if len(aruco_corner_points) > 0:
        # Convert to numpy array
        aruco_corner_points = np.array(aruco_corner_points, dtype=np.int32)
        
        # Compute the convex hull or bounding area of the outer corners of markers
        hull = cv2.convexHull(aruco_corner_points)
        cv2.polylines(output_image, [hull], True, (0, 255, 255), 2)
        
        # Calculate the approximate center of the dartboard
        # (based on the geometric center of the convex hull)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            dartboard_center_x = int(M["m10"] / M["m00"])
            dartboard_center_y = int(M["m01"] / M["m00"])
        else:
            # Fallback to mean of corner points
            dartboard_center_x = int(np.mean(aruco_corner_points[:, 0]))
            dartboard_center_y = int(np.mean(aruco_corner_points[:, 1]))
            
        cv2.circle(output_image, (dartboard_center_x, dartboard_center_y), 5, (255, 0, 255), -1)
        cv2.putText(output_image, "Est. Dartboard Center", (dartboard_center_x+10, dartboard_center_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # ------------ LED DETECTION ------------
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding
        _, thresh = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY)
        
        # Find contours for LED
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Variables to store LED position
        led_position = None
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Ignore borders
            if 0 <= x <= int(0.1 * width) or int(0.9 * width) <= x <= width:
                continue
            if 0 <= y <= int(0.1 * height) or int(0.9 * height) <= y <= height:
                continue
            # Ignore small boxes (noise)
            if w < 25 or h < 25:
                continue
            
            # Store LED center
            led_center_x = x + w//2
            led_center_y = y + h//2
            led_position = (led_center_x, led_center_y)
            
            # Draw LED
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(output_image, "LED", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            #Score:
            if led_position:
                normalized_point, dart_score = process_dart_throw(output_image, aruco_reference_points, led_position)

                # Optionally, visualize the normalized point
                cv2.circle(output_image, (normalized_point[0] + aruco_reference_points[1][0], normalized_point[1] + aruco_reference_points[1][1]), 5, (0, 0, 255), -1)
                cv2.putText(output_image, f"Score: {dart_score}", 
                    (normalized_point[0]+10, normalized_point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
else:
    print("No ArUco markers detected")

# Show result
cv2.imshow("Dartboard Analysis", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

