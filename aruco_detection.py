import cv2
import numpy as np


def normalize_point(point, perspective_matrix):
    """Apply the perspective transform to a single point."""
    homogenous = np.array([point[0], point[1], 1])
    norm_pt = perspective_matrix @ homogenous
    norm_pt /= norm_pt[2]
    return (int(norm_pt[0]), int(norm_pt[1]))

def normalize_coordinates(aruco_points, led_position, board_size=(800, 800)):
    """
    Normalize coordinates from camera view to standardized dartboard coordinates.
    
    Returns:
    - Normalized LED point
    - Perspective transform matrix for further use
    """
    # Order points (make sure your marker IDs match your expected order)
    src_points = np.float32([
        aruco_points[marker_id] for marker_id in sorted(aruco_points.keys())
    ])
    
    # Destination points for the normalized board
    dst_points = np.float32([
        [0, 0],  # Top-left
        [board_size[0], 0],  # Top-right
        [board_size[0], board_size[1]],  # Bottom-right
        [0, board_size[1]]  # Bottom-left
    ])
    
    # Compute perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Transform LED position
    normalized_led = normalize_point(led_position, perspective_matrix)
    print("Normalized LED:", normalized_led)
    return normalized_led, perspective_matrix

def calculate_dart_score(normalized_point, board_size=(800, 800)):
    """
    Calculate dart score based on normalized coordinates.
    """
    center_x, center_y = board_size[0] // 2, board_size[1] // 2
    distance = np.sqrt((normalized_point[0] - center_x) ** 2 + 
                       (normalized_point[1] - center_y) ** 2)
    
    # Define scoring zones (approximate, should be calibrated)
    radius = board_size[0] // 2
    scoring_zones = [
        (radius * 0.1, 50),   # Bullseye
        (radius * 0.3, 25),   # Inner bullseye
        (radius * 0.5, 20),   # Triples ring
        (radius * 0.7, 1),    # Outer rings scoring 1-20
        (radius, 0)           # Miss
    ]
    
    for zone_radius, score in scoring_zones:
        if distance <= zone_radius:
            return score
    return 0  # Miss

def process_dart_throw(image, aruco_reference_points, led_position, board_size=(800, 800)):
    normalized_led, perspective_matrix = normalize_coordinates(aruco_reference_points, led_position, board_size)
    dart_score = calculate_dart_score(normalized_led, board_size)
    
    # Normalize all ArUco marker centers
    normalized_markers = {}
    for marker_id, center in aruco_reference_points.items():
        normalized_markers[marker_id] = normalize_point(center, perspective_matrix)
    
    # Compute average (center) of the four markers
    avg_x = int(sum(pt[0] for pt in normalized_markers.values()) / len(normalized_markers))
    avg_y = int(sum(pt[1] for pt in normalized_markers.values()) / len(normalized_markers))
    normalized_center = (avg_x, avg_y)
    
    # Use warpPerspective to get a normalized view of the original image
    normalized_board = cv2.warpPerspective(image, perspective_matrix, board_size)
    
    # Draw normalized marker centers on the warped image
    for marker_id, pt in normalized_markers.items():
        cv2.circle(normalized_board, pt, 5, (0, 255, 0), -1)
        cv2.putText(normalized_board, f"ID:{marker_id}", (pt[0]+5, pt[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Draw the computed normalized center
    cv2.circle(normalized_board, normalized_center, 7, (0, 0, 255), -1)
    cv2.putText(normalized_board, "Center", (normalized_center[0]+10, normalized_center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Optionally, also draw the normalized LED point for reference
    cv2.circle(normalized_board, normalized_led, 5, (255, 0, 0), -1)
    cv2.putText(normalized_board, "LED", (normalized_led[0]+5, normalized_led[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display the warped (normalized) image
    cv2.imshow("Normalized Dartboard", normalized_board)
    cv2.waitKey(0)
    cv2.destroyWindow("Normalized Dartboard")
    
    return normalized_led, dart_score


# ------------------ (Rest of your detection code remains unchanged) ------------------ #

# Load image
image = cv2.imread("newimage.JPEG")
height, width, _ = image.shape
output_image = image.copy()

# ARUCO DETECTION
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshConstant = 3
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, rejected = detector.detectMarkers(gray)

aruco_reference_points = {}
aruco_corner_points = []

if ids is not None and len(ids) > 0:
    print(f"Successfully detected {len(ids)} ArUco markers")
    cv2.aruco.drawDetectedMarkers(output_image, corners, ids)
    
    for i, corner in enumerate(corners):
        center_x = int(corner[0][:, 0].mean())
        center_y = int(corner[0][:, 1].mean())
        marker_id = ids[i][0]
        aruco_reference_points[marker_id] = (center_x, center_y)
        cv2.putText(output_image, f"ID:{marker_id}", (center_x-20, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        for pt in corner[0]:
            aruco_corner_points.append(pt)
    
    print("Aruco Reference Points:", aruco_reference_points)
    
    # (Optional) Draw the convex hull of all marker corners as before
    if len(aruco_corner_points) > 0:
        aruco_corner_points = np.array(aruco_corner_points, dtype=np.int32)
        hull = cv2.convexHull(aruco_corner_points)
        cv2.polylines(output_image, [hull], True, (0, 255, 255), 2)
    
    # ------------ LED DETECTION ------------
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    led_position = None
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if 0 <= x <= int(0.1 * width) or int(0.9 * width) <= x <= width:
            continue
        if 0 <= y <= int(0.1 * height) or int(0.9 * height) <= y <= height:
            continue
        if w < 25 or h < 25:
            continue
        
        led_center = (x + w//2, y + h//2)
        led_position = led_center
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(output_image, "LED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        if led_position:
            normalized_led, dart_score = process_dart_throw(output_image, aruco_reference_points, led_position)
            # Optionally, mark the LED on the original image as well
            cv2.circle(output_image, led_position, 5, (255, 0, 0), -1)
            cv2.putText(output_image, f"Score: {dart_score}", (led_position[0]+10, led_position[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            break
else:
    print("No ArUco markers detected")

cv2.imshow("Dartboard Analysis", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
