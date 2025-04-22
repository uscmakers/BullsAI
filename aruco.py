import cv2


class ArucoManager:

    def __init__(self, adaptive_thresh_constant = 3, output_image = None) -> None:
        self.output_image = output_image
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = adaptive_thresh_constant
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect_markers(self, image: cv2.typing.MatLike) -> dict:
        """
        Detects ArUco markers in the given image and returns their positions.
        """
        if image is None:
            raise ValueError("Image is None")
        

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        aruco_reference_points = {}
        aruco_corner_points = []

        if ids is not None and len(ids) > 0:
            if self.output_image:
                cv2.aruco.drawDetectedMarkers(self.output_image, corners, ids)
            
            for i, corner in enumerate(corners):
                center_x = int(corner[0][:, 0].mean())
                center_y = int(corner[0][:, 1].mean())
                marker_id = ids[i][0]
                aruco_reference_points[marker_id] = (center_x, center_y)

                if self.output_image:
                    cv2.putText(self.output_image, f"ID:{marker_id}", (center_x-20, center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                for pt in corner[0]:
                    aruco_corner_points.append(pt)
                
            # (Optional) Draw the convex hull of all marker corners as before
            if len(aruco_corner_points) > 0 and self.output_image:
                import numpy as np

                aruco_corner_points = np.array(aruco_corner_points, dtype=np.int32)
                hull = cv2.convexHull(aruco_corner_points)
                cv2.polylines(self.output_image, [hull], True, (0, 255, 255), 2)
            
            return aruco_reference_points
        
        else:
            raise ValueError("No markers detected")