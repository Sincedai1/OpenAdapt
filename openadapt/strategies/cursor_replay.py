import numpy as np
import cv2
from typing import Tuple, Optional
from .base import VanillaReplayStrategy

class CursorReplayStrategy(VanillaReplayStrategy):
    def __init__(self, 
                 dot_radius: int = 5, 
                 dot_color: Tuple[int, int, int] = (0, 0, 255),
                 enable_self_correction: bool = False,
                 confidence_threshold: float = 0.7):
        """Initialize the CursorReplayStrategy.
        
        Args:
            dot_radius: Radius of the cursor dot in pixels
            dot_color: Color of the dot in BGR format
            enable_self_correction: Whether to enable visual self-correction
            confidence_threshold: Threshold for self-correction confidence
        """
        self.dot_radius = dot_radius
        self.dot_color = dot_color
        self.enable_self_correction = enable_self_correction
        self.confidence_threshold = confidence_threshold
        self.last_correction = None

    def _analyze_target_region(self, screenshot: np.ndarray, target_coords: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Analyze the region around the target coordinates for better positioning.
        
        Uses edge detection and contour analysis to find better click targets.
        """
        x, y = target_coords
        region_size = self.dot_radius * 4
        
        # Extract region of interest
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(screenshot.shape[1], x + region_size)
        y2 = min(screenshot.shape[0], y + region_size)
        roi = screenshot[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour near the center
        center_x, center_y = region_size, region_size
        best_contour = None
        min_dist = float('inf')
        
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            if dist < min_dist:
                min_dist = dist
                best_contour = contour
        
        if best_contour is None:
            return None
            
        # Get the center of the best contour
        M = cv2.moments(best_contour)
        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
        
        return (cx, cy)

    def process_target(self, screenshot: np.ndarray, target_coords: Tuple[int, int]) -> Tuple[int, int]:
        """Process the target coordinates with optional self-correction.
        
        Args:
            screenshot: The current screenshot
            target_coords: The original target coordinates
            
        Returns:
            Tuple[int, int]: The processed target coordinates
        """
        # Ensure coordinates are within bounds
        x, y = target_coords
        height, width = screenshot.shape[:2]
        x = min(max(0, x), width - 1)
        y = min(max(0, y), height - 1)
        
        if not self.enable_self_correction:
            return (x, y)
            
        # Analyze the target region for better positioning
        corrected_coords = self._analyze_target_region(screenshot, (x, y))
        
        if corrected_coords is not None:
            # Store the correction for visualization
            self.last_correction = {
                'original': (x, y),
                'corrected': corrected_coords
            }
            return corrected_coords
            
        return (x, y)

    def visualize_target(self, screenshot: np.ndarray, target_coords: Tuple[int, int]) -> np.ndarray:
        """Visualize the target with a red dot and optional correction path.
        
        Args:
            screenshot: The current screenshot
            target_coords: The target coordinates
            
        Returns:
            np.ndarray: The screenshot with visualization overlay
        """
        visualization = screenshot.copy()
        
        # Draw correction path if available
        if self.enable_self_correction and self.last_correction is not None:
            orig = self.last_correction['original']
            corr = self.last_correction['corrected']
            
            # Draw path from original to corrected position
            cv2.line(visualization, orig, corr, (0, 255, 0), 1)
            # Draw original position (smaller, yellow dot)
            cv2.circle(visualization, orig, self.dot_radius-2, (0, 255, 255), -1)
        
        # Draw target dot with white outline for visibility
        cv2.circle(visualization, target_coords, self.dot_radius+2, (255, 255, 255), -1)
        cv2.circle(visualization, target_coords, self.dot_radius, self.dot_color, -1)
        
        return visualization
