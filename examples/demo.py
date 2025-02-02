import numpy as np
import cv2
from replay_strategies.cursor_replay import CursorReplayStrategy

def create_test_image():
    """Create a test image with various UI elements."""
    # Create a 400x300 image with a white background
    image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Add UI elements
    # Button
    cv2.rectangle(image, (50, 50), (150, 100), (200, 200, 200), -1)
    cv2.rectangle(image, (50, 50), (150, 100), (100, 100, 100), 2)
    cv2.putText(image, "Button", (70, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Checkbox
    cv2.rectangle(image, (200, 50), (220, 70), (255, 255, 255), -1)
    cv2.rectangle(image, (200, 50), (220, 70), (100, 100, 100), 1)
    
    # Text input field
    cv2.rectangle(image, (50, 150), (350, 180), (255, 255, 255), -1)
    cv2.rectangle(image, (50, 150), (350, 180), (100, 100, 100), 1)
    cv2.putText(image, "Text input", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Dropdown
    cv2.rectangle(image, (50, 200), (150, 230), (240, 240, 240), -1)
    cv2.rectangle(image, (50, 200), (150, 230), (100, 100, 100), 1)
    cv2.putText(image, "▼", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def test_cursor_strategy(image, strategy, name, coords):
    """Test the cursor strategy with given coordinates."""
    # Process coordinates
    processed_coords = strategy.process_target(image, coords)
    print(f"\n{name}:")
    print(f"Original coordinates: {coords}")
    print(f"Processed coordinates: {processed_coords}")
    
    # Create visualization
    visualization = strategy.visualize_target(image, processed_coords)
    filename = f'visualization_{name.lower().replace(" ", "_")}.png'
    cv2.imwrite(filename, visualization)
    print(f"Created {filename}")
    
    return visualization

def main():
    # Create test image
    test_image = create_test_image()
    cv2.imwrite('original.png', test_image)
    print("Created original.png")
    
    # Initialize strategy with self-correction
    strategy = CursorReplayStrategy(
        dot_radius=6,
        dot_color=(0, 0, 255),  # Red in BGR
        enable_self_correction=True,
        confidence_threshold=0.7
    )
    
    # Test cases
    test_cases = [
        ("Button Click", (100, 80)),      # Near button center
        ("Checkbox", (210, 55)),          # Slightly off checkbox
        ("Text Input", (200, 165)),       # Text input field
        ("Dropdown", (140, 215)),         # Dropdown arrow
        ("Out of Bounds", (500, 500))     # Out of bounds
    ]
    
    # Run tests
    for name, coords in test_cases:
        test_cursor_strategy(test_image, strategy, name, coords)
    
    print("\nAll visualizations have been created!")

if __name__ == "__main__":
    main()
