import pytest
import numpy as np
from replay_strategies.cursor_replay import CursorReplayStrategy

@pytest.fixture
def strategy():
    return CursorReplayStrategy()

@pytest.fixture
def test_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_initialization():
    strategy = CursorReplayStrategy(
        dot_radius=10,
        dot_color=(255, 0, 0),
        enable_self_correction=True
    )
    assert strategy.dot_radius == 10
    assert strategy.dot_color == (255, 0, 0)
    assert strategy.enable_self_correction is True

def test_process_target_without_correction(strategy, test_image):
    coords = (50, 50)
    processed_coords = strategy.process_target(test_image, coords)
    assert processed_coords == coords

def test_process_target_bounds_checking(strategy, test_image):
    # Test coordinates outside image bounds
    coords = (150, 150)
    processed_coords = strategy.process_target(test_image, coords)
    assert processed_coords[0] < 100
    assert processed_coords[1] < 100

def test_visualize_target(strategy, test_image):
    coords = (50, 50)
    visualization = strategy.visualize_target(test_image, coords)
    
    # Check that visualization has changed the image
    assert not np.array_equal(visualization, test_image)
    
    # Check that the dot was drawn (should have non-zero pixels)
    assert np.sum(visualization) > 0

def test_self_correction_enabled():
    strategy = CursorReplayStrategy(enable_self_correction=True)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    coords = (50, 50)
    
    processed_coords = strategy.process_target(test_image, coords)
    assert isinstance(processed_coords, tuple)
    assert len(processed_coords) == 2
    assert all(isinstance(x, (int, np.integer)) for x in processed_coords)
