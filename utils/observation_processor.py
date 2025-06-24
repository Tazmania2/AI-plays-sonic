import numpy as np
import cv2
from typing import Dict, Any, Tuple


class ObservationProcessor:
    """
    Processes raw screen observations for the RL agent.
    
    Handles image preprocessing including:
    - Resizing
    - Cropping
    - Normalization
    - Grayscale conversion
    - Frame stacking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resize_dims = tuple(config['resize'])
        self.crop_region = config.get('crop', None)
        self.normalize = config.get('normalize', True)
        self.frame_stack = config.get('frame_stack', 1)
        
        # Frame buffer for stacking
        self.frame_buffer = []
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process a raw screen image.
        
        Args:
            image: Raw screen capture as numpy array (BGR format)
            
        Returns:
            Processed observation ready for the RL agent
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop if specified
        if self.crop_region:
            x, y, w, h = self.crop_region
            image = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize
        image = cv2.resize(image, self.resize_dims, interpolation=cv2.INTER_AREA)
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.uint8)
        
        # Add channel dimension if grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def process_with_stack(self, image: np.ndarray) -> np.ndarray:
        """
        Process image and return stacked frames.
        
        Args:
            image: Raw screen capture as numpy array
            
        Returns:
            Stacked observation with temporal information
        """
        processed = self.process(image)
        
        # Add to frame buffer
        self.frame_buffer.append(processed)
        
        # Keep only the last N frames
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
        
        # If we don't have enough frames, pad with zeros
        while len(self.frame_buffer) < self.frame_stack:
            zero_frame = np.zeros_like(processed)
            self.frame_buffer.insert(0, zero_frame)
        
        # Stack frames along channel dimension
        if self.frame_stack == 1:
            return self.frame_buffer[0]
        else:
            return np.concatenate(self.frame_buffer, axis=-1)
    
    def reset(self):
        """Reset the frame buffer."""
        self.frame_buffer = []
    
    def get_observation_shape(self) -> Tuple[int, int, int]:
        """Get the shape of processed observations."""
        height, width = self.resize_dims
        channels = 1  # Grayscale
        if self.frame_stack > 1:
            channels *= self.frame_stack
        return (height, width, channels)


class AdvancedObservationProcessor(ObservationProcessor):
    """
    Advanced observation processor with additional features.
    
    Features:
    - Background subtraction
    - Motion detection
    - Object detection hints
    - Color-based feature extraction
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Background model for motion detection
        self.background_model = None
        self.background_alpha = config.get('background_alpha', 0.1)
        
        # Motion detection
        self.motion_threshold = config.get('motion_threshold', 30)
        self.motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Color-based features
        self.color_features = config.get('color_features', False)
        self.color_ranges = {
            'sonic_blue': ([100, 150, 150], [130, 255, 255]),  # HSV ranges
            'rings_yellow': ([20, 100, 100], [30, 255, 255]),
            'enemies_red': ([0, 100, 100], [10, 255, 255])
        }
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Enhanced processing with motion and color features."""
        # Basic processing
        processed = super().process(image)
        
        # Convert back to BGR for color processing
        if self.color_features:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            color_features = self._extract_color_features(bgr_image)
            
            # Combine with grayscale
            if len(processed.shape) == 3 and processed.shape[2] == 1:
                processed = np.concatenate([processed, color_features], axis=2)
        
        # Motion detection
        motion_mask = self._detect_motion(image)
        if motion_mask is not None:
            motion_features = np.expand_dims(motion_mask, axis=-1)
            if len(processed.shape) == 3 and processed.shape[2] == 1:
                processed = np.concatenate([processed, motion_features], axis=2)
        
        return processed
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color-based features for important game objects."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_masks = []
        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Resize mask to match observation size
            mask = cv2.resize(mask, self.resize_dims, interpolation=cv2.INTER_AREA)
            
            # Normalize
            mask = mask.astype(np.float32) / 255.0
            color_masks.append(mask)
        
        # Combine all color masks
        if color_masks:
            combined = np.stack(color_masks, axis=-1)
            return combined
        else:
            return np.zeros((*self.resize_dims, len(self.color_ranges)))
    
    def _detect_motion(self, image: np.ndarray) -> np.ndarray:
        """Detect motion in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, self.resize_dims, interpolation=cv2.INTER_AREA)
        
        # Initialize background model
        if self.background_model is None:
            self.background_model = gray.astype(np.float32)
            return np.zeros_like(gray)
        
        # Update background model
        cv2.accumulateWeighted(gray, self.background_model, self.background_alpha)
        
        # Calculate difference
        diff = cv2.absdiff(gray, self.background_model.astype(np.uint8))
        
        # Threshold
        _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.motion_kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.motion_kernel)
        
        # Normalize
        motion_mask = motion_mask.astype(np.float32) / 255.0
        
        return motion_mask
    
    def reset(self):
        """Reset processor state."""
        super().reset()
        self.background_model = None


class SonicSpecificProcessor(AdvancedObservationProcessor):
    """
    Sonic-specific observation processor.
    
    Optimized for Sonic games with:
    - Sonic character detection
    - Ring detection
    - Enemy detection
    - Platform detection
    - Speed detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Sonic-specific color ranges (HSV)
        self.sonic_colors = {
            'sonic_body': ([100, 150, 150], [130, 255, 255]),  # Blue
            'sonic_shoes': ([0, 100, 100], [10, 255, 255]),    # Red
            'rings': ([20, 100, 100], [30, 255, 255]),         # Yellow
            'enemies': ([0, 100, 100], [10, 255, 255]),        # Red
            'platforms': ([0, 0, 100], [180, 255, 255]),       # Gray/White
            'spikes': ([0, 0, 0], [180, 255, 50])              # Dark
        }
        
        # Object tracking
        self.sonic_position = None
        self.ring_positions = []
        self.enemy_positions = []
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """Sonic-specific processing."""
        # Basic processing
        processed = super().process(image)
        
        # Extract Sonic-specific features
        sonic_features = self._extract_sonic_features(image)
        
        # Combine features
        if len(processed.shape) == 3:
            processed = np.concatenate([processed, sonic_features], axis=2)
        
        return processed
    
    def _extract_sonic_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Sonic-specific game features."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        features = []
        
        # Sonic character detection
        sonic_mask = self._detect_sonic(hsv)
        features.append(sonic_mask)
        
        # Ring detection
        ring_mask = self._detect_rings(hsv)
        features.append(ring_mask)
        
        # Enemy detection
        enemy_mask = self._detect_enemies(hsv)
        features.append(enemy_mask)
        
        # Platform detection
        platform_mask = self._detect_platforms(hsv)
        features.append(platform_mask)
        
        # Combine all features
        combined = np.stack(features, axis=-1)
        return combined
    
    def _detect_sonic(self, hsv: np.ndarray) -> np.ndarray:
        """Detect Sonic character in the image."""
        # Blue body
        lower_blue = np.array(self.sonic_colors['sonic_body'][0])
        upper_blue = np.array(self.sonic_colors['sonic_body'][1])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Red shoes
        lower_red = np.array(self.sonic_colors['sonic_shoes'][0])
        upper_red = np.array(self.sonic_colors['sonic_shoes'][1])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Combine
        sonic_mask = cv2.bitwise_or(blue_mask, red_mask)
        
        # Resize and normalize
        sonic_mask = cv2.resize(sonic_mask, self.resize_dims, interpolation=cv2.INTER_AREA)
        sonic_mask = sonic_mask.astype(np.float32) / 255.0
        
        return sonic_mask
    
    def _detect_rings(self, hsv: np.ndarray) -> np.ndarray:
        """Detect rings in the image."""
        lower_yellow = np.array(self.sonic_colors['rings'][0])
        upper_yellow = np.array(self.sonic_colors['rings'][1])
        ring_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Resize and normalize
        ring_mask = cv2.resize(ring_mask, self.resize_dims, interpolation=cv2.INTER_AREA)
        ring_mask = ring_mask.astype(np.float32) / 255.0
        
        return ring_mask
    
    def _detect_enemies(self, hsv: np.ndarray) -> np.ndarray:
        """Detect enemies in the image."""
        lower_red = np.array(self.sonic_colors['enemies'][0])
        upper_red = np.array(self.sonic_colors['enemies'][1])
        enemy_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Resize and normalize
        enemy_mask = cv2.resize(enemy_mask, self.resize_dims, interpolation=cv2.INTER_AREA)
        enemy_mask = enemy_mask.astype(np.float32) / 255.0
        
        return enemy_mask
    
    def _detect_platforms(self, hsv: np.ndarray) -> np.ndarray:
        """Detect platforms and solid surfaces."""
        lower_gray = np.array(self.sonic_colors['platforms'][0])
        upper_gray = np.array(self.sonic_colors['platforms'][1])
        platform_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Resize and normalize
        platform_mask = cv2.resize(platform_mask, self.resize_dims, interpolation=cv2.INTER_AREA)
        platform_mask = platform_mask.astype(np.float32) / 255.0
        
        return platform_mask 