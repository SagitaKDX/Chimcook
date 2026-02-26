"""
Face Detection Module for Voice Assistant
==========================================

CPU-optimized face detection using HOG (Histogram of Oriented Gradients).
Perfect for mini PC without GPU.

Features:
- HOG-based detection (fast on CPU)
- Easy "training": just drop photos in known_faces/ folder
- Face count detection for single-user verification
- Talking detection using lip/jaw movement analysis

Requirements:
    sudo apt-get install cmake build-essential
    pip install face_recognition opencv-python

Usage:
    from core.face_detector import FaceDetector, FaceDetectorConfig
    
    detector = FaceDetector(FaceDetectorConfig(
        known_faces_dir="known_faces",
    ))
    
    # In loop:
    result = detector.process_frame(frame)
    if result.face_count == 1:
        print(f"Hello, {result.recognized_name}!")
        if result.is_talking:
            # Person is speaking
            pass
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
import time


@dataclass
class FaceDetectorConfig:
    """Configuration for face detection."""
    # Detection
    model: str = "hog"              # "hog" (CPU fast) or "cnn" (GPU accurate)
    detection_scale: float = 0.5   # Downscale for faster detection (0.5 = half size)
    detection_interval_ms: int = 500  # Run detection every N ms (save CPU)
    
    # Recognition
    known_faces_dir: str = "known_faces"  # Directory with reference photos
    recognition_tolerance: float = 0.6    # Lower = stricter (0.4-0.6 typical)
    
    # Talking detection
    enable_talking_detection: bool = True
    mouth_movement_threshold: float = 0.15  # Movement threshold for talking
    talking_frames_window: int = 5          # Frames to average for smoothing
    
    # Camera
    camera_index: int = 0           # Camera device index
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15            # Lower FPS = less CPU
    
    # Image preprocessing for better detection
    enhance_brightness: bool = True      # Enhance brightness/contrast
    brightness_alpha: float = 1.2         # Brightness multiplier (1.0 = no change, >1.0 = brighter)
    brightness_beta: int = 20            # Brightness offset (-100 to 100)
    enhance_contrast: bool = True        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe_clip_limit: float = 2.0        # CLAHE clip limit (higher = more contrast)
    clahe_tile_size: int = 8             # CLAHE tile grid size
    denoise: bool = True                 # Apply denoising filter
    sharpen: bool = True                 # Apply sharpening filter


@dataclass
class FaceResult:
    """Result from face detection."""
    face_count: int = 0
    recognized_name: Optional[str] = None  # Name if recognized, None if unknown
    confidence: float = 0.0
    is_talking: bool = False
    face_location: Optional[Tuple[int, int, int, int]] = None  # top, right, bottom, left
    debug_frame: Optional[np.ndarray] = None  # Frame with annotations (if debug enabled)


class FaceDetector:
    """
    CPU-optimized face detector with recognition.
    
    "Training" is easy - just add photos to the known_faces/ directory:
        known_faces/
            john.jpg
            jane.png
            boss.jpg
    
    The filename (without extension) becomes the person's name.
    """
    
    def __init__(self, config: FaceDetectorConfig):
        self.config = config
        self._cap = None
        self._known_encodings: List[np.ndarray] = []
        self._known_names: List[str] = []
        self._last_detection_time = 0
        self._cached_result: Optional[FaceResult] = None
        
        # Talking detection state
        self._mouth_positions: List[float] = []
        self._prev_mouth_height = 0.0
        
        # Import face_recognition (lazy load for faster startup)
        try:
            import face_recognition
            self._face_recognition = face_recognition
            print("      Face recognition: HOG detector (CPU optimized)")
        except ImportError:
            raise ImportError(
                "face_recognition not installed. Run:\n"
                "  sudo apt-get install cmake build-essential\n"
                "  pip install face_recognition opencv-python"
            )
        
        try:
            import cv2
            import os
            # Suppress OpenCV warnings
            os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
            # Try to set log level (method varies by OpenCV version)
            try:
                cv2.setLogLevel(0)  # 0 = SILENT, 1 = FATAL, 2 = ERROR, 3 = WARN, 4 = INFO
            except (AttributeError, TypeError):
                # Older OpenCV versions don't have setLogLevel
                pass
            self._cv2 = cv2
        except ImportError:
            raise ImportError("opencv-python not installed. Run: pip install opencv-python")
        
        # Load known faces
        self._load_known_faces()
    
    def _load_known_faces(self) -> None:
        """Load face encodings from known_faces directory."""
        faces_dir = Path(self.config.known_faces_dir)
        
        if not faces_dir.exists():
            faces_dir.mkdir(parents=True, exist_ok=True)
            print(f"      Created {faces_dir}/ - add photos to recognize people")
            return
        
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        count = 0
        for img_path in faces_dir.iterdir():
            if img_path.suffix.lower() not in extensions:
                continue
            
            try:
                # Load and encode face
                image = self._face_recognition.load_image_file(str(img_path))
                encodings = self._face_recognition.face_encodings(image)
                
                if encodings:
                    self._known_encodings.append(encodings[0])
                    # Use filename without extension as name
                    name = img_path.stem.replace('_', ' ').title()
                    self._known_names.append(name)
                    count += 1
                    print(f"        Loaded: {name}")
                else:
                    print(f"        Warning: No face found in {img_path.name}")
            
            except Exception as e:
                print(f"        Error loading {img_path.name}: {e}")
        
        if count > 0:
            print(f"      Loaded {count} known face(s)")
        else:
            print(f"      No known faces loaded (add photos to {faces_dir}/)")
    
    def start(self) -> bool:
        """Start video capture. Tries configured index first, then 0, 1, 2. Returns True if successful."""
        if self._cap is not None:
            return True

        # Check if camera devices exist
        import glob
        import stat
        import os
        video_devices = glob.glob('/dev/video*')
        if not video_devices:
            print(f"      Error: No camera devices found (/dev/video*)")
            print(f"      Tip: Check if camera is connected and drivers are installed")
            return False
        
        # Check permissions on first camera device
        try:
            first_device = video_devices[0]
            device_stat = os.stat(first_device)
            device_mode = stat.filemode(device_stat.st_mode)
            user_groups = os.getgroups()
            
            # Check if user can access the device
            # Devices are typically owned by root:video with 0660 permissions
            # User needs to be in 'video' group
            try:
                import grp
                video_gid = grp.getgrnam('video').gr_gid
                has_video_group = video_gid in user_groups
            except (KeyError, ImportError):
                has_video_group = False
            
            if not has_video_group and not os.access(first_device, os.R_OK | os.W_OK):
                print(f"      Warning: Camera permissions issue detected")
                print(f"      Device: {first_device} ({device_mode})")
                print(f"      Fix: Add user to 'video' group:")
                print(f"         sudo usermod -a -G video $USER")
                print(f"         (then logout and login again)")
        except Exception:
            pass  # Don't fail if permission check fails

        indices_to_try = list(dict.fromkeys([
            self.config.camera_index,
            0,
            1,
            2,
        ]))

        # Try different backends in order of preference
        backends_to_try = [
            None,  # Auto-detect (often most reliable)
            self._cv2.CAP_V4L2,  # V4L2 for Linux
            self._cv2.CAP_ANY,   # Any available backend
        ]

        # Try each index with different backends
        for idx in indices_to_try:
            for backend in backends_to_try:
                try:
                    # Release any previous attempt
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                    
                    # Small delay (helps if camera is locked)
                    time.sleep(0.2)
                    
                    # Try to open camera with this backend
                    if backend is None:
                        self._cap = self._cv2.VideoCapture(idx)
                    else:
                        self._cap = self._cv2.VideoCapture(idx, backend)
                    
                    if self._cap.isOpened():
                        # Set properties before testing read
                        self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
                        self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
                        self._cap.set(self._cv2.CAP_PROP_FPS, self.config.camera_fps)
                        self._cap.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Test read to ensure camera actually works
                        # Try multiple reads to flush any stale frames
                        ret = False
                        frame = None
                        for _ in range(5):  # More attempts to flush
                            ret, frame = self._cap.read()
                            if ret and frame is not None and frame.size > 0:
                                break
                            time.sleep(0.1)
                        
                        if ret and frame is not None and frame.size > 0:
                            backend_name = "auto" if backend is None else f"backend_{backend}"
                            print(f"      Camera started: {self.config.camera_width}x{self.config.camera_height}@{self.config.camera_fps}fps (index {idx}, {backend_name})")
                            return True
                    
                    # If we got here, camera didn't work with this backend
                    if self._cap is not None:
                        self._cap.release()
                        self._cap = None
                        
                except Exception as e:
                    # Suppress errors, try next backend
                    if self._cap is not None:
                        try:
                            self._cap.release()
                        except:
                            pass
                        self._cap = None
                    continue

        # If all attempts failed
        print(f"      Error: Could not open camera (tried indices: {indices_to_try})")
        print(f"      Available devices: {', '.join(video_devices)}")
        
        # Check if it's a permissions issue
        try:
            import grp
            user_groups = os.getgroups()
            video_gid = grp.getgrnam('video').gr_gid
            has_video_group = video_gid in user_groups
            
            if not has_video_group:
                print(f"      ⚠️  PERMISSIONS ISSUE DETECTED:")
                print(f"         Your user is not in the 'video' group")
                print(f"         Run this command to fix:")
                print(f"            sudo usermod -a -G video $USER")
                print(f"         Then logout and login again (or restart)")
        except Exception:
            pass
        
        print(f"      Troubleshooting:")
        print(f"        1. Add user to video group: sudo usermod -a -G video $USER")
        print(f"        2. Close other applications using the camera (Firefox, Zoom, etc.)")
        print(f"        3. Check permissions: ls -l /dev/video*")
        print(f"        4. Test camera: v4l2-ctl --list-devices")
        self._cap = None
        return False
    
    def stop(self) -> None:
        """Stop video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera."""
        if self._cap is None:
            if not self.start():
                return None
        
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        return frame
    
    def process_frame(self, frame: Optional[np.ndarray] = None, debug: bool = False) -> FaceResult:
        """
        Process a frame for face detection/recognition.
        
        Args:
            frame: BGR image from camera. If None, captures from camera.
            debug: If True, includes annotated frame in result.
        
        Returns:
            FaceResult with detection info.
        """
        # Capture frame if not provided
        if frame is None:
            frame = self.get_frame()
            if frame is None:
                return FaceResult()
        
        current_time = time.time() * 1000  # ms
        
        # Rate limiting: return cached result if too soon
        time_since_last = current_time - self._last_detection_time
        if time_since_last < self.config.detection_interval_ms and self._cached_result is not None:
            return self._cached_result
        
        self._last_detection_time = current_time
        
        # Preprocess frame for better detection
        processed_frame = self._preprocess_frame(frame)
        
        # Try at configured scale first, then full resolution if no face found
        scale = self.config.detection_scale
        small_frame = self._cv2.resize(processed_frame, (0, 0), fx=scale, fy=scale)
        rgb_frame = self._cv2.cvtColor(small_frame, self._cv2.COLOR_BGR2RGB)
        
        face_locations = self._face_recognition.face_locations(
            rgb_frame, model=self.config.model
        )
        
        # If no face at downscaled size, retry at full resolution (catches smaller faces)
        if len(face_locations) == 0 and scale < 1.0:
            full_rgb = self._cv2.cvtColor(processed_frame, self._cv2.COLOR_BGR2RGB)
            face_locations = self._face_recognition.face_locations(
                full_rgb, model=self.config.model
            )
            scale = 1.0
            rgb_frame = full_rgb
        
        result = FaceResult(face_count=len(face_locations))
        
        if len(face_locations) == 0:
            self._cached_result = result
            return result
        
        # Scale locations back to original size
        face_locations_scaled = [
            (int(top/scale), int(right/scale), int(bottom/scale), int(left/scale))
            for top, right, bottom, left in face_locations
        ]
        
        # Use the first (largest/closest) face
        result.face_location = face_locations_scaled[0]
        
        # Recognition (if we have known faces)
        if self._known_encodings:
            # Get encoding for detected face
            face_encodings = self._face_recognition.face_encodings(
                rgb_frame, 
                face_locations
            )
            
            if face_encodings:
                # Compare with known faces
                encoding = face_encodings[0]
                distances = self._face_recognition.face_distance(
                    self._known_encodings, 
                    encoding
                )
                
                if len(distances) > 0:
                    best_match_idx = np.argmin(distances)
                    best_distance = distances[best_match_idx]
                    
                    if best_distance < self.config.recognition_tolerance:
                        result.recognized_name = self._known_names[best_match_idx]
                        result.confidence = 1.0 - best_distance
        
        # Talking detection using face landmarks
        if self.config.enable_talking_detection and len(face_locations) > 0:
            result.is_talking = self._detect_talking(rgb_frame, face_locations[0])
        
        # Debug visualization (use preprocessed frame for better visibility)
        if debug:
            result.debug_frame = self._draw_debug(processed_frame.copy(), result)
        
        self._cached_result = result
        return result
    
    def _detect_talking(self, rgb_frame: np.ndarray, face_location: Tuple) -> bool:
        """
        Detect if person is talking by analyzing mouth movement.
        
        Uses lip landmarks to track mouth opening/closing.
        """
        # Get facial landmarks
        landmarks_list = self._face_recognition.face_landmarks(rgb_frame, [face_location])
        
        if not landmarks_list:
            return False
        
        landmarks = landmarks_list[0]
        
        # Get mouth landmarks
        top_lip = landmarks.get('top_lip', [])
        bottom_lip = landmarks.get('bottom_lip', [])
        
        if not top_lip or not bottom_lip:
            return False
        
        # Calculate mouth height (opening)
        # Top lip bottom points vs bottom lip top points
        top_lip_bottom = np.mean([p[1] for p in top_lip[6:]], axis=0) if len(top_lip) > 6 else top_lip[-1][1]
        bottom_lip_top = np.mean([p[1] for p in bottom_lip[:3]], axis=0) if len(bottom_lip) > 3 else bottom_lip[0][1]
        
        mouth_height = abs(bottom_lip_top - top_lip_bottom)
        
        # Normalize by face height
        top, right, bottom, left = face_location
        face_height = bottom - top
        if face_height > 0:
            mouth_height = mouth_height / face_height
        
        # Track movement over time
        if self._prev_mouth_height > 0:
            movement = abs(mouth_height - self._prev_mouth_height)
            self._mouth_positions.append(movement)
            
            # Keep only recent frames
            if len(self._mouth_positions) > self.config.talking_frames_window:
                self._mouth_positions.pop(0)
        
        self._prev_mouth_height = mouth_height
        
        # Average movement determines talking
        if len(self._mouth_positions) >= 3:
            avg_movement = np.mean(self._mouth_positions)
            return avg_movement > self.config.mouth_movement_threshold
        
        return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame to enhance brightness, contrast, and quality for better face detection.
        
        Applies:
        - Brightness/contrast adjustment
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Denoising
        - Sharpening
        """
        cv2 = self._cv2
        processed = frame.copy()
        
        # 1. Brightness and contrast adjustment
        if self.config.enhance_brightness:
            processed = cv2.convertScaleAbs(
                processed,
                alpha=self.config.brightness_alpha,  # Contrast control (1.0-3.0)
                beta=self.config.brightness_beta      # Brightness control (0-100)
            )
        
        # 2. CLAHE for adaptive contrast enhancement (better than global histogram equalization)
        if self.config.enhance_contrast:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size)
            )
            l_channel = clahe.apply(l_channel)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l_channel, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. Denoising (removes noise while preserving edges)
        if self.config.denoise:
            processed = cv2.fastNlMeansDenoisingColored(
                processed,
                None,
                h=10,        # Filter strength (higher = more denoising, slower)
                hColor=10,   # Color component filter strength
                templateWindowSize=7,
                searchWindowSize=21
            )
        
        # 4. Sharpening filter (enhances edges for better detection)
        if self.config.sharpen:
            # Create sharpening kernel
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            processed = cv2.filter2D(processed, -1, kernel)
        
        return processed
    
    def _draw_debug(self, frame: np.ndarray, result: FaceResult) -> np.ndarray:
        """Draw debug annotations on frame."""
        cv2 = self._cv2
        
        if result.face_location:
            top, right, bottom, left = result.face_location
            
            # Color based on recognition
            if result.recognized_name:
                color = (0, 255, 0)  # Green for recognized
                label = f"{result.recognized_name} ({result.confidence:.0%})"
            else:
                color = (0, 165, 255)  # Orange for unknown
                label = "Unknown"
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            cv2.putText(frame, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Talking indicator
            if result.is_talking:
                cv2.putText(frame, "TALKING", (left, bottom + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Face count
        cv2.putText(frame, f"Faces: {result.face_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop()


# Simple test
if __name__ == "__main__":
    import cv2
    
    print("Face Detection Test")
    print("=" * 40)
    
    config = FaceDetectorConfig()
    detector = FaceDetector(config)
    
    print("\nStarting camera... Press 'q' to quit")
    
    while True:
        frame = detector.get_frame()
        if frame is None:
            print("Failed to get frame")
            break
        
        result = detector.process_frame(frame, debug=True)
        
        # Print status
        status = f"Faces: {result.face_count}"
        if result.recognized_name:
            status += f" | {result.recognized_name}"
        if result.is_talking:
            status += " | TALKING"
        print(f"\r{status}    ", end="", flush=True)
        
        # Show frame
        if result.debug_frame is not None:
            cv2.imshow("Face Detection", result.debug_frame)
        else:
            cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    detector.stop()
    cv2.destroyAllWindows()
    print("\nDone!")
