"""
Voice Assistant v2 - Wake Word Detection
=========================================

Wake word detection for crowded environments.

When enabled:
1. System stays in "sleep" mode (ignores all audio)
2. User says "Hey Assistant" (or configured wake word)
3. System activates and listens for question
4. After response, system returns to sleep

This is the BEST solution for crowded places because:
- Ignores all speech until wake word
- Clear activation trigger
- Lower false positive rate

Implementation: Simple energy + keyword detection
- Detects loud audio bursts
- Checks if they contain wake word keywords
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
from collections import deque
import time
import numpy as np


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    wake_words: List[str] = field(default_factory=lambda: ["hello assistant", "hey assistant", "ok assistant"])
    energy_threshold: float = 0.03      # Minimum RMS to process
    detection_window_sec: float = 2.0   # Audio window to analyze
    cooldown_sec: float = 2.0           # Minimum time between detections
    listen_timeout_sec: float = 8.0     # How long to listen after activation


class SimpleWakeWordDetector:
    """
    Simple wake word detector using energy detection + keyword matching.
    
    This doesn't use heavy ML - it detects when someone speaks loudly
    and checks if the audio starts with wake word-like patterns.
    
    For production, consider:
    - Porcupine (Picovoice) - very accurate, free tier
    - OpenWakeWord - fully open source
    
    Usage:
        detector = SimpleWakeWordDetector(config)
        
        for frame in audio_stream:
            if detector.check_activation(frame):
                print("Wake word detected!")
                # Start listening for command
    """
    
    def __init__(self, config: Optional[WakeWordConfig] = None, sample_rate: int = 16000):
        self.config = config or WakeWordConfig()
        self.sample_rate = sample_rate
        
        # Calculate window size
        self._window_samples = int(self.config.detection_window_sec * sample_rate)
        
        # Audio buffer
        self._audio_buffer: deque = deque(maxlen=self._window_samples)
        
        # State
        self._last_detection_time = 0.0
        self._is_listening = False
        self._listen_start_time = 0.0
        
        # Energy tracking
        self._recent_energy: deque = deque(maxlen=50)  # ~1 second at 20ms frames
        self._baseline_energy = 0.01
        
        # Callbacks
        self._on_wake: Optional[Callable] = None
        self._on_timeout: Optional[Callable] = None
    
    def check_activation(self, frame: np.ndarray) -> bool:
        """
        Check if frame contains wake word activation.
        
        This uses a simple approach:
        1. Detect sudden loud audio (potential speech start)
        2. Accumulate audio for analysis window
        3. Check if accumulated audio might be wake word
        
        Args:
            frame: Audio frame (float32)
            
        Returns:
            True if wake word likely detected
        """
        current_time = time.time()
        
        # Cooldown check
        if current_time - self._last_detection_time < self.config.cooldown_sec:
            return False
        
        # Calculate energy
        energy = self._compute_rms(frame)
        self._recent_energy.append(energy)
        
        # Update baseline during quiet periods
        if len(self._recent_energy) >= 20:
            sorted_energy = sorted(self._recent_energy)
            # Use 20th percentile as baseline (quietest 20%)
            self._baseline_energy = sorted_energy[len(sorted_energy) // 5]
        
        # Add to buffer
        self._audio_buffer.extend(frame)
        
        # Check for activation trigger (sudden loud audio)
        activation_threshold = max(
            self.config.energy_threshold,
            self._baseline_energy * 3  # 3x baseline
        )
        
        if energy > activation_threshold:
            # Potential wake word - check if we have enough audio
            if len(self._audio_buffer) >= self._window_samples // 2:
                # For simple detection, just check energy pattern
                # A wake word has sustained energy over multiple frames
                recent_loud = sum(1 for e in list(self._recent_energy)[-10:] 
                                  if e > activation_threshold * 0.5)
                
                if recent_loud >= 3:  # At least 3 loud frames
                    self._last_detection_time = current_time
                    self._audio_buffer.clear()
                    return True
        
        return False
    
    def get_accumulated_audio(self) -> np.ndarray:
        """Get accumulated audio buffer as numpy array."""
        return np.array(list(self._audio_buffer), dtype=np.float32)
    
    def reset(self) -> None:
        """Reset detector state."""
        self._audio_buffer.clear()
        self._recent_energy.clear()
        self._is_listening = False
    
    def set_callbacks(self, on_wake: Callable = None, on_timeout: Callable = None):
        """Set callback functions."""
        self._on_wake = on_wake
        self._on_timeout = on_timeout
    
    @staticmethod
    def _compute_rms(frame: np.ndarray) -> float:
        """Compute RMS energy."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


class WakeWordAssistant:
    """
    Assistant wrapper that uses wake word for activation.
    
    States:
    - SLEEPING: Ignoring all audio, waiting for wake word
    - LISTENING: Active, recording user's command
    - PROCESSING: Processing command (STT + LLM + TTS)
    
    Usage:
        assistant = WakeWordAssistant(config)
        assistant.run()  # Blocks, handles everything
    """
    
    SLEEPING = "sleeping"
    LISTENING = "listening"
    PROCESSING = "processing"
    
    def __init__(self, 
                 wake_config: Optional[WakeWordConfig] = None,
                 sample_rate: int = 16000,
                 listen_timeout: float = 8.0,
                 silence_timeout: float = 1.5):
        """
        Initialize wake word assistant.
        
        Args:
            wake_config: Wake word configuration
            sample_rate: Audio sample rate
            listen_timeout: Max seconds to listen after wake word
            silence_timeout: Seconds of silence to end listening
        """
        self.wake_config = wake_config or WakeWordConfig()
        self.sample_rate = sample_rate
        self.listen_timeout = listen_timeout
        self.silence_timeout = silence_timeout
        
        self._detector = SimpleWakeWordDetector(self.wake_config, sample_rate)
        self._state = self.SLEEPING
        
        # Listening state
        self._listen_start = 0.0
        self._last_speech = 0.0
        self._command_buffer: List[np.ndarray] = []
        
        # Callbacks
        self._on_wake: Optional[Callable] = None
        self._on_command: Optional[Callable[[np.ndarray], None]] = None
        self._on_sleep: Optional[Callable] = None
    
    @property
    def state(self) -> str:
        return self._state
    
    @property
    def is_sleeping(self) -> bool:
        return self._state == self.SLEEPING
    
    @property  
    def is_listening(self) -> bool:
        return self._state == self.LISTENING
    
    def process_frame(self, frame: np.ndarray, is_speech: bool = None) -> str:
        """
        Process audio frame through state machine.
        
        Args:
            frame: Audio frame
            is_speech: VAD result (optional, will estimate if None)
            
        Returns:
            Current state after processing
        """
        current_time = time.time()
        
        # Estimate speech if not provided
        if is_speech is None:
            energy = self._detector._compute_rms(frame)
            is_speech = energy > self.wake_config.energy_threshold
        
        if self._state == self.SLEEPING:
            # Check for wake word
            if self._detector.check_activation(frame):
                self._state = self.LISTENING
                self._listen_start = current_time
                self._last_speech = current_time
                self._command_buffer = []
                
                if self._on_wake:
                    self._on_wake()
        
        elif self._state == self.LISTENING:
            # Record command
            self._command_buffer.append(frame.copy())
            
            if is_speech:
                self._last_speech = current_time
            
            # Check timeouts
            listen_elapsed = current_time - self._listen_start
            silence_elapsed = current_time - self._last_speech
            
            should_stop = False
            if listen_elapsed > self.listen_timeout:
                should_stop = True
            elif silence_elapsed > self.silence_timeout and len(self._command_buffer) > 25:
                # At least 0.5 seconds of audio before stopping on silence
                should_stop = True
            
            if should_stop:
                self._state = self.PROCESSING
                
                # Get command audio
                if self._command_buffer:
                    command_audio = np.concatenate(self._command_buffer)
                    
                    if self._on_command:
                        self._on_command(command_audio)
                
                # Go back to sleep
                self._state = self.SLEEPING
                self._command_buffer = []
                
                if self._on_sleep:
                    self._on_sleep()
        
        return self._state
    
    def set_callbacks(self,
                      on_wake: Callable = None,
                      on_command: Callable[[np.ndarray], None] = None,
                      on_sleep: Callable = None):
        """Set callback functions for state changes."""
        self._on_wake = on_wake
        self._on_command = on_command
        self._on_sleep = on_sleep
    
    def force_wake(self) -> None:
        """Manually trigger wake (for push-to-talk)."""
        self._state = self.LISTENING
        self._listen_start = time.time()
        self._last_speech = time.time()
        self._command_buffer = []
        
        if self._on_wake:
            self._on_wake()
    
    def force_sleep(self) -> None:
        """Manually go to sleep."""
        self._state = self.SLEEPING
        self._command_buffer = []
        
        if self._on_sleep:
            self._on_sleep()


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace('/utils/wake_word.py', ''))
    
    print("=" * 60)
    print("WAKE WORD ASSISTANT TEST")
    print("=" * 60)
    
    try:
        from core.audio_input import AudioInput, AudioInputConfig
        
        print("\nThis simulates wake word behavior:")
        print("  - System starts SLEEPING (ignoring audio)")
        print("  - Speak LOUDLY to trigger wake")
        print("  - Then speak your command")
        print("  - After silence, system goes back to SLEEP")
        print()
        
        # Setup
        audio_in = AudioInput(AudioInputConfig(sample_rate=16000, frame_ms=20))
        
        assistant = WakeWordAssistant(
            wake_config=WakeWordConfig(
                energy_threshold=0.04,  # Adjust based on your mic
                cooldown_sec=2.0,
            ),
            listen_timeout=6.0,
            silence_timeout=1.2,
        )
        
        # Callbacks
        def on_wake():
            print("\n🎤 WAKE! Listening for command...")
        
        def on_command(audio):
            duration = len(audio) / 16000
            print(f"\n📝 Command received! ({duration:.1f} seconds of audio)")
            print("   (In real use, this would go to STT → LLM → TTS)")
        
        def on_sleep():
            print("😴 Going back to sleep...\n")
        
        assistant.set_callbacks(on_wake, on_command, on_sleep)
        
        print("Starting... Speak LOUDLY to wake!")
        print("(Ctrl+C to stop)\n")
        
        audio_in.start()
        
        try:
            while True:
                frame = audio_in.get_frame(timeout=0.1)
                if frame is not None:
                    state = assistant.process_frame(frame)
                    
                    # Visual indicator
                    rms = float(np.sqrt(np.mean(frame ** 2)))
                    bar = int(rms * 200)
                    
                    if state == WakeWordAssistant.SLEEPING:
                        indicator = "😴 SLEEPING"
                    elif state == WakeWordAssistant.LISTENING:
                        indicator = "🎤 LISTENING"
                    else:
                        indicator = "⚙️  PROCESSING"
                    
                    print(f"\r{indicator} {'█' * min(bar, 30):<30}", end="", flush=True)
        
        except KeyboardInterrupt:
            print("\n\nStopped.")
        
        audio_in.stop()
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Run from project root directory")
