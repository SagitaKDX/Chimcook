"""
Voice Assistant v2 - Audio Output Module
========================================

Step 8: Play synthesized audio through speakers.

Features:
- Non-blocking playback
- Interrupt support (stop current audio)
- Volume control
- Fade in/out for smooth transitions
- Queue support for sequential playback

Dependencies:
- sounddevice (uses PortAudio)
"""

from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
import threading
import time

# Import sounddevice
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("[AudioOutput] Warning: sounddevice not installed")


@dataclass
class AudioOutputConfig:
    """Configuration for audio output."""
    device: Optional[int] = None    # None = default device
    volume: float = 1.0             # 0.0 to 1.0
    latency: str = "low"            # "low", "high"
    fade_ms: int = 10               # Fade in/out duration for smooth starts/stops


class AudioOutput:
    """
    Plays audio through system speakers.
    
    Features:
    - Blocking and non-blocking playback
    - Interrupt current playback
    - Volume control with smooth fading
    - Check playback status
    
    Usage:
        # Basic playback
        output = AudioOutput()
        output.play(audio, sample_rate)  # Blocking
        
        # Non-blocking
        output.play(audio, sample_rate, blocking=False)
        # ... do other things ...
        output.wait()  # Wait for completion
        
        # Interrupt
        output.play(long_audio, sr, blocking=False)
        time.sleep(1)
        output.stop()  # Stop immediately
        
        # Check status
        if output.is_playing():
            print("Still playing...")
    """
    
    def __init__(self, config: Optional[AudioOutputConfig] = None):
        if not HAS_SOUNDDEVICE:
            raise ImportError(
                "sounddevice not installed. "
                "Run: pip install sounddevice"
            )
        
        self.config = config or AudioOutputConfig()
        
        # Playback state
        self._is_playing = False
        self._lock = threading.Lock()
        self._current_stream: Optional[sd.OutputStream] = None
        
        # Verify device
        if self.config.device is not None:
            devices = sd.query_devices()
            if self.config.device >= len(devices):
                raise ValueError(f"Device {self.config.device} not found")
        
        print(f"[AudioOutput] Initialized")
        print(f"      Device: {self.config.device or 'default'}")
        print(f"      Volume: {self.config.volume:.0%}")
    
    def play(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        blocking: bool = True,
    ) -> None:
        """
        Play audio through speakers.
        
        Args:
            audio: Audio data (int16 or float32)
            sample_rate: Sample rate in Hz
            blocking: If True, wait for playback to finish
        """
        # Stop any current playback
        self.stop()
        
        # Convert int16 to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Apply volume
        audio = audio * self.config.volume
        
        # Clip to prevent distortion
        audio = np.clip(audio, -1.0, 1.0)
        
        # Apply fade in/out for smooth playback
        audio = self._apply_fade(audio, sample_rate)
        
        # Ensure proper shape for sounddevice (samples, channels)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        
        with self._lock:
            self._is_playing = True
        
        # Play with error handling for ALSA conflicts
        try:
            sd.play(
                audio, 
                samplerate=sample_rate,
                device=self.config.device,
                latency=self.config.latency,
            )
        except sd.PortAudioError as e:
            print(f"[AudioOutput] Warning: Audio playback error: {e}")
            with self._lock:
                self._is_playing = False
            return
        
        if blocking:
            try:
                sd.wait()
            except sd.PortAudioError:
                pass  # Ignore wait errors
            with self._lock:
                self._is_playing = False
        else:
            # Start a thread to track when playback ends
            duration = len(audio) / sample_rate
            thread = threading.Thread(
                target=self._track_playback,
                args=(duration,),
                daemon=True
            )
            thread.start()
    
    def _track_playback(self, duration: float) -> None:
        """Track non-blocking playback completion."""
        time.sleep(duration)
        with self._lock:
            self._is_playing = False
    
    def _apply_fade(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply fade in/out to prevent clicks."""
        fade_samples = int(self.config.fade_ms * sample_rate / 1000)
        
        if len(audio) < fade_samples * 2:
            return audio  # Too short to fade
        
        # Create fade curves
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        
        # Apply fades
        audio = audio.copy()
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        return audio
    
    def stop(self) -> None:
        """Stop current playback immediately."""
        sd.stop()
        with self._lock:
            self._is_playing = False
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        with self._lock:
            return self._is_playing
    
    def wait(self) -> None:
        """Wait for current playback to finish."""
        sd.wait()
        with self._lock:
            self._is_playing = False
    
    def set_volume(self, volume: float) -> None:
        """
        Set playback volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.config.volume = max(0.0, min(1.0, volume))
    
    def play_sequence(
        self,
        audio_list: List[tuple],
        gap_ms: int = 100,
    ) -> None:
        """
        Play multiple audio clips in sequence.
        
        Args:
            audio_list: List of (audio, sample_rate) tuples
            gap_ms: Gap between clips in milliseconds
        """
        gap_seconds = gap_ms / 1000.0
        
        for audio, sample_rate in audio_list:
            self.play(audio, sample_rate, blocking=True)
            if gap_ms > 0:
                time.sleep(gap_seconds)
    
    @staticmethod
    def list_devices() -> List[dict]:
        """
        List available audio output devices.
        
        Returns:
            List of device info dictionaries
        """
        devices = sd.query_devices()
        output_devices = []
        
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                output_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_output_channels'],
                    'sample_rate': device['default_samplerate'],
                })
        
        return output_devices
    
    @staticmethod
    def get_default_device() -> dict:
        """Get default output device info."""
        device_id = sd.default.device[1]  # Output device
        device = sd.query_devices(device_id)
        return {
            'id': device_id,
            'name': device['name'],
            'channels': device['max_output_channels'],
            'sample_rate': device['default_samplerate'],
        }


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test AudioOutput module."""
    import sys
    
    print("=" * 60)
    print("AUDIO OUTPUT MODULE TEST")
    print("=" * 60)
    
    print(f"\nsounddevice available: {HAS_SOUNDDEVICE}")
    
    if not HAS_SOUNDDEVICE:
        print("Install with: pip install sounddevice")
        sys.exit(1)
    
    # Test 1: List devices
    print("\n[Test 1] Available output devices:")
    devices = AudioOutput.list_devices()
    for device in devices[:5]:  # Show first 5
        print(f"  [{device['id']}] {device['name']} ({device['channels']}ch, {device['sample_rate']}Hz)")
    
    default = AudioOutput.get_default_device()
    print(f"\n  Default: [{default['id']}] {default['name']}")
    
    # Test 2: Initialize
    print("\n[Test 2] Initializing AudioOutput...")
    config = AudioOutputConfig(volume=0.8)
    output = AudioOutput(config)
    
    # Test 3: Generate test tone
    print("\n[Test 3] Generating test tone...")
    sample_rate = 22050
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    tone = np.sin(2 * np.pi * frequency * t) * 0.5  # 50% amplitude
    
    # Add envelope for pleasant sound
    envelope = np.ones_like(t)
    attack = int(0.05 * sample_rate)
    release = int(0.1 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    tone *= envelope
    
    print(f"  Tone: {frequency}Hz, {duration}s")
    
    # Test 4: Blocking playback
    print("\n[Test 4] Blocking playback...")
    print("  Playing tone (you should hear a beep)...")
    output.play(tone, sample_rate, blocking=True)
    print("  Done!")
    
    # Test 5: Non-blocking playback
    print("\n[Test 5] Non-blocking playback...")
    
    # Generate longer tone
    long_duration = 2.0
    t_long = np.linspace(0, long_duration, int(sample_rate * long_duration), dtype=np.float32)
    long_tone = np.sin(2 * np.pi * 523.25 * t_long) * 0.5  # C5 note
    
    print("  Starting 2-second tone (non-blocking)...")
    output.play(long_tone, sample_rate, blocking=False)
    
    # Check status
    time.sleep(0.5)
    print(f"  Is playing: {output.is_playing()}")
    
    print("  Waiting for completion...")
    output.wait()
    print(f"  Is playing: {output.is_playing()}")
    print("  Done!")
    
    # Test 6: Interrupt
    print("\n[Test 6] Interrupt test...")
    print("  Starting tone, will interrupt after 0.5s...")
    output.play(long_tone, sample_rate, blocking=False)
    time.sleep(0.5)
    output.stop()
    print("  Stopped!")
    
    # Test 7: Volume control
    print("\n[Test 7] Volume test...")
    for vol in [0.3, 0.6, 1.0]:
        output.set_volume(vol)
        print(f"  Playing at {vol:.0%} volume...")
        output.play(tone, sample_rate, blocking=True)
        time.sleep(0.2)
    
    print("\n" + "=" * 60)
    print("AUDIO OUTPUT TEST COMPLETE")
    print("=" * 60)
