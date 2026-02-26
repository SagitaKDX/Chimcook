"""
Voice Assistant v2 - Audio Input Module
=======================================

Captures audio from microphone in fixed-size frames.

Features:
- Uses sounddevice (with pyaudio fallback)
- Thread-safe frame queue
- Frame alignment for consistent output size
- Graceful error handling
"""

from dataclasses import dataclass
from typing import Optional, Iterator, List, Dict, Tuple
import numpy as np
import queue
import threading
import time
import os


@dataclass
class AudioInputConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20
    device: Optional[int] = None
    mic_gain: float = 1.2  # Microphone boost (1.0 = none, 1.1 = 20% boost)
    backend: str = "sounddevice"  # "sounddevice" or "pyaudio"
    queue_max_size: int = 100     # Max frames to buffer


class AudioInput:
    """
    Captures audio from microphone in fixed-size frames.
    
    Features:
    - Auto-selects best input device
    - Handles buffer underruns gracefully
    - Thread-safe frame queue
    - Frame alignment for consistent sizes
    
    Usage:
        config = AudioInputConfig(sample_rate=16000, frame_ms=20)
        audio = AudioInput(config)
        audio.start()
        
        for frame in audio.frames():
            # frame: np.ndarray float32, shape (320,) for 20ms at 16kHz
            process(frame)
        
        audio.stop()
    """

    def __init__(self, config: Optional[AudioInputConfig] = None):
        self.config = config or AudioInputConfig()
        self.frame_samples = int(self.config.sample_rate * self.config.frame_ms / 1000)
        
        self._running = False
        self._stream = None
        self._queue: queue.Queue = queue.Queue(maxsize=self.config.queue_max_size)
        
        # Buffer for frame alignment (mic may deliver variable chunk sizes)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        
        # Status tracking
        self._overflow_count = 0
        self._underflow_count = 0
        
        # Validate configuration
        if self.config.frame_ms not in [10, 20, 30]:
            print(f"Warning: frame_ms={self.config.frame_ms} may not be optimal for VAD (use 10, 20, or 30)")
        
        print(f"AudioInput initialized:")
        print(f"  Sample rate: {self.config.sample_rate} Hz")
        print(f"  Frame size: {self.config.frame_ms} ms ({self.frame_samples} samples)")
        print(f"  Backend: {self.config.backend}")
        if self.config.device is not None:
            print(f"  Device: index {self.config.device}")
        if self.config.mic_gain != 1.0:
            pct = int(round((self.config.mic_gain - 1.0) * 100))
            print(f"  Mic gain: +{pct}% boost")
        # Check mic permissions (informational only; start() will enforce)
        self._warn_audio_permissions()

    @staticmethod
    def _check_audio_groups() -> Tuple[bool, bool, bool]:
        """Check if user is in 'audio' and 'pulse' groups. Returns (has_audio, has_pulse, pulse_group_exists)."""
        try:
            import grp
            user_gids = set(os.getgroups())
            has_audio = grp.getgrnam('audio').gr_gid in user_gids
            try:
                pulse_gid = grp.getgrnam('pulse').gr_gid
                pulse_group_exists = True
                has_pulse = pulse_gid in user_gids
            except KeyError:
                pulse_group_exists = False
                has_pulse = True  # No pulse group = no need to be in it
            return (has_audio, has_pulse, pulse_group_exists)
        except (KeyError, ImportError, OSError):
            return (False, False, False)

    def _warn_audio_permissions(self) -> None:
        """Print a warning if user is not in audio/pulse group (Linux mic access)."""
        try:
            has_audio, has_pulse, pulse_group_exists = self._check_audio_groups()
            if not has_audio:
                print(f"  [Mic] Warning: user not in 'audio' group — mic may fail. Fix: sudo usermod -a -G audio $USER")
            if pulse_group_exists and not has_pulse:
                print(f"  [Mic] Warning: user not in 'pulse' group — mic may fail. Fix: sudo usermod -a -G pulse $USER")
        except Exception:
            pass

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for sounddevice stream."""
        if status:
            if status.input_overflow:
                self._overflow_count += 1
            if status.input_underflow:
                self._underflow_count += 1
            # Only print occasionally to avoid spam
            if (self._overflow_count + self._underflow_count) % 100 == 1:
                print(f"Audio status: {status}")
        
        # Convert to mono float32 if needed
        audio = indata.copy().flatten().astype(np.float32)
        # Apply mic gain (e.g. 1.1 = 10% boost), clip to avoid overflow
        if self.config.mic_gain != 1.0:
            audio = (audio * self.config.mic_gain).clip(-1.0, 1.0)
        
        # Add to alignment buffer
        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, audio])
            
            # Extract complete frames
            while len(self._buffer) >= self.frame_samples:
                frame = self._buffer[:self.frame_samples].copy()
                self._buffer = self._buffer[self.frame_samples:]
                
                # Try to add to queue (non-blocking)
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame to make room
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(frame)
                    except queue.Empty:
                        pass

    def start(self) -> None:
        """Start capturing audio from microphone."""
        if self._running:
            return
        
        self._running = True
        self._overflow_count = 0
        self._underflow_count = 0
        
        # Clear any old data
        with self._buffer_lock:
            self._buffer = np.zeros(0, dtype=np.float32)
        
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        
        try:
            if self.config.backend == "sounddevice":
                self._start_sounddevice()
            else:
                self._start_pyaudio()
        except Exception as e:
            self._running = False
            self._print_mic_permission_help(e)
            raise

        print("Audio capture started")

    def _print_mic_permission_help(self, err: Exception) -> None:
        """Print microphone permission troubleshooting when open fails."""
        has_audio, has_pulse, pulse_group_exists = self._check_audio_groups()
        print(f"  [Mic] Error opening microphone: {err}")
        if not has_audio or (pulse_group_exists and not has_pulse):
            print(f"  [Mic] PERMISSIONS ISSUE DETECTED:")
            if not has_audio:
                print(f"         Add user to 'audio' group: sudo usermod -a -G audio $USER")
            if pulse_group_exists and not has_pulse:
                print(f"         Add user to 'pulse' group: sudo usermod -a -G pulse $USER")
            print(f"         Then logout and login again (or restart)")
        else:
            print(f"  [Mic] TROUBLESHOOTING:")
            print(f"         If you added 'audio' group, logout and login again (or reboot).")
            # Show available input devices
            try:
                devs = self.list_devices()
                if devs:
                    print(f"         Available input devices ({len(devs)}):")
                    for d in devs[:8]:
                        print(f"           [{d['index']}] {d['name']} ({d['sample_rate']} Hz)")
                else:
                    print(f"         No input devices found. Check mic connection and drivers.")
            except Exception:
                print(f"         List devices: python -c \"import sounddevice as sd; print(sd.query_devices())\"")

    def _start_sounddevice(self) -> None:
        """Initialize and start sounddevice stream."""
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype='float32',
            blocksize=self.frame_samples,
            callback=self._audio_callback,
            device=self.config.device,
            latency='low',
        )
        self._stream.start()

    def _start_pyaudio(self) -> None:
        """Initialize and start PyAudio stream (fallback)."""
        import pyaudio
        
        self._pa = pyaudio.PyAudio()
        
        def pa_callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.float32)
            self._audio_callback(audio.reshape(-1, 1), frame_count, time_info, None)
            return (None, pyaudio.paContinue)
        
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.frame_samples,
            stream_callback=pa_callback,
            input_device_index=self.config.device,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        """Stop capturing audio and cleanup resources."""
        if not self._running:
            return
        
        self._running = False
        
        if self._stream is not None:
            if self.config.backend == "sounddevice":
                self._stream.stop()
                self._stream.close()
            else:
                self._stream.stop_stream()
                self._stream.close()
                if hasattr(self, '_pa'):
                    self._pa.terminate()
            
            self._stream = None
        
        # Print statistics
        if self._overflow_count > 0 or self._underflow_count > 0:
            print(f"Audio stats: {self._overflow_count} overflows, {self._underflow_count} underflows")
        
        print("Audio capture stopped")

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get next audio frame from queue.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            np.ndarray of shape (frame_samples,) dtype float32, or None if timeout
        """
        if not self._running:
            return None
        
        try:
            frame = self._queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None

    def frames(self) -> Iterator[np.ndarray]:
        """
        Generator that yields frames continuously until stopped.
        
        Yields:
            np.ndarray of shape (frame_samples,) dtype float32
        """
        while self._running:
            frame = self.get_frame(timeout=0.1)
            if frame is not None:
                yield frame

    @staticmethod
    def list_devices() -> List[Dict]:
        """
        List available audio input devices.
        
        Returns:
            List of dicts with keys: index, name, channels, sample_rate
        """
        devices = []
        
        try:
            import sounddevice as sd
            
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': dev['name'],
                        'channels': dev['max_input_channels'],
                        'sample_rate': int(dev['default_samplerate']),
                        'backend': 'sounddevice',
                    })
        except ImportError:
            pass
        
        # Try PyAudio as fallback
        if not devices:
            try:
                import pyaudio
                pa = pyaudio.PyAudio()
                
                for i in range(pa.get_device_count()):
                    dev = pa.get_device_info_by_index(i)
                    if dev['maxInputChannels'] > 0:
                        devices.append({
                            'index': i,
                            'name': dev['name'],
                            'channels': dev['maxInputChannels'],
                            'sample_rate': int(dev['defaultSampleRate']),
                            'backend': 'pyaudio',
                        })
                
                pa.terminate()
            except ImportError:
                pass
        
        return devices

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# =============================================================================
# TEST / DEMO
# =============================================================================

def test_audio_input():
    """Test audio input with live microphone."""
    print("\n" + "=" * 60)
    print("AUDIO INPUT TEST")
    print("=" * 60)
    
    # List devices
    print("\nAvailable input devices:")
    devices = AudioInput.list_devices()
    for dev in devices:
        print(f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch, {dev['sample_rate']}Hz)")
    
    if not devices:
        print("No audio input devices found!")
        return
    
    print("\n")
    
    # Create audio input
    config = AudioInputConfig(
        sample_rate=16000,
        channels=1,
        frame_ms=20,
        device=None,  # Use default
    )
    
    audio = AudioInput(config)
    
    print("Recording for 5 seconds... Speak into the microphone!")
    print("(You should see RMS levels changing)\n")
    
    audio.start()
    
    start_time = time.time()
    frame_count = 0
    
    try:
        for frame in audio.frames():
            frame_count += 1
            
            # Calculate RMS
            rms = float(np.sqrt(np.mean(frame ** 2)))
            
            # Simple visualization
            bar_len = min(50, int(rms * 200))
            bar = "█" * bar_len
            
            print(f"\rFrame {frame_count:4d} | RMS: {rms:.4f} |{bar:<50}|", end="", flush=True)
            
            # Stop after 5 seconds
            if time.time() - start_time > 5:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        audio.stop()
    
    print(f"\n\nRecorded {frame_count} frames")
    print(f"Expected: {5 * 1000 // config.frame_ms} frames")
    print("Test complete!")


if __name__ == "__main__":
    test_audio_input()
