"""
Voice Assistant v2 - Speaker Isolation Module
==============================================

Step 3: Focus on ONE speaker in crowded places.

This is the KEY FEATURE for reliability in noisy environments.

Strategy: The nearest speaker is loudest and clearest.
- Lock onto the first speaker who talks
- Ignore quieter (more distant) speakers
- Release lock after silence

Features:
1. Volume-based speaker tracking
2. Speaker lock mechanism with adaptive baseline
3. Lock release after silence
4. Statistics tracking for debugging
"""

from dataclasses import dataclass
from typing import Optional, Dict
from collections import deque
import numpy as np


@dataclass  
class SpeakerIsolatorConfig:
    """Configuration for speaker isolation."""
    volume_threshold: float = 0.05      # Minimum RMS to consider as speech
    consistency_window: int = 10        # Frames to track for averaging
    min_frames_to_lock: int = 3         # Minimum frames before locking
    volume_margin: float = 0.4          # Accept if within 40% of baseline (was 0.5)
    baseline_adaptation: float = 0.1    # How fast baseline adapts (0.0-1.0)
    release_silence_frames: int = 25    # Frames of silence to release lock (~500ms at 20ms/frame)
    release_quiet_frames: int = 15      # Frames of quiet speech to consider releasing


class SpeakerIsolator:
    """
    Isolates the primary speaker in crowded environments.
    
    Strategy:
    1. Wait for speech above volume threshold
    2. Lock onto that speaker's volume level after a few frames
    3. Reject speech significantly quieter (other speakers)
    4. Release lock after sustained silence
    
    This works because:
    - The user (nearest speaker) is loudest
    - Background speakers are quieter due to distance
    - Locking prevents switching to louder interruptions
    
    Usage:
        isolator = SpeakerIsolator(config)
        
        for frame in audio_frames:
            is_speech = vad.is_speech(frame)
            is_primary = isolator.update(frame, is_speech)
            
            if is_primary:
                # This frame is from the primary speaker
                process_frame(frame)
        
        # After utterance ends:
        isolator.reset()
    """
    
    def __init__(self, config: Optional[SpeakerIsolatorConfig] = None):
        self.config = config or SpeakerIsolatorConfig()
        
        # Tracking state
        self._locked = False
        self._baseline_rms = 0.0
        self._silence_count = 0
        self._quiet_count = 0
        self._speech_count = 0
        self._recent_rms: deque = deque(maxlen=self.config.consistency_window)
        
        # Statistics
        self._stats = {
            "frames_total": 0,
            "frames_accepted": 0,
            "frames_rejected": 0,
            "lock_count": 0,
            "release_count": 0,
        }
    
    def update(self, frame: np.ndarray, is_speech: bool) -> bool:
        """
        Update tracking and determine if frame is from primary speaker.
        
        Args:
            frame: Audio frame (float32)
            is_speech: VAD decision for this frame
            
        Returns:
            True if frame is from primary speaker, False otherwise
        """
        self._stats["frames_total"] += 1
        rms = self.compute_rms(frame)
        
        if is_speech:
            self._silence_count = 0
            self._speech_count += 1
            self._recent_rms.append(rms)
            
            # --- NOT YET LOCKED: Try to lock onto speaker ---
            if not self._locked:
                # Need minimum frames and volume to lock
                if (len(self._recent_rms) >= self.config.min_frames_to_lock and 
                    rms >= self.config.volume_threshold):
                    
                    # Calculate average RMS over recent frames
                    avg_rms = sum(self._recent_rms) / len(self._recent_rms)
                    
                    if avg_rms >= self.config.volume_threshold:
                        # Lock onto this speaker
                        self._locked = True
                        self._baseline_rms = avg_rms
                        self._quiet_count = 0
                        self._stats["lock_count"] += 1
                        self._stats["frames_accepted"] += 1
                        return True
                
                # Not locked yet, but speech detected - accept it
                if rms >= self.config.volume_threshold:
                    self._stats["frames_accepted"] += 1
                    return True
                
                self._stats["frames_rejected"] += 1
                return False
            
            # --- LOCKED: Check if same speaker ---
            # Accept if within margin of baseline
            min_acceptable = self._baseline_rms * self.config.volume_margin
            
            if rms >= min_acceptable:
                # Good volume - same speaker (or closer)
                self._quiet_count = 0
                
                # Adapt baseline slowly (speaker may move)
                adapt = self.config.baseline_adaptation
                self._baseline_rms = (1 - adapt) * self._baseline_rms + adapt * rms
                
                self._stats["frames_accepted"] += 1
                return True
            else:
                # Too quiet - probably different (more distant) speaker
                self._quiet_count += 1
                
                # Release if too many quiet frames (speaker changed)
                if self._quiet_count >= self.config.release_quiet_frames:
                    self._release_lock(reason="quiet_speech")
                
                self._stats["frames_rejected"] += 1
                return False
        
        else:
            # --- SILENCE ---
            self._silence_count += 1
            
            # Release lock after sustained silence
            if self._locked and self._silence_count >= self.config.release_silence_frames:
                self._release_lock(reason="silence")
            
            return False
    
    def _release_lock(self, reason: str = "manual") -> None:
        """Release the speaker lock."""
        if self._locked:
            self._locked = False
            self._baseline_rms = 0.0
            self._quiet_count = 0
            self._recent_rms.clear()
            self._stats["release_count"] += 1
    
    def reset(self) -> None:
        """
        Reset tracking (call after utterance ends or to force release).
        """
        self._locked = False
        self._baseline_rms = 0.0
        self._silence_count = 0
        self._quiet_count = 0
        self._speech_count = 0
        self._recent_rms.clear()
    
    def get_stats(self) -> Dict:
        """
        Get isolation statistics.
        
        Returns:
            Dict with tracking statistics
        """
        total = self._stats["frames_total"]
        accepted = self._stats["frames_accepted"]
        
        return {
            **self._stats,
            "acceptance_ratio": accepted / total if total > 0 else 0,
            "is_locked": self._locked,
            "baseline_rms": self._baseline_rms,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "frames_total": 0,
            "frames_accepted": 0,
            "frames_rejected": 0,
            "lock_count": 0,
            "release_count": 0,
        }
    
    @property
    def is_locked(self) -> bool:
        """Check if locked onto a speaker."""
        return self._locked
    
    @property
    def baseline_rms(self) -> float:
        """Get current baseline RMS (0 if not locked)."""
        return self._baseline_rms
    
    @staticmethod
    def compute_rms(frame: np.ndarray) -> float:
        """
        Compute RMS (Root Mean Square) energy of audio frame.
        
        Args:
            frame: Audio samples
            
        Returns:
            RMS value (0.0 to ~1.0 for normalized audio)
        """
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    """Test speaker isolation module."""
    import time
    
    print("=" * 60)
    print("SPEAKER ISOLATION TEST")
    print("=" * 60)
    
    # Test 1: Basic functionality with synthetic data
    print("\n[Test 1] Basic locking with synthetic data...")
    
    config = SpeakerIsolatorConfig(
        volume_threshold=0.05,
        min_frames_to_lock=3,
        volume_margin=0.4,
        release_silence_frames=10,
    )
    isolator = SpeakerIsolator(config)
    
    frame_size = 320
    
    # Simulate: silence -> loud speech -> quiet speech -> silence
    scenarios = [
        ("Silence", 0.001, False),
        ("Silence", 0.001, False),
        ("Loud speech (primary)", 0.15, True),
        ("Loud speech (primary)", 0.14, True),
        ("Loud speech (primary)", 0.16, True),
        ("Loud speech (primary)", 0.13, True),
        ("Quiet speech (background)", 0.04, True),  # Should be rejected
        ("Quiet speech (background)", 0.03, True),  # Should be rejected
        ("Loud speech (primary)", 0.12, True),      # Should be accepted
        ("Silence", 0.001, False),
        ("Silence", 0.001, False),
    ]
    
    print("\n  Simulating audio frames:")
    for i, (label, volume, is_speech) in enumerate(scenarios):
        frame = np.random.randn(frame_size).astype(np.float32) * volume
        is_primary = isolator.update(frame, is_speech)
        
        status = "✓ ACCEPTED" if is_primary else "✗ rejected"
        lock_status = "LOCKED" if isolator.is_locked else "unlocked"
        
        print(f"    {i:2d}. {label:<25} RMS={volume:.3f} -> {status:<12} [{lock_status}]")
    
    stats = isolator.get_stats()
    print(f"\n  Stats: {stats['frames_accepted']}/{stats['frames_total']} accepted, "
          f"{stats['lock_count']} locks, {stats['release_count']} releases")
    
    # Test 2: Simulate crowded place scenario
    print("\n[Test 2] Simulating crowded place scenario...")
    
    isolator2 = SpeakerIsolator(SpeakerIsolatorConfig(
        volume_threshold=0.05,
        min_frames_to_lock=3,
        volume_margin=0.4,
    ))
    
    # Scenario: Primary speaker talks, background person interrupts, primary continues
    crowd_scenario = [
        # Primary speaker starts (close, loud)
        *[("Primary", 0.12, True) for _ in range(5)],
        
        # Background person talks (far, quiet) - should be ignored
        *[("Background", 0.04, True) for _ in range(3)],
        
        # Primary continues - should be accepted
        *[("Primary", 0.11, True) for _ in range(4)],
        
        # Brief pause
        *[("Silence", 0.001, False) for _ in range(5)],
        
        # Primary speaks again
        *[("Primary", 0.10, True) for _ in range(3)],
    ]
    
    primary_accepted = 0
    background_rejected = 0
    
    for label, volume, is_speech in crowd_scenario:
        frame = np.random.randn(frame_size).astype(np.float32) * volume
        is_primary = isolator2.update(frame, is_speech)
        
        if label == "Primary" and is_primary:
            primary_accepted += 1
        elif label == "Background" and not is_primary:
            background_rejected += 1
    
    total_primary = sum(1 for l, _, _ in crowd_scenario if l == "Primary")
    total_background = sum(1 for l, _, _ in crowd_scenario if l == "Background")
    
    print(f"  Primary speaker frames: {primary_accepted}/{total_primary} accepted")
    print(f"  Background speaker frames: {background_rejected}/{total_background} rejected")
    
    # Test 3: Live audio test
    print("\n[Test 3] Live audio test...")
    
    try:
        from audio_input import AudioInput, AudioInputConfig
        from noise_reduction import NoiseReducer, NoiseReducerConfig
        
        print("  This test requires you to:")
        print("  1. Speak clearly (you are the PRIMARY speaker)")
        print("  2. Have background noise/music/TV (will be rejected)")
        print("\n  Recording for 10 seconds...\n")
        
        # Setup
        audio_input = AudioInput(AudioInputConfig(sample_rate=16000, frame_ms=20))
        noise_reducer = NoiseReducer(NoiseReducerConfig(adaptive=True), sample_rate=16000)
        isolator_live = SpeakerIsolator(SpeakerIsolatorConfig(
            volume_threshold=0.03,
            min_frames_to_lock=5,
            volume_margin=0.35,
            release_silence_frames=30,
        ))
        
        audio_input.start()
        time.sleep(0.2)
        
        # Simple energy-based "VAD" for this test
        energy_threshold = 0.02
        
        start_time = time.time()
        while time.time() - start_time < 10.0:
            frame = audio_input.get_frame(timeout=0.05)
            if frame is not None:
                # Noise reduction
                frame = noise_reducer.process(frame)
                
                # Simple VAD
                rms = isolator_live.compute_rms(frame)
                is_speech = rms > energy_threshold
                
                # Speaker isolation
                is_primary = isolator_live.update(frame, is_speech)
                
                # Visual feedback
                bar = int(rms * 200)
                lock = "LOCKED  " if isolator_live.is_locked else "unlocked"
                status = "PRIMARY " if is_primary else "rejected"
                
                print(f"\r  [{lock}] {status} {'█' * min(bar, 40):<40} RMS={rms:.3f}", end="", flush=True)
        
        audio_input.stop()
        
        live_stats = isolator_live.get_stats()
        print(f"\n\n  Results:")
        print(f"    Frames accepted: {live_stats['frames_accepted']}/{live_stats['frames_total']} "
              f"({live_stats['acceptance_ratio']:.1%})")
        print(f"    Lock events: {live_stats['lock_count']}")
        print(f"    Release events: {live_stats['release_count']}")
        
    except ImportError as e:
        print(f"  Skipped - missing module: {e}")
        print("  Run from project root to test with live audio")
    
    print("\n" + "=" * 60)
    print("SPEAKER ISOLATION TEST COMPLETE")
    print("=" * 60)
