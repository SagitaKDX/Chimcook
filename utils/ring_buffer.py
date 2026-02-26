"""
Voice Assistant v2 - Ring Buffer
================================

Circular buffer for audio streaming.
Useful for:
- Fixed-size frame output from variable input
- Rolling window for STT
- Audio buffering
"""

import numpy as np
from typing import Optional


class RingBuffer:
    """
    Circular buffer for audio data.
    
    Features:
    - Fixed capacity with automatic wraparound
    - Push variable-size chunks
    - Pop fixed-size frames
    - Get rolling window
    
    Usage:
        buffer = RingBuffer(capacity=16000)  # 1 second at 16kHz
        buffer.push(audio_chunk)
        frame = buffer.pop(frame_size=320)  # 20ms frame
    """
    
    def __init__(self, capacity: int, dtype=np.float32):
        """
        Initialize ring buffer.
        
        Args:
            capacity: Maximum number of samples to store
            dtype: NumPy dtype for audio data
        """
        self.capacity = capacity
        self.dtype = dtype
        self._buffer = np.zeros(capacity, dtype=dtype)
        self._write_pos = 0
        self._read_pos = 0
        self._size = 0
    
    def push(self, data: np.ndarray) -> int:
        """
        Add samples to buffer.
        
        Args:
            data: Audio samples to add
            
        Returns:
            Number of samples actually written (may be less if buffer full)
        """
        data = np.asarray(data, dtype=self.dtype).flatten()
        n = len(data)
        
        if n == 0:
            return 0
        
        # If more data than capacity, only keep last 'capacity' samples
        if n > self.capacity:
            data = data[-self.capacity:]
            n = self.capacity
        
        # Calculate available space
        available = self.capacity - self._size
        to_write = min(n, available)
        
        if to_write == 0:
            return 0
        
        # Write data (may wrap around)
        end_pos = (self._write_pos + to_write) % self.capacity
        
        if end_pos > self._write_pos:
            # No wrap
            self._buffer[self._write_pos:end_pos] = data[:to_write]
        else:
            # Wrap around
            first_part = self.capacity - self._write_pos
            self._buffer[self._write_pos:] = data[:first_part]
            self._buffer[:end_pos] = data[first_part:to_write]
        
        self._write_pos = end_pos
        self._size += to_write
        
        return to_write
    
    def pop(self, n: int) -> Optional[np.ndarray]:
        """
        Remove and return n samples from buffer.
        
        Args:
            n: Number of samples to pop
            
        Returns:
            Audio samples, or None if not enough data
        """
        if n > self._size:
            return None
        
        end_pos = (self._read_pos + n) % self.capacity
        
        if end_pos > self._read_pos:
            # No wrap
            result = self._buffer[self._read_pos:end_pos].copy()
        else:
            # Wrap around
            first_part = self._buffer[self._read_pos:].copy()
            second_part = self._buffer[:end_pos].copy()
            result = np.concatenate([first_part, second_part])
        
        self._read_pos = end_pos
        self._size -= n
        
        return result
    
    def peek(self, n: int) -> Optional[np.ndarray]:
        """
        Return n samples without removing them.
        
        Args:
            n: Number of samples to peek
            
        Returns:
            Audio samples, or None if not enough data
        """
        if n > self._size:
            return None
        
        end_pos = (self._read_pos + n) % self.capacity
        
        if end_pos > self._read_pos:
            return self._buffer[self._read_pos:end_pos].copy()
        else:
            first_part = self._buffer[self._read_pos:].copy()
            second_part = self._buffer[:end_pos].copy()
            return np.concatenate([first_part, second_part])
    
    def get_all(self) -> np.ndarray:
        """
        Get all data in buffer without removing.
        
        Returns:
            All audio samples currently in buffer
        """
        if self._size == 0:
            return np.zeros(0, dtype=self.dtype)
        return self.peek(self._size)
    
    def clear(self) -> None:
        """Clear all data from buffer."""
        self._write_pos = 0
        self._read_pos = 0
        self._size = 0
    
    @property
    def size(self) -> int:
        """Number of samples currently in buffer."""
        return self._size
    
    @property
    def available_space(self) -> int:
        """Number of samples that can be added."""
        return self.capacity - self._size
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._size == 0
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._size == self.capacity


class FrameAligner:
    """
    Aligns variable-size audio chunks into fixed-size frames.
    
    Usage:
        aligner = FrameAligner(frame_size=320)
        
        for chunk in audio_stream:
            aligner.push(chunk)
            while True:
                frame = aligner.pop()
                if frame is None:
                    break
                process(frame)
    """
    
    def __init__(self, frame_size: int, dtype=np.float32):
        """
        Initialize frame aligner.
        
        Args:
            frame_size: Target frame size in samples
            dtype: NumPy dtype for audio
        """
        self.frame_size = frame_size
        self._buffer = np.zeros(0, dtype=dtype)
        self.dtype = dtype
    
    def push(self, data: np.ndarray) -> None:
        """Add audio data to buffer."""
        data = np.asarray(data, dtype=self.dtype).flatten()
        self._buffer = np.concatenate([self._buffer, data])
    
    def pop(self) -> Optional[np.ndarray]:
        """
        Pop one frame if available.
        
        Returns:
            Frame of exactly frame_size samples, or None
        """
        if len(self._buffer) < self.frame_size:
            return None
        
        frame = self._buffer[:self.frame_size]
        self._buffer = self._buffer[self.frame_size:]
        return frame
    
    def pop_all(self):
        """Generator that yields all available frames."""
        while True:
            frame = self.pop()
            if frame is None:
                break
            yield frame
    
    def clear(self) -> None:
        """Clear buffer."""
        self._buffer = np.zeros(0, dtype=self.dtype)
    
    @property
    def buffered_samples(self) -> int:
        """Number of samples in buffer."""
        return len(self._buffer)
