#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Real-time Neural Analysis GUI System
Production-ready system with proper SpikeGLX integration and analysis
Fixed version with improved cleanup and stability
"""

import sys
import time
import threading
import queue
import multiprocessing as mp
from multiprocessing import Manager
import logging
import socket
import struct
import yaml
import colorsys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union, Set, Callable
from collections import deque, OrderedDict
import pandas as pd
from scipy import stats, signal
import psutil
import atexit
import matplotlib as mpl
from multiprocessing.synchronize import Lock as LockType
try:
    from sglx_pkg import sglx
    from ctypes import byref, POINTER, c_int, c_short
except ImportError:
    # Fallback for simulation mode
    pass

# Qt5 imports
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QEvent, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                            QDoubleSpinBox, QComboBox, QGroupBox, QTextEdit,
                            QFileDialog, QTabWidget, QCheckBox, QSplitter,
                            QGridLayout, QProgressBar, QListWidget, QListWidgetItem,
                            QDialog, QDialogButtonBox, QMessageBox, QSlider)
from PyQt5.QtGui import QFont, QCloseEvent, QIcon

# # Qt6 imports
# from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QEvent, pyqtSlot
# from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
#                             QHBoxLayout, QPushButton, QLabel, QSpinBox, 
#                             QDoubleSpinBox, QComboBox, QGroupBox, QTextEdit,
#                             QFileDialog, QTabWidget, QCheckBox, QSplitter,
#                             QGridLayout, QProgressBar, QListWidget, QListWidgetItem,
#                             QDialog, QDialogButtonBox, QMessageBox, QSlider)
# from PyQt6.QtGui import QFont, QCloseEvent, QIcon
import pyqtgraph as pg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RealTimeNeural")

# Global process list for cleanup
ACTIVE_PROCESSES = []

def cleanup_processes():
    """Enhanced cleanup for all active processes"""
    print("Starting cleanup_processes...")
    
    for proc in ACTIVE_PROCESSES:
        try:
            if proc.is_alive():
                print(f"Terminating process: {proc}")
                proc.terminate()
                proc.join(timeout=1.0)
                
                if proc.is_alive():
                    print(f"Force killing process: {proc}")
                    proc.kill()
                    proc.join(timeout=0.5)
        except Exception as e:
            print(f"Error cleaning up process {proc}: {e}")
    
    ACTIVE_PROCESSES.clear()
    print("cleanup_processes completed")

# Register cleanup handler
atexit.register(cleanup_processes)

# =============================================================================
# Configuration Management
# =============================================================================

@dataclass
class SystemConfig:
    """System configuration parameters"""
    # Operation mode
    simulation_mode: bool = True
    
    # SpikeGLX connection (for real mode)
    sglx_host: str = "127.0.0.1"
    sglx_port: int = 4142
    
    # Network settings
    udp_ip: str = "127.0.0.1"
    udp_port: int = 33433
    
    # Data acquisition
    neural_channels: int = 384
    neural_sample_rate: int = 30000
    ttl_sample_rate: int = 10593
    ttl_sync_bit: int = 0
    ttl_event_bit: int = 6
    
    # Synchronization
    sync_epsilon_ms: float = 150.0
    sync_skew_window_size: int = 50
    
    # Processing
    downsample_factor: int = 30
    psth_window_ms: List[float] = field(default_factory=lambda: [-50, 300])
    spike_threshold: float = -4
    
    # Buffers
    ring_buffer_size_mb: int = 1000
    max_trials_per_category: int = 1000
    
    # GUI
    gui_update_interval_ms: int = 500
    monitor_update_interval_ms: int = 500
    
    # File paths
    image_info_path: str = "stim_info.tsv"
    save_path: str = "./data"
    
    # Simulation parameters
    sim_neural_noise_level: float = 30.0
    sim_spike_rate_hz: float = 80.0
    sim_sync_rate_hz: float = 1.0
    sim_event_rate_hz: float = 3.0
    sim_n_categories: int = 4
    
    # D-prime defaults
    dprime_defaults: dict = field(default_factory=lambda: {
        'contrast_threshold': 0.5,
        'time_window_ms': [50, 250]
    })
    
    # Performance
    performance: dict = field(default_factory=lambda: {
        'worker_processes': 4,
        'queue_max_size': 1000
    })
    
    # Logging
    logging: dict = field(default_factory=lambda: {
        'level': 'INFO',
        'file_rotation': True,
        'max_log_size_mb': 100
    })
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from YAML file"""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {filepath}: {e}, using defaults")
            return cls()
    
    def to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        data = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                data[key] = value
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SynchronizedEvent:
    """Synchronized TTL-UDP event pair"""
    ttl_timestamp: int
    ttl_time_ns: int
    udp_timestamp: float
    image_idx: int
    image_name: str
    category: str
    trial_number: int = 0

@dataclass
class ContrastDefinition:
    """Contrast for d-prime analysis"""
    name: str
    positive_categories: Set[str]
    negative_categories: Set[str]

# =============================================================================
# Simulation Producers
# =============================================================================

class SimulatedSpikeGLXProducer(threading.Thread):
    """Enhanced simulated SpikeGLX producer with realistic data patterns"""
    
    def __init__(self, config: SystemConfig, ttl_queue: queue.Queue, 
                 neural_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.ttl_queue = ttl_queue
        self.neural_queue = neural_queue
        self.running = False
        
        # Dynamic parameters
        self.event_rate_hz = config.sim_event_rate_hz
        self.sync_rate_hz = config.sim_sync_rate_hz
        
        # Timing
        self.neural_sample_count = 0
        self.ttl_sample_count = 0
        self.last_neural_time = time.perf_counter()
        self.last_ttl_time = time.perf_counter()
        self.last_sync_time = time.perf_counter()
        self.last_event_time = time.perf_counter()
        self.sync_state = 0
        
        # Neural data buffer for continuous streaming
        self.neural_buffer = np.zeros((0, config.neural_channels), dtype=np.int16)
    
    def set_event_rate(self, rate_hz: float):
        """Update event rate dynamically"""
        self.event_rate_hz = rate_hz
    
    def set_sync_rate(self, rate_hz: float):
        """Update sync rate dynamically"""
        self.sync_rate_hz = rate_hz
    
    def run(self):
        """Main producer loop"""
        self.running = True
        logger.info("Simulated SpikeGLX Producer started")
        # Send rate info at startup (using config values)
        self.neural_queue.put(('rate_info', {
            'neural_rate': self.config.neural_sample_rate,
            'ttl_rate': self.config.ttl_sample_rate
        }))
        while self.running:
            try:
                current_time = time.perf_counter()
                
                # Generate neural data
                self._generate_neural_batch(current_time)
                
                # Generate TTL data
                self._generate_ttl_batch(current_time)
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Simulation producer error: {e}")
                break
    
    def _generate_neural_batch(self, current_time: float):
        """Generate realistic neural data at 30kHz"""
        elapsed = current_time - self.last_neural_time
        expected_samples = int(elapsed * self.config.neural_sample_rate)
        
        if expected_samples > 0:
            # Generate baseline neural data with noise
            neural_data = np.random.randn(expected_samples, self.config.neural_channels)
            neural_data = (neural_data * self.config.sim_neural_noise_level).astype(np.int16)
            
            # Add realistic spike patterns
            n_spikes = np.random.poisson(self.config.sim_spike_rate_hz * elapsed * self.config.neural_channels)
            if n_spikes > 0:
                spike_positions = np.random.randint(0, expected_samples * self.config.neural_channels, n_spikes)
                for pos in spike_positions:
                    sample_idx = pos // self.config.neural_channels
                    channel_idx = pos % self.config.neural_channels
                    if sample_idx < expected_samples:
                        # Create spike waveform
                        spike_amplitude = np.random.randint(500, 2000)
                        neural_data[sample_idx, channel_idx] = -spike_amplitude
                        
                        # Add spike afterpotential
                        if sample_idx + 1 < expected_samples:
                            neural_data[sample_idx + 1, channel_idx] = spike_amplitude // 3
            
            # Add to buffer
            self.neural_buffer = np.vstack([self.neural_buffer, neural_data])
            
            # Send chunks of data
            chunk_size = 1000  # samples
            while len(self.neural_buffer) >= chunk_size:
                chunk = self.neural_buffer[:chunk_size]
                self.neural_buffer = self.neural_buffer[chunk_size:]
                
                # Flatten to match SpikeGLX format
                flat_data = chunk.flatten()
                self.neural_queue.put(('neural', self.neural_sample_count, flat_data))
                self.neural_sample_count += chunk_size
            
            self.last_neural_time = current_time
    
    def _generate_ttl_batch(self, current_time: float):
        """Generate TTL data with sync and event signals"""
        elapsed = current_time - self.last_ttl_time
        expected_samples = int(elapsed * self.config.ttl_sample_rate)
        
        if expected_samples > 0:
            ttl_data = np.zeros(expected_samples, dtype=np.uint16)
            
            # Sync signal (square wave)
            if self.sync_rate_hz > 0 and current_time - self.last_sync_time >= 1.0 / self.sync_rate_hz:
                self.sync_state = 1 - self.sync_state
                self.last_sync_time = current_time
            
            if self.sync_state:
                ttl_data |= (1 << self.config.ttl_sync_bit)
            
            # Event signal (brief pulse)
            if self.event_rate_hz > 0 and current_time - self.last_event_time >= 1.0 / self.event_rate_hz:
                # Create a 10-sample pulse
                pulse_samples = min(10, expected_samples)
                ttl_data[:pulse_samples] |= (1 << self.config.ttl_event_bit)
                
                # Record event timing
                ttl_time_ns = time.perf_counter_ns()
                self.ttl_queue.put(('ttl', self.ttl_sample_count, ttl_time_ns))
                
                self.last_event_time = current_time
                logger.debug(f"Generated TTL event at sample {self.ttl_sample_count}")
            
            # Send TTL data
            self.neural_queue.put(('ttl_data', self.ttl_sample_count, ttl_data))
            self.ttl_sample_count += expected_samples
            self.last_ttl_time = current_time
    
    def stop(self):
        """Stop the producer"""
        self.running = False

class SimulatedUDPProducer(threading.Thread):
    """Simulated UDP producer with realistic event generation"""
    
    def __init__(self, config: SystemConfig, udp_queue: queue.Queue, 
                 stim_info: Optional[pd.DataFrame] = None):
        super().__init__(daemon=True)
        self.config = config
        self.udp_queue = udp_queue
        self.stim_info = stim_info
        self.running = False
        self.event_rate_hz = config.sim_event_rate_hz
        self.last_event_time = time.perf_counter()
        self.event_counter = 0
    
    def set_event_rate(self, rate_hz: float):
        """Update event rate dynamically"""
        self.event_rate_hz = rate_hz
    
    def run(self):
        """Generate UDP events"""
        self.running = True
        logger.info("Simulated UDP Producer started")
        
        while self.running:
            try:
                current_time = time.perf_counter()
                
                if self.event_rate_hz > 0 and current_time - self.last_event_time >= 1.0 / self.event_rate_hz:
                    self._generate_udp_event()
                    self.last_event_time = current_time
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"UDP producer error: {e}")
                break
    
    def _generate_udp_event(self):
        """Generate realistic UDP event"""
        self.event_counter += 1
        
        if self.stim_info is not None and not self.stim_info.empty:
            # Use real stimulus info
            stim = self.stim_info.sample(n=1).iloc[0]
            image_idx = int(stim.get('Idx', stim.get('Index', 0)))
            image_name = str(stim.get('FileName', f'stim_{image_idx:03d}.jpg'))
            # Note: category will be looked up by SynchronizationConsumer
        else:
            # Generate synthetic stimulus info
            image_idx = np.random.randint(0, 100)
            image_name = f'stim_{image_idx:03d}.jpg'
        
        # Create realistic timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        message = f"timestamp={timestamp},value={image_idx},name={image_name}"
        
        event_data = {
            'udp_time': time.perf_counter_ns(),
            'image_idx': image_idx,
            'image_name': image_name,
            'packet_num': self.event_counter,
            'message': message
        }
        
        self.udp_queue.put(('udp', event_data))
        logger.debug(f"Generated UDP event: {image_name}")
    
    def stop(self):
        """Stop the producer"""
        self.running = False

# =============================================================================
# Real Hardware Producers
# =============================================================================

class RealSpikeGLXProducer(threading.Thread):
    """Real SpikeGLX producer with proper SDK usage"""
    
    def __init__(self, config: SystemConfig, ttl_queue: queue.Queue, 
                 neural_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        # Output queue
        self.ttl_queue = ttl_queue
        self.neural_queue = neural_queue
        # initial running state
        self.running = False

        # Get ACTUAL sample rates from SpikeGLX
        self.actual_neural_rate = config.neural_sample_rate  # default
        self.actual_ttl_rate = config.ttl_sample_rate  # default
        self.handle = None
        
        if sys.platform == 'win32':
            import os
            p = psutil.Process(os.getpid())
            # Check current priority
            logger.info(f"Process priority: {p.nice()}")
            
            # Try to set high priority
            try:
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info("Set process to HIGH priority")
            except:
                logger.warning("Could not set high priority - may need admin rights")

        try:
            # Create handle
            self.handle = sglx.c_sglx_createHandle()
            
            # Connect to SpikeGLX
            success = sglx.c_sglx_connect(
                self.handle, 
                self.config.sglx_host.encode(), 
                self.config.sglx_port
            )
            
            if success:
                # Get actual sample rates from SpikeGLX
                # js=2 for imec probe, ip=0 for first probe
                self.actual_neural_rate = sglx.c_sglx_getStreamSampleRate(self.handle, 2, 0)
                # js=0 for NI stream, ip=0
                self.actual_ttl_rate = sglx.c_sglx_getStreamSampleRate(self.handle, 0, 0)
                
                logger.info(f"SpikeGLX actual rates: Neural={self.actual_neural_rate}Hz, "
                          f"TTL={self.actual_ttl_rate}Hz")
                
                # Send rate info through queue for consumers
                self.neural_queue.put(('rate_info', {
                    'neural_rate': self.actual_neural_rate,
                    'ttl_rate': self.actual_ttl_rate
                }))
                self.test_spikeglx_latency() # Test TODO
            else:
                error_msg = sglx.c_sglx_getError(self.handle)
                raise RuntimeError(f"Failed to connect to SpikeGLX: {error_msg}")
                
            logger.info("Connected to SpikeGLX successfully")
            logger.info(f"SpikeGLX neural sampler {self.actual_neural_rate} Hz; TTL sampler {self.actual_ttl_rate} Hz")
        except Exception as e:
            logger.error(f"Failed to initialize SpikeGLX SDK: {e}")
            if self.handle:
                sglx.c_sglx_destroyHandle(self.handle)
            self.handle = None
    
    def test_spikeglx_latency(self): # Test TODO
        """Test SpikeGLX communication latency"""
        if self.handle:
            times = []
            for i in range(10):
                start = time.perf_counter()
                count = sglx.c_sglx_getStreamSampleCount(self.handle, 2, 0)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)
            
            avg_ms = np.mean(times)
            logger.info(f"SpikeGLX latency: avg={avg_ms:.2f}ms, max={np.max(times):.2f}ms, min={np.min(times):.2f}ms")
            
            if avg_ms > 10:
                logger.warning(f"HIGH LATENCY DETECTED: {avg_ms:.2f}ms average")

    def run(self):
        """Main producer loop using c_sglx_fetch for gapless data"""
        if not self.handle:
            logger.error("SpikeGLX SDK not available")
            return
        
        self.running = True
        logger.info("Real SpikeGLX Producer started")
        
        # Test # TODO
        cpu_freq = psutil.cpu_freq()
        logger.info(f"CPU: current={cpu_freq.current:.0f}MHz, max={cpu_freq.max:.0f}MHz")
        logger.info(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        
        # Configure fetch parameters
        fetch_size_neural = 30000   # samples per fetch
        fetch_size_ttl = 10593   
        neural_sample_count = 0
        ttl_sample_count = 0
        
        # Initialize starting sample counts for fetch
        neural_from_ct = sglx.c_sglx_getStreamSampleCount(self.handle, 2, 0)  # imec stream
        ttl_from_ct = sglx.c_sglx_getStreamSampleCount(self.handle, 0, 0)  # NI stream
        
        # Setup channel arrays
        # For neural data - fetch all neural channels
        neural_channels_list = list(range(self.config.neural_channels))
        neural_channels = (c_int * len(neural_channels_list))(*neural_channels_list)
        neural_nc = len(neural_channels_list)
        
        # For TTL data - fetch digital channels (usually starting at 384 for NI)
        # Adjust based on your actual channel configuration
        ttl_channel_idx = [1]
        ttl_channels = (c_int * 1)(*ttl_channel_idx)
        ttl_nc = 1
        last_fecth_time = time.perf_counter()
        while self.running:
            try:
                # Fetch neural data from imec probe
                neural_data_ptr = POINTER(c_short)()
                neural_ndata = c_int()

                pre_fetch = time.perf_counter()
                neural_head_ct = sglx.c_sglx_fetch(
                    byref(neural_data_ptr),
                    byref(neural_ndata),
                    self.handle,
                    2,  # js=2 for imec
                    0,  # ip=0 for first probe
                    neural_from_ct,
                    fetch_size_neural,
                    neural_channels,
                    neural_nc,
                    1  # downsample factor
                )
                fetch_time = time.perf_counter() - pre_fetch
                if fetch_time > 0.1:  # More than 100ms is suspicious
                    logger.warning(f"SLOW FETCH: {fetch_time*1000:.1f}ms for {fetch_size_neural} samples")
                gap_fetch_time = time.perf_counter() - last_fecth_time
                last_fecth_time = time.perf_counter()
                if neural_head_ct > 0:
                    # Convert to numpy array
                    n_samples = neural_ndata.value // neural_nc

                    # Add safety check
                    expected_bytes = neural_ndata.value * 2  # c_short is 2 bytes
                    logger.info(f"DIAGNOSTIC: Fetched {n_samples} samples, {neural_ndata.value} values, "
                            f"{expected_bytes} bytes, Gap {gap_fetch_time}s")
                    
                    if n_samples > 0:
                        try:
                            # Log before the dangerous operation
                            logger.debug(f"DIAGNOSTIC: About to convert {neural_ndata.value} c_short values to numpy")
                            conv_start = time.perf_counter() 
                            neural_data = np.ctypeslib.as_array(
                                neural_data_ptr, 
                                shape=(neural_ndata.value,)
                            ).copy()
                            conv_time = time.perf_counter() - conv_start
                            if conv_time > 0.01:  # More than 10ms is slow
                                logger.warning(f"SLOW CONVERSION: {conv_time*1000:.1f}ms for {neural_ndata.value} values")
                        except Exception as e:
                            logger.error(f"DIAGNOSTIC: Failed to convert neural data: {e}")
                            logger.error(f"DIAGNOSTIC: Pointer value: {neural_data_ptr}, ndata: {neural_ndata.value}")
                            continue
                        self.neural_queue.put(('neural', neural_sample_count, neural_data))
                        neural_sample_count += n_samples
                        neural_from_ct = neural_head_ct + n_samples
                else:
                    logger.warning(f'May lost track of SpikeGLX Stream!')
                # Fetch TTL/digital data from NI stream
                ttl_data_ptr = POINTER(c_short)()
                ttl_ndata = c_int()
                
                ttl_head_ct = sglx.c_sglx_fetch(
                    byref(ttl_data_ptr),
                    byref(ttl_ndata),
                    self.handle,
                    0,  # js=0 for NI
                    0,  # ip=0
                    ttl_from_ct,
                    fetch_size_ttl,
                    ttl_channels,
                    ttl_nc,
                    1  # downsample factor
                )
                
                if ttl_head_ct > 0:
                    n_ttl_samples = ttl_ndata.value // ttl_nc
                    if n_ttl_samples > 0:
                        ttl_data = np.ctypeslib.as_array(
                            ttl_data_ptr,
                            shape=(n_ttl_samples,)
                        ).copy().astype(np.uint16)
                        
                        # Check for event onset (bit 6 = value 64)
                        event_mask = (1 << self.config.ttl_event_bit)
                        # Detect rising edges
                        prev_data = np.roll(ttl_data, 1)
                        prev_data[0] = 0  # Assume no event before first sample
                        event_indices = np.where(
                            (ttl_data & event_mask) & ~(prev_data & event_mask)
                        )[0]
                        
                        for idx in event_indices:
                            ttl_timestamp = ttl_sample_count + idx
                            ttl_time_ns = time.perf_counter_ns()
                            self.ttl_queue.put(('ttl', ttl_timestamp, ttl_time_ns))
                            logger.debug(f"TTL event detected at sample {ttl_timestamp}")
                        
                        # Also send raw TTL data for monitoring
                        self.neural_queue.put(('ttl_data', ttl_sample_count, ttl_data))
                        ttl_sample_count += n_ttl_samples
                        ttl_from_ct = ttl_head_ct + n_ttl_samples
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"SpikeGLX fetch error: {e}")
                if not self.running:
                    break
    
    def stop(self):
        """Stop the producer and clean up"""
        self.running = False
        time.sleep(0.1)
        if self.handle:
            try:
                sglx.c_sglx_close(self.handle)
                sglx.c_sglx_destroyHandle(self.handle)
            except:
                pass

class RealUDPProducer(threading.Thread):
    """Real UDP listener for behavioral events"""
    
    def __init__(self, config: SystemConfig, udp_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.udp_queue = udp_queue
        self.running = False
        self.socket = None
    
    def run(self):
        """Main UDP listener loop"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.config.udp_ip, self.config.udp_port))
        self.socket.settimeout(0.1)
        
        self.running = True
        logger.info(f"Real UDP Producer listening on port {self.config.udp_port}")
        
        packet_num = 0
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                receipt_time = time.perf_counter_ns()
                packet_num += 1
                
                # Parse message
                message = data.decode('utf-8').strip()
                params = self._parse_message(message)
                
                if params:
                    event_data = {
                        'udp_time': receipt_time,
                        'image_idx': int(params.get('value', 0)),
                        'image_name': params.get('name', ''),
                        'packet_num': packet_num,
                        'message': message
                    }
                    self.udp_queue.put(('udp', event_data))
                    logger.debug(f"UDP event received: {params.get('name')}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"UDP error: {e}")
        
        self.cleanup()
    
    def _parse_message(self, message: str) -> Optional[Dict[str, str]]:
        """Parse UDP message in MonkeyLogic format"""
        try:
            params = {}
            for part in message.split(','):
                if '=' in part:
                    key, value = part.strip().split('=', 1)
                    params[key] = value
            return params
        except:
            return None
    
    def stop(self):
        """Stop the listener"""
        self.running = False
    
    def cleanup(self):
        """Clean up resources"""
        if self.socket:
            self.socket.close()

# =============================================================================
# Simple Epoch File Manager
# =============================================================================

class EpochFileManager:
    """Simple file-based epoch data storage"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.epochs_saved = 0
        
        # Create save directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(config.save_path) / f"epochs_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simple metadata file
        self.metadata_file = self.session_dir / "metadata.txt"
        self._init_metadata()
        
        logger.info(f"EpochFileManager saving to: {self.session_dir}")
    
    def _init_metadata(self):
        """Initialize metadata file with header"""
        with open(self.metadata_file, 'w') as f:
            f.write("# Epoch Recording Session\n")
            f.write(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Sample Rate: {self.config.neural_sample_rate} Hz\n")
            f.write(f"# Channels: {self.config.neural_channels}\n")
            f.write("#\n")
            f.write("# Format: trial_number | timestamp | category | image_idx | image_name | neural_sample | filename\n")
            f.write("#" + "-"*70 + "\n")
    
    def save_epoch(self, trial_number: int, category: str, image_idx: int, 
                   image_name: str, neural_data: np.ndarray, 
                   first_sample_count: int = 0) -> bool:
        """Enhanced epoch saving with first sample count"""
        try:
            # Create filename with sample count info
            filename = f"trial_{trial_number:05d}_{category}_{image_idx}_s{first_sample_count}.npy"
            filepath = self.session_dir / filename
            
            # Create a dictionary to save both data and metadata
            epoch_dict = {
                'neural_data': neural_data,
                'first_sample_count': first_sample_count,
                'trial_number': trial_number,
                'category': category,
                'image_idx': image_idx,
                'image_name': image_name,
                'timestamp': time.time()
            }
            
            # Save as numpy file
            np.save(filepath, epoch_dict, allow_pickle=True)
            
            # Append to metadata file with sample count
            with open(self.metadata_file, 'a') as f:
                f.write(f"{trial_number:5d} | {time.time():.3f} | {category:8s} | "
                       f"{image_idx:4d} | {image_name:20s} | {first_sample_count:10d} | {filename}\n")
            
            self.epochs_saved += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to save epoch {trial_number}: {e}")
            return False
    
    def get_save_path(self) -> str:
        """Get the current save directory path"""
        return str(self.session_dir.absolute())
    
    def get_summary(self) -> Dict:
        """Get simple summary"""
        return {
            'epochs_saved': self.epochs_saved,
            'save_path': str(self.session_dir)
        }

# =============================================================================
# Synchronization Consumer 
# =============================================================================

class SynchronizationConsumer(mp.Process):
    """Enhanced synchronization with category lookup and robust matching"""
    
    def __init__(self, config: SystemConfig, ttl_queue: mp.Queue,
                 udp_queue: mp.Queue, sync_queue: mp.Queue,
                 category_lookup: Optional[dict] = None):
        super().__init__(daemon=True)
        self.config = config
        # Input queue
        self.ttl_queue = ttl_queue
        self.udp_queue = udp_queue
        # output queue
        self.sync_queue = sync_queue

        # Build lookup table for categories
        self.category_lookup = category_lookup
    
        # Event buffers
        self.ttl_buffer = deque(maxlen=1000)
        self.udp_buffer = deque(maxlen=1000)
        
        # Clock skew estimation
        self.clock_offsets = deque(maxlen=self.config.sync_skew_window_size)
        
        # Statistics
        self.stats = {
            'matched_count': 0,
            'orphaned_ttl': 0,
            'orphaned_udp': 0,
            'categories_found': set(),
            'last_log_time': time.time()
        }
        
        self.running = True
        self.trial_counter = 0
    
    def run(self):
        """Main synchronization loop"""
        logger.info("Enhanced Synchronization Consumer started")
        
        while self.running:
            try:
                # Collect events
                self._collect_events()
                
                # Match events
                self._match_events()
                
                # Clean old events
                self._cleanup_old_events()
                # Log status every 5 seconds
                if time.time() - self.stats['last_log_time'] > 30.0:
                    if self.stats['matched_count'] > 0 or len(self.ttl_buffer) > 0 or len(self.udp_buffer) > 0:
                        self._log_status()
                    self.stats['last_log_time'] = time.time()
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Sync error: {e}", exc_info=True)
    
    def _log_status(self):
        """Log current status"""
        logger.info(f"SYNC STATUS: Matched={self.stats['matched_count']}, "
                   f"TTL buffer={len(self.ttl_buffer)}, UDP buffer={len(self.udp_buffer)}, "
                   f"Orphaned TTL={self.stats['orphaned_ttl']}, "
                   f"Orphaned UDP={self.stats['orphaned_udp']}, "
                   f"Categories={self.stats['categories_found']}")
    
    def _collect_events(self):
        """Collect events from queues"""
        # Collect TTL events
        MAX_ITERATIONS = 10  # Prevent infinite loops
        # Collect TTL events (bounded)
        for _ in range(MAX_ITERATIONS):
            try:
                event = self.ttl_queue.get_nowait()
                if event[0] == 'ttl':
                    self.ttl_buffer.append({
                        'sample': event[1],
                        'ttl_time': event[2],
                        'received_at': time.time()
                    })
            except:
                break  # Queue is empty, exit loop
        
        # Collect UDP events (bounded)
        for _ in range(MAX_ITERATIONS):
            try:
                event = self.udp_queue.get_nowait()
                if event[0] == 'udp':
                    self.udp_buffer.append({
                        **event[1],
                        'received_at': time.time()
                    })
            except:
                break  # Queue is empty, exit loop
        
        # Log warning if we hit the limit (queue might be backing up)
        if len(self.ttl_buffer) >= MAX_ITERATIONS or len(self.udp_buffer) >= MAX_ITERATIONS:
            current_time = time.time()
            if not hasattr(self, '_last_overflow_warning'):
                self._last_overflow_warning = 0
            
            if current_time - self._last_overflow_warning > 5.0:
                logger.warning(f"Event collection may be falling behind. "
                            f"TTL buffer: {len(self.ttl_buffer)}, "
                            f"UDP buffer: {len(self.udp_buffer)}")
                self._last_overflow_warning = current_time
    
    def _match_events(self):
        """Match TTL and UDP events with category lookup"""
        epsilon_ns = self.config.sync_epsilon_ms * 1e6
        matched_indices = []
        logger.debug(f"DIAGNOSTIC: Matching with {len(self.ttl_buffer)} TTL and {len(self.udp_buffer)} UDP events")

        for i, ttl_event in enumerate(self.ttl_buffer):
            for j, udp_event in enumerate(self.udp_buffer):
                # Calculate time difference
                time_diff = abs(ttl_event['ttl_time'] - udp_event['udp_time'])
                
                if time_diff < epsilon_ns:
                    # Match found - lookup category
                    image_idx = udp_event['image_idx']
                    category = self.category_lookup.get(image_idx, 'unknown')
                    
                    # Create synchronized event
                    sync_event = SynchronizedEvent(
                        ttl_timestamp=ttl_event['sample'],
                        ttl_time_ns=ttl_event['ttl_time'],
                        udp_timestamp=udp_event['udp_time'],
                        image_idx=image_idx,
                        image_name=udp_event['image_name'],
                        category=category,
                        trial_number=self.trial_counter
                    )
                    
                    self.sync_queue.put(sync_event)
                    self.trial_counter += 1
                    
                    # Update statistics
                    self.stats['matched_count'] += 1
                    self.stats['categories_found'].add(category)
                    
                    # Track clock offset for skew estimation
                    offset = ttl_event['ttl_time'] - udp_event['udp_time']
                    self.clock_offsets.append(offset)
                    
                    matched_indices.append((i, j))
                    
                    logger.debug(f"Matched event: {udp_event['image_name']} -> {category}")
                    break

        logger.debug(f"DIAGNOSTIC: Attempting to delete {len(matched_indices)} matched pairs")
        # Remove matched events
        for i, j in sorted(matched_indices, reverse=True):
            try:
                del self.ttl_buffer[i]
                del self.udp_buffer[j]
            except IndexError as e:
                logger.error(f"DIAGNOSTIC: Index error during deletion - ttl[{i}], udp[{j}], "
                            f"buffer sizes: ttl={len(self.ttl_buffer)}, udp={len(self.udp_buffer)}")
    
    def _cleanup_old_events(self):
        """Remove events that are too old to match"""
        current_time = time.time()
        timeout = 2.0  # seconds
        
        # Clean TTL buffer
        while self.ttl_buffer and current_time - self.ttl_buffer[0]['received_at'] > timeout:
            self.stats['orphaned_ttl'] += 1
            self.ttl_buffer.popleft()
        
        # Clean UDP buffer
        while self.udp_buffer and current_time - self.udp_buffer[0]['received_at'] > timeout:
            self.stats['orphaned_udp'] += 1
            self.udp_buffer.popleft()
        
        # Log statistics periodically
        if self.stats['matched_count'] % 100 == 0 and self.stats['matched_count'] > 0:
            logger.info(f"Sync stats: {self.stats}")
    
    def stop(self):
        """Stop the consumer"""
        self.running = False

# =============================================================================
# Analysis Pipeline (unchanged from original)
# =============================================================================

class AnalysisPipeline(mp.Process):
    """Neural data analysis pipeline with epoch extraction and disk saving"""
    
    def __init__(self, config: SystemConfig, sync_queue: mp.Queue,
                 neural_queue: mp.Queue, psth_dict: dict, psth_lock: LockType,
                 epoch_save_enabled: mp.Value):
        super().__init__(daemon=True)
        self.config = config
        self.sync_queue = sync_queue
        self.neural_queue = neural_queue
        self.psth_dict = psth_dict
        self.psth_lock = psth_lock
        self.epoch_save_enabled = epoch_save_enabled
        self.epoch_manager = None  # Will be created when saving is enabled
        self.running = True
        
        # Calculate window parameters
        self.pre_samples = int(abs(config.psth_window_ms[0]) * config.neural_sample_rate / 1000)
        self.post_samples = int(config.psth_window_ms[1] * config.neural_sample_rate / 1000)
        self.total_samples = self.pre_samples + self.post_samples
        
        # Downsampling parameters
        self.downsampled_samples = self.total_samples // config.downsample_factor
        
        # Neural data ring buffer
        self.buffer_size = config.neural_sample_rate * 10
        self.neural_ring_buffer = np.zeros((self.buffer_size, config.neural_channels), dtype=np.int16)
        
        # Buffer tracking
        self.buffer_write_pos = 0
        self.first_sample_number = None
        self.latest_sample_number = None
        self.total_samples_in_buffer = 0
        
        # PSTH storage
        self.psth_data = {}
        self.trial_counts = {}
        self.categories = {}
        
        # Spike detection parameters
        self.spike_threshold = config.spike_threshold
        
        self.max_events_per_iteration = 10
        self.pending_events = deque(maxlen=1000)
        self.dropped_events = 0
        self.last_warning_time = 0

        # Sample rate conversion
        self.ttl_to_neural_ratio = None
        self.actual_neural_rate = config.neural_sample_rate
        self.actual_ttl_rate = config.ttl_sample_rate
        self.rates_initialized = False

        logger.info(f"Analysis Pipeline initialized: window={config.psth_window_ms}ms, "
                   f"downsample={config.downsample_factor}x")
    
    def run(self):
        """Main analysis loop"""
        logger.info("Analysis Pipeline started")
        
        while self.running:
            try:
                if hasattr(self, 'control_queue'):
                    try:
                        command = self.control_queue.get_nowait()
                        if command == 'reset':
                            self.reset_internal_state()
                    except:
                        pass
                
                self._process_neural_data()
                self._process_events()
                
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Analysis error: {e}", exc_info=True)
    
    def _process_neural_data(self):
        """Process incoming neural data"""
        processed_count = 0
        start_time = time.perf_counter()
        MAXIMUM_ITERATION = 20
        try:
            for _ in range(MAXIMUM_ITERATION):
                data = self.neural_queue.get_nowait()
                processed_count += 1

                if data[0] == 'rate_info':
                    rate_info = data[1]
                    self.actual_neural_rate = int(rate_info['neural_rate'])
                    self.actual_ttl_rate = int(rate_info['ttl_rate'])
                    self.ttl_to_neural_ratio = self.actual_neural_rate / self.actual_ttl_rate
                    self.rates_initialized = True
                    
                    logger.info(f"Analysis Pipeline using rates: "
                              f"Neural={self.actual_neural_rate}Hz, "
                              f"TTL={self.actual_ttl_rate}Hz, "
                              f"Ratio={self.ttl_to_neural_ratio:.4f}")
                    
                    self.pre_samples = int(abs(self.config.psth_window_ms[0]) * 
                                          self.actual_neural_rate / 1000)
                    self.post_samples = int(self.config.psth_window_ms[1] * 
                                           self.actual_neural_rate / 1000)
                    self.total_samples = self.pre_samples + self.post_samples
                    continue

                if data[0] == 'neural':
                    sample_count = data[1]
                    neural_data = data[2]
                    
                    n_samples = len(neural_data) // self.config.neural_channels
                    neural_data = neural_data.reshape(n_samples, self.config.neural_channels)
                    
                    if self.first_sample_number is None:
                        self.first_sample_number = sample_count
                        self.latest_sample_number = sample_count
                        logger.info(f"First neural sample number: {self.first_sample_number}")
                        self.neural_ring_buffer = np.zeros((self.buffer_size, self.config.neural_channels), dtype=np.int16)
                        self.buffer_write_pos = 0
                        self.total_samples_in_buffer = 0

                    expected_sample = (self.latest_sample_number if self.latest_sample_number is not None 
                                     else sample_count)
                    if sample_count != expected_sample and self.latest_sample_number is not None:
                        gap = sample_count - expected_sample
                        if gap > 0:
                            logger.warning(f"Gap in neural data: expected {expected_sample}, got {sample_count} (gap={gap})")
                            for _ in range(min(gap, self.buffer_size)):
                                self.neural_ring_buffer[self.buffer_write_pos] = 0
                                self.buffer_write_pos = (self.buffer_write_pos + 1) % self.buffer_size
                                self.total_samples_in_buffer = min(self.total_samples_in_buffer + 1, self.buffer_size)
                    
                    start_pos = self.buffer_write_pos
                    end_pos = start_pos + n_samples
                    if end_pos <= self.buffer_size:
                        self.neural_ring_buffer[start_pos:end_pos] = neural_data
                    else:
                        part1_size = self.buffer_size - start_pos
                        self.neural_ring_buffer[start_pos:] = neural_data[:part1_size]
                        part2_size = n_samples - part1_size
                        self.neural_ring_buffer[:part2_size] = neural_data[part1_size:]
                    
                    self.buffer_write_pos = end_pos % self.buffer_size
                    self.total_samples_in_buffer = min(self.total_samples_in_buffer + n_samples, self.buffer_size)
                    
                    self.latest_sample_number = sample_count + n_samples

        except queue.Empty:
            pass                    
        except Exception as e:
            # THIS IS KEY - log any exceptions that might be silently failing
            logger.error(f"DIAGNOSTIC: Exception in _process_neural_data: {e}", exc_info=True)
        
        elapsed = time.perf_counter() - start_time
        if processed_count > 0:
            logger.info(f"DIAGNOSTIC: Processed {processed_count} neural messages in {elapsed*1000:.1f}ms "
                    f"({processed_count/elapsed:.0f} msg/sec)")
    
    def _process_events(self):
        """Process synchronized events"""
        events_processed = 0
        
        for _ in range(min(len(self.pending_events), 5)):
            if not self.pending_events:
                break
                
            event = self.pending_events.popleft()
            has_data, status = self._has_sufficient_data(event)
            if has_data:
                self._extract_and_process_epoch(event)
                events_processed += 1
            else:
                self.pending_events.append(event)

        for _ in range(self.max_events_per_iteration):
            try:
                event = self.sync_queue.get_nowait()
                
                has_data, status = self._has_sufficient_data(event)
                if has_data:
                    self._extract_and_process_epoch(event)
                    events_processed += 1
                else:
                    if len(self.pending_events) < self.pending_events.maxlen:
                        self.pending_events.append(event)
                        logger.debug(f"Event pending: {status}")
                    else:
                        self.dropped_events += 1
            except:
                break

        if hasattr(self, 'total_events_processed'):
            self.total_events_processed += events_processed
        else:
            self.total_events_processed = events_processed
        
        if self.total_events_processed > 0 and self.total_events_processed % 10 == 0:
            # Only log if it's been at least 10 seconds since last log
            current_time = time.time()
            if not hasattr(self, '_last_analysis_log') or current_time - self._last_analysis_log > 10:
                logger.info(f"ANALYSIS: Processed {self.total_events_processed} events, "
                            f"Pending: {len(self.pending_events)}, "
                            f"Dropped: {self.dropped_events}")
                self._last_analysis_log = current_time
    
    def _has_sufficient_data(self, event: SynchronizedEvent) -> Tuple[bool, str]:
        """Check if we have sufficient data for processing"""
        if not self.rates_initialized or self.latest_sample_number is None:
            return False, "not_initialized"
        
        if self.first_sample_number is None or self.latest_sample_number is None:
            return False, "no_neural_data_yet"
        
        neural_timestamp = int(event.ttl_timestamp * self.ttl_to_neural_ratio)
        
        oldest_sample = self._get_oldest_sample_in_buffer()
        if oldest_sample is None:
            return False, "buffer_not_initialized"
        newest_sample = self.latest_sample_number
        
        event_start = neural_timestamp - self.pre_samples
        event_end = neural_timestamp + self.post_samples
        
        if event_end < oldest_sample:
            return False, f"expired (event_end={event_end} < oldest={oldest_sample})"
        
        if event_end > newest_sample:
            missing_samples = event_end - newest_sample
            return False, f"waiting_{missing_samples}_samples"
        
        if event_start < oldest_sample:
            return False, f"partial_overwrite (start={event_start} < oldest={oldest_sample})"
        
        return True, "ready"

    def _get_oldest_sample_in_buffer(self) -> int:
        """Calculate the oldest sample number currently in buffer"""
        if self.first_sample_number is None or self.latest_sample_number is None:
            return None
        
        if self.total_samples_in_buffer < self.buffer_size:
            return self.first_sample_number
        
        return self.latest_sample_number - self.buffer_size
    
    def _extract_and_process_epoch(self, event: SynchronizedEvent):
        """Extract neural epoch and optionally save to disk"""
        if not self.rates_initialized:
            logger.warning("Cannot process event - pipeline not initialized")
            return
        
        if self.first_sample_number is None or self.latest_sample_number is None:
            logger.warning("Cannot process event - no neural data received yet")
            return
        
        neural_timestamp = int(event.ttl_timestamp * self.ttl_to_neural_ratio)
        
        oldest_sample = self._get_oldest_sample_in_buffer()
        if oldest_sample is None:
            logger.warning("Cannot process event - buffer not initialized")
            return
        
        logical_position = neural_timestamp - oldest_sample
        
        if logical_position < self.pre_samples:
            logger.warning(f"Event at neural sample {neural_timestamp} missing pre-data")
            return
        
        if logical_position + self.post_samples > self.total_samples_in_buffer:
            logger.warning(f"Event at neural sample {neural_timestamp} missing post-data")
            return
        
        # Calculate physical positions in circular buffer
        if self.total_samples_in_buffer < self.buffer_size:
            oldest_physical_pos = 0
        else:
            oldest_physical_pos = self.buffer_write_pos
        
        # Extract epoch
        epoch_start_logical = logical_position - self.pre_samples
        epoch_start_physical = (oldest_physical_pos + epoch_start_logical) % self.buffer_size
        epoch_length = self.pre_samples + self.post_samples
        
        if epoch_start_physical + epoch_length <= self.buffer_size:
            epoch = self.neural_ring_buffer[epoch_start_physical:epoch_start_physical + epoch_length].copy()
        else:
            first_part_size = self.buffer_size - epoch_start_physical
            second_part_size = epoch_length - first_part_size
            
            first_part = self.neural_ring_buffer[epoch_start_physical:].copy()
            second_part = self.neural_ring_buffer[:second_part_size].copy()
            epoch = np.vstack([first_part, second_part])
        
        logger.debug(f"Extracted epoch for event at TTL={event.ttl_timestamp}, "
                    f"neural={neural_timestamp}, logical_pos={logical_position}")
        
        # Process epoch data
        spike_trains = self._detect_spikes(epoch)
        downsampled_spikes = self._downsample_spikes(spike_trains)
        
        # Calculate first sample coun
        first_sample_count = neural_timestamp - self.pre_samples
        # Save epoch to disk if enabled
        if self.epoch_save_enabled.value:
            # Create epoch manager on first save
            if self.epoch_manager is None:
                self.epoch_manager = EpochFileManager(self.config)
                logger.info(f"Created epoch file manager, saving to: {self.epoch_manager.get_save_path()}")
                self._update_epoch_status()
    
            # Save the epoch
            if self.epoch_manager.save_epoch(
                trial_number=event.trial_number,
                category=event.category,
                image_idx=event.image_idx,
                image_name=event.image_name,
                neural_data=epoch,
                first_sample_count=first_sample_count
            ):
                if self.epoch_manager.epochs_saved % 10 == 0:
                    logger.info(f"Saved {self.epoch_manager.epochs_saved} epochs to disk")
        
        # Update PSTH data
        if event.image_idx not in self.psth_data:
            self.psth_data[event.image_idx] = np.zeros(
                (self.downsampled_samples, self.config.neural_channels),
                dtype=np.float32
            )
            self.trial_counts[event.image_idx] = 0
            self.categories[event.image_idx] = event.category
        
        n = self.trial_counts[event.image_idx]
        self.psth_data[event.image_idx] = (
            self.psth_data[event.image_idx] * n + downsampled_spikes
        ) / (n + 1)
        self.trial_counts[event.image_idx] += 1

        # Update shared dict periodically
        if event.trial_number % 5 == 0:
            data_copy = {k: v.copy() for k, v in self.psth_data.items()}
            counts_copy = self.trial_counts.copy()
            categories_copy = self.categories.copy()
            
            acquired = self.psth_lock.acquire(timeout=0.002)
            if acquired:
                try:
                    self.psth_dict['data'] = data_copy
                    self.psth_dict['counts'] = counts_copy
                    self.psth_dict['categories'] = categories_copy
                    self._update_epoch_status_in_dict()
                    self.psth_dict['last_update'] = time.time()
                finally:
                    self.psth_lock.release()
            else:
                logger.debug(f"Skipped shared dict update for trial {event.trial_number}")
        # Also update epoch status every trial if saving is enabled
        elif self.epoch_save_enabled.value and self.epoch_manager:
            self._update_epoch_status()
        logger.info(f'Processed trial {event.trial_number}: {event.image_name} ({event.category})')
    
    def _update_epoch_status(self):
        """Update epoch status in shared dict"""
        if self.psth_lock.acquire(timeout=0.002):
            try:
                self._update_epoch_status_in_dict()
            finally:
                self.psth_lock.release()

    def _update_epoch_status_in_dict(self):
        """Update epoch status in dict (assumes lock is held)"""
        self.psth_dict['epoch_status'] = {
            'enabled': self.epoch_save_enabled.value,
            'saved_count': self.epoch_manager.epochs_saved if self.epoch_manager else 0,
            'save_path': str(self.epoch_manager.get_save_path()) if self.epoch_manager else ''
        }

    def _detect_spikes(self, epoch: np.ndarray) -> np.ndarray:
        """Detect spikes in neural data"""
        if epoch.dtype != np.float32:
            data = epoch.astype(np.float32)
        else:
            data = epoch
        
        abs_data = np.abs(data, out=np.empty_like(data))
        mad_per_channel = np.median(abs_data, axis=0)
        
        thresholds = (self.spike_threshold / 0.6745) * mad_per_channel
        
        return (data < thresholds).astype(np.int8)
    
    def _downsample_spikes(self, spike_trains: np.ndarray) -> np.ndarray:
        """Downsample spike trains"""
        n_samples, n_channels = spike_trains.shape
    
        n_complete_bins = n_samples // self.config.downsample_factor
        truncated_samples = n_complete_bins * self.config.downsample_factor
        
        spike_trains_truncated = spike_trains[:truncated_samples]
        reshaped = spike_trains_truncated.reshape(
            n_complete_bins, 
            self.config.downsample_factor, 
            n_channels
        )
        
        downsampled = reshaped.sum(axis=1) * (1000.0 / self.config.downsample_factor)
        
        if n_samples > truncated_samples:
            remaining = spike_trains[truncated_samples:]
            last_bin = remaining.sum(axis=0) * (1000.0 / remaining.shape[0])
            downsampled = np.vstack([downsampled, last_bin[np.newaxis, :]])
        
        return downsampled
    
    def reset_internal_state(self):
        """Reset analysis pipeline internal state"""
        logger.info("Resetting analysis pipeline internal state")
        
        epochs_saved = self.epoch_manager.epochs_saved if self.epoch_manager else 0
        logger.info(f"Before reset - Trials: {sum(self.trial_counts.values())}, "
                   f"Categories: {len(self.categories)}, "
                   f"Buffer samples: {self.total_samples_in_buffer}, "
                   f"Epochs saved: {epochs_saved}")
        
        self.psth_data = {}
        self.trial_counts = {}
        self.categories = {}
        self.pending_events.clear()
        self.dropped_events = 0
        self.buffer_write_pos = 0
        self.first_sample_number = None
        self.latest_sample_number = None
        self.total_samples_in_buffer = 0
        
        if hasattr(self, 'total_events_processed'):
            self.total_events_processed = 0
        
        if self.epoch_manager:
            logger.info(f"Epochs saved to: {self.epoch_manager.get_save_path()}")
        
        logger.info("Analysis pipeline internal state reset complete")
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
        if self.epoch_manager:
            summary = self.epoch_manager.get_summary()
            logger.info(f"Epoch saving complete: {summary['epochs_saved']} epochs saved to {summary['save_path']}")

# =============================================================================
# D-prime Analyzer
# =============================================================================

class DPrimeAnalyzer:
    """D-prime analyzer for flexible contrast definitions"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def calculate_contrast(self, psth_data: dict, contrast: ContrastDefinition,
                          time_window_ms: Tuple[float, float]) -> np.ndarray:
        """Calculate d-prime for a user-defined contrast"""
        
        if not psth_data or 'data' not in psth_data:
            return np.zeros(self.config.neural_channels)
        
        start_ms, end_ms = self.config.psth_window_ms
        window_start, window_end = time_window_ms
        
        total_samples = list(psth_data['data'].values())[0].shape[0]
        time_points = np.linspace(start_ms, end_ms, total_samples)
        
        start_idx = np.argmin(np.abs(time_points - window_start))
        end_idx = np.argmin(np.abs(time_points - window_end))
        
        positive_data = []
        negative_data = []
        
        categories = psth_data.get('categories', {})
        
        for stim_idx, psth in psth_data['data'].items():
            category = categories.get(stim_idx, '')
            
            if category in contrast.positive_categories:
                positive_data.append(psth[start_idx:end_idx].mean(axis=0))
            elif category in contrast.negative_categories:
                negative_data.append(psth[start_idx:end_idx].mean(axis=0))
        
        if not positive_data or not negative_data:
            logger.warning("Insufficient data for d-prime calculation")
            return np.zeros(self.config.neural_channels)
        
        positive_data = np.array(positive_data)
        negative_data = np.array(negative_data)
        
        d_primes = np.zeros(self.config.neural_channels)
        
        for ch in range(self.config.neural_channels):
            pos_mean = positive_data[:, ch].mean()
            neg_mean = negative_data[:, ch].mean()
            
            pos_std = positive_data[:, ch].std()
            neg_std = negative_data[:, ch].std()
            pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
            
            if pooled_std > 0:
                d_primes[ch] = (pos_mean - neg_mean) / pooled_std
    
        return d_primes

# =============================================================================
# Shutdown Manager
# =============================================================================

class ShutdownManager:
    """Centralized shutdown manager to handle all cleanup operations"""
    
    def __init__(self):
        self.is_shutting_down = False
        self.shutdown_lock = threading.Lock()
        self.cleanup_callbacks = []
        self.active_processes = []
        self.active_threads = []
        self.active_timers = []
        self.resources = {}
        
    def register_cleanup(self, callback, priority=0):
        """Register a cleanup callback with priority (lower = earlier)"""
        self.cleanup_callbacks.append((priority, callback))
        self.cleanup_callbacks.sort(key=lambda x: x[0])
    
    def register_process(self, process, name=None):
        """Register a process for cleanup with optional name"""
        if process not in [p[0] for p in self.active_processes]:
            self.active_processes.append((process, name or str(process)))
    
    def register_thread(self, thread, name=None):
        """Register a thread for cleanup with optional name"""
        if thread not in [t[0] for t in self.active_threads]:
            self.active_threads.append((thread, name or str(thread)))
    
    def register_timer(self, timer):
        """Register a QTimer for cleanup"""
        if timer not in self.active_timers:
            self.active_timers.append(timer)
    
    def register_resource(self, name: str, resource: Any):
        """Register a generic resource for cleanup"""
        self.resources[name] = resource
    
    def unregister_process(self, process):
        """Remove a process from tracking"""
        self.active_processes = [(p, n) for p, n in self.active_processes if p != process]
    
    def unregister_thread(self, thread):
        """Remove a thread from tracking"""
        self.active_threads = [(t, n) for t, n in self.active_threads if t != thread]

    def shutdown(self, force=False):
        """Execute complete shutdown sequence"""
        with self.shutdown_lock:
            if self.is_shutting_down:
                return False
            self.is_shutting_down = True
        
        logger.info("Starting shutdown sequence...")
        
        try:
            self._stop_timers()
            self._execute_callbacks()
            self._stop_threads(force)
            self._stop_processes(force)
            self._cleanup_resources()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("Shutdown sequence completed")
        return True
    
    def _stop_timers(self):
        """Stop all registered QTimers"""
        for timer in self.active_timers:
            try:
                if timer and timer.isActive():
                    timer.stop()
                    logger.debug(f"Stopped timer: {timer}")
            except Exception as e:
                logger.error(f"Error stopping timer: {e}")
        self.active_timers.clear()
    
    def _execute_callbacks(self):
        """Execute cleanup callbacks in priority order"""
        for priority, callback in self.cleanup_callbacks:
            try:
                logger.debug(f"Executing cleanup callback priority {priority}")
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
    
    def _stop_threads(self, force=False):
        """Stop all registered threads"""
        logger.info(f"Stopping {len(self.active_threads)} threads...")
        # First pass: request stop
        for thread, name in self.active_threads:
            try:
                logger.debug(f"Requesting stop for thread: {name}")
                if hasattr(thread, 'stop'):
                    thread.stop()
                if hasattr(thread, 'running'):
                    thread.running = False
            except Exception as e:
                logger.error(f"Error stopping thread {name}: {e}")
        # Second pass: join with timeout
        for thread, name in self.active_threads:
            try:
                if hasattr(thread, 'join') and thread.is_alive():
                    logger.debug(f"Joining thread: {name}")
                    thread.join(timeout=2.0 if not force else 0.5)
                    if thread.is_alive():
                        logger.warning(f"Thread {name} did not stop cleanly")
            except Exception as e:
                logger.error(f"Error joining thread {name}: {e}")
        self.active_threads.clear()
    
    def _stop_processes(self, force=False):
        """Stop all registered processes with proper sequencing"""
        logger.info(f"Stopping {len(self.active_processes)} processes...")
        
        # First pass: request stop
        for process, name in self.active_processes:
            try:
                logger.debug(f"Requesting stop for process: {name}")
                if hasattr(process, 'stop'):
                    process.stop()
                if hasattr(process, 'running'):
                    process.running = False
            except Exception as e:
                logger.error(f"Error stopping process {name}: {e}")
        
        # Give processes time to stop gracefully
        time.sleep(0.5)
        
        # Second pass: terminate
        for process, name in self.active_processes:
            try:
                if hasattr(process, 'is_alive') and process.is_alive():
                    logger.debug(f"Terminating process: {name}")
                    process.terminate()
                    process.join(timeout=2.0 if not force else 0.5)
                    
                    if process.is_alive():
                        logger.warning(f"Force killing process: {name}")
                        process.kill()
                        process.join(timeout=0.5)
            except Exception as e:
                logger.error(f"Error terminating process {name}: {e}")
        
        self.active_processes.clear()
    
    def _cleanup_resources(self):
        """Cleanup generic resources"""
        for name, resource in self.resources.items():
            try:
                logger.debug(f"Cleaning up resource: {name}")
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'shutdown'):
                    resource.shutdown()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up {name}: {e}")
        self.resources.clear()

# =============================================================================
# System Monitor
# =============================================================================

class SystemMonitor(QObject):
    """System monitor with comprehensive metrics"""
    
    stats_updated = pyqtSignal(dict)
    
    def __init__(self, config: SystemConfig):
        super().__init__()
        self.config = config
        self.queues = {}
        self.metrics = {}
        self.process = psutil.Process()
        self.counters = {'ttl': 0, 'udp': 0, 'sync': 0, 'neural': 0}
        self.last_update = time.time()
        self.last_counters = self.counters.copy()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)

    def register_queue(self, name: str, queue_obj):
        """Register queue for monitoring"""
        self.queues[name] = queue_obj
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment counter"""
        if name in self.counters:
            self.counters[name] += amount
    
    def start_monitoring(self):
        """Start monitoring"""
        self.timer.start(self.config.monitor_update_interval_ms)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.timer.isActive():
            self.timer.stop()
    
    def update_metrics(self):
        """Update all system metrics"""
        current_time = time.time()
        time_delta = current_time - self.last_update
        
        for name, queue_obj in self.queues.items():
            try:
                if hasattr(queue_obj, 'qsize'):
                    self.metrics[f'{name}_queue_size'] = queue_obj.qsize()
            except:
                self.metrics[f'{name}_queue_size'] = 0
        
        if time_delta > 0:
            for name in self.counters:
                rate = (self.counters[name] - self.last_counters[name]) / time_delta
                self.metrics[f'{name}_rate'] = rate
            self.last_counters = self.counters.copy()
            self.last_update = current_time
        
        self.metrics['cpu_percent'] = self.process.cpu_percent()
        self.metrics['memory_percent'] = self.process.memory_percent()
        self.metrics['memory_mb'] = self.process.memory_info().rss / (1024 * 1024)
        
        self.metrics['neural_data_rate_mb'] = (
            self.metrics.get('neural_rate', 0) * 
            self.config.neural_channels * 2 / (1024 * 1024)
        )
        
        self.stats_updated.emit(self.metrics)

    def cleanup(self):
        """Clean shutdown"""
        self.stop_monitoring()
        try:
            self.stats_updated.disconnect()
        except:
            pass
        self.queues.clear()
        self.metrics.clear()

# =============================================================================
# Contrast Dialog
# =============================================================================

class ContrastDialog(QDialog):
    """Dialog for defining custom contrasts"""
    
    def __init__(self, categories: List[str], parent=None):
        super().__init__(parent)
        self.categories = categories
        self.contrast = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Define Contrast")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Contrast name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Contrast Name:"))
        self.name_input = QTextEdit()
        self.name_input.setMaximumHeight(30)
        self.name_input.setPlainText("Custom Contrast")
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Positive categories
        layout.addWidget(QLabel("Select Positive Categories (Group A):"))
        self.positive_list = QListWidget()
        self.positive_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for cat in self.categories:
            self.positive_list.addItem(cat)
        layout.addWidget(self.positive_list)
        
        # Negative categories
        layout.addWidget(QLabel("Select Negative Categories (Group B):"))
        self.negative_list = QListWidget()
        self.negative_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for cat in self.categories:
            self.negative_list.addItem(cat)
        layout.addWidget(self.negative_list)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def accept(self):
        """Validate and accept contrast definition"""
        positive = {item.text() for item in self.positive_list.selectedItems()}
        negative = {item.text() for item in self.negative_list.selectedItems()}
        
        if not positive or not negative:
            QMessageBox.warning(self, "Invalid Contrast", 
                               "Please select at least one category for each group.")
            return
        
        if positive & negative:
            QMessageBox.warning(self, "Invalid Contrast",
                               "Categories cannot be in both groups.")
            return
        
        self.contrast = ContrastDefinition(
            name=self.name_input.toPlainText(),
            positive_categories=positive,
            negative_categories=negative
        )
        
        super().accept()

# =============================================================================
# Display Worker - Simplified for computations only
# =============================================================================

class DisplayWorker(QObject):
    """Worker for heavy display computations in separate thread"""
    
    psth_display_ready = pyqtSignal(dict, np.ndarray, str)
    dprime_display_ready = pyqtSignal(np.ndarray, list, list)
    firing_rate_ready = pyqtSignal(object, list, dict, dict)  # NEW SIGNAL
    log_message = pyqtSignal(str)
    processing_time = pyqtSignal(str, float)
    
    def __init__(self, config, category_colors):
        super().__init__()
        self.config = config
        self.category_colors = category_colors
        
        # PSTH parameters
        self.current_channel = 0
        self.smoothing_params = {
            'bin_size_ms': 10,
            'method': 'boxcar'
        }
        
        # D-prime parameters
        self.dprime_threshold = 1.0
        self.dprime_analyzer = DPrimeAnalyzer(config)
        self.current_contrast = None
        self.dprime_time_window = (50, 250)

        self._psth_cache = {}  # Cache for processed PSTH data
        self._last_data_counts = None  # To detect when data changes
        self._cache_valid = False  # Flag to know when cache needs refresh

        self._is_running = True
    
    def set_channel(self, channel: int):
        """Update the channel to display"""
        self.current_channel = channel
        
    def set_smoothing(self, bin_size_ms: int, method: str):
        """Update smoothing parameters"""
        self.smoothing_params['bin_size_ms'] = bin_size_ms
        self.smoothing_params['method'] = method
    
    def set_dprime_threshold(self, threshold: float):
        """Update d-prime threshold"""
        self.dprime_threshold = threshold
    
    def set_contrast(self, contrast):
        """Set the contrast for d-prime calculation"""
        self.current_contrast = contrast
    
    def set_dprime_window(self, start_ms: float, end_ms: float):
        """Set time window for d-prime calculation"""
        self.dprime_time_window = (start_ms, end_ms)
    
    @pyqtSlot(dict)
    def process_psth_display(self, psth_data: dict):
        """Process PSTH data for display"""
        if not self._is_running or not psth_data or 'data' not in psth_data:
            return
        
        start_time = time.perf_counter()
        
        try:
            # Group by category
            category_psth = {}
            category_counts = {}
            categories = psth_data.get('categories', {})
            
            for stim_idx, psth in psth_data['data'].items():
                cat = categories.get(stim_idx, 'unknown')
                
                if cat not in category_psth:
                    category_psth[cat] = np.zeros_like(psth)
                    category_counts[cat] = 0
                
                category_psth[cat] += psth
                category_counts[cat] += 1
            
            # Average within categories
            for cat in category_psth:
                if category_counts[cat] > 0:
                    category_psth[cat] /= category_counts[cat]
            
            # Create time axis
            start_ms, end_ms = self.config.psth_window_ms
            n_samples = list(psth_data['data'].values())[0].shape[0]
            time_points = np.linspace(start_ms, end_ms, n_samples)
            
            # Apply smoothing
            ms_per_sample = (end_ms - start_ms) / n_samples
            bin_size_samples = max(1, int(self.smoothing_params['bin_size_ms'] / ms_per_sample))
            
            display_data = {}
            for cat, psth in category_psth.items():
                # Extract just the channel we need BEFORE smoothing
                channel_data = psth[:, self.current_channel]
                
                # Smooth only this single channel (1D instead of 2D)
                if bin_size_samples > 1:
                    if self.smoothing_params['method'] == 'boxcar':
                        kernel = np.ones(bin_size_samples) / bin_size_samples
                    elif self.smoothing_params['method'] == 'gaussian':
                        x = np.arange(bin_size_samples) - bin_size_samples // 2
                        kernel = np.exp(-x**2 / (2 * (bin_size_samples/4)**2))
                        kernel /= kernel.sum()
                    else:  # hamming
                        kernel = signal.windows.hamming(bin_size_samples)
                        kernel /= kernel.sum()
                    
                    smoothed_data = signal.convolve(channel_data, kernel, mode='same')
                else:
                    smoothed_data = channel_data
                
                display_data[cat] = {
                    'data': smoothed_data,
                    'count': category_counts[cat],
                    'color': self.category_colors.get(cat, 'w')
                }
            
            title = (f"Category-Averaged PSTH (Ch {self.current_channel}, "
                    f"Smoothing: {self.smoothing_params['bin_size_ms']}ms "
                    f"{self.smoothing_params['method']})")

            self.psth_display_ready.emit(display_data, time_points, title)
            
            elapsed = time.perf_counter() - start_time
            self.processing_time.emit('psth', elapsed * 1000)
            
        except Exception as e:
            pass 
            # self.log_message.emit(f"Worker error processing PSTH: {e}")
    
    @pyqtSlot(dict)
    def calculate_dprime(self, psth_data: dict):
        """Calculate d-prime in worker thread"""
        if not self._is_running or not psth_data or not self.current_contrast:
            return
        
        start_time = time.perf_counter()
        
        try:
            d_primes = self.dprime_analyzer.calculate_contrast(
                psth_data,
                self.current_contrast,
                self.dprime_time_window
            )
            
            channels = list(range(len(d_primes)))
            
            colors = []
            for d in d_primes:
                if abs(d) >= self.dprime_threshold:
                    intensity = min(1.0, abs(d) / 1.0)
                    if d > 0:
                        colors.append((255, int(255 * (1 - intensity)), int(255 * (1 - intensity)), 255))
                    else:
                        colors.append((int(255 * (1 - intensity)), int(255 * (1 - intensity)), 255, 255))
                else:
                    colors.append((128, 128, 128, 128))
            
            self.dprime_display_ready.emit(d_primes, channels, colors)
            
            elapsed = time.perf_counter() - start_time
            self.processing_time.emit('dprime', elapsed * 1000)
            
        except Exception as e:
            self.log_message.emit(f"Worker error calculating d-prime: {e}")

    @pyqtSlot(dict, tuple, bool)  # Added normalization flag
    def calculate_firing_rate_matrix(self, psth_data: dict, time_window: tuple, 
                                    normalize: bool = True):
        """Calculate mean firing rate matrix with per-channel normalization"""
        if not self._is_running or not psth_data or 'data' not in psth_data:
            return
        
        start_time = time.perf_counter()
        
        try:
            # Get time window indices
            start_ms, end_ms = self.config.psth_window_ms
            window_start, window_end = time_window
            
            # Get time points
            if len(psth_data['data']) == 0:
                return
                
            total_samples = list(psth_data['data'].values())[0].shape[0]
            time_points = np.linspace(start_ms, end_ms, total_samples)
            
            # Find indices for time window
            start_idx = np.argmin(np.abs(time_points - window_start))
            end_idx = np.argmin(np.abs(time_points - window_end))
            
            # Get categories
            categories = psth_data.get('categories', {})
            
            # Sort stimuli by category for better visualization
            stim_by_category = {}
            for stim_idx, psth in psth_data['data'].items():
                cat = categories.get(stim_idx, 'unknown')
                if cat not in stim_by_category:
                    stim_by_category[cat] = []
                stim_by_category[cat].append(stim_idx)
            
            # Sort categories and stimuli within categories
            sorted_categories = sorted(stim_by_category.keys())
            stim_order = []
            category_boundaries = {}
            current_row = 0
            
            for cat in sorted_categories:
                stim_indices = sorted(stim_by_category[cat])
                start_row = current_row
                stim_order.extend(stim_indices)
                current_row += len(stim_indices)
                category_boundaries[cat] = (start_row, current_row)
            
            # Create matrix
            n_stim = len(stim_order)
            n_chan = self.config.neural_channels
            firing_rate_matrix = np.zeros((n_stim, n_chan))
            
            # Fill matrix
            for row_idx, stim_idx in enumerate(stim_order):
                psth = psth_data['data'][stim_idx]
                # Average over time window
                mean_rate = psth[start_idx:end_idx].mean(axis=0)
                firing_rate_matrix[row_idx, :] = mean_rate
            
            # NORMALIZE per channel (z-score across stimuli)
            if normalize:
                # For each channel, normalize across all stimuli
                for ch in range(n_chan):
                    channel_rates = firing_rate_matrix[:, ch]
                    mean_rate = np.mean(channel_rates)
                    std_rate = np.std(channel_rates)
                    
                    # Avoid division by zero
                    if std_rate > 0:
                        firing_rate_matrix[:, ch] = (channel_rates - mean_rate) / std_rate
                    else:
                        firing_rate_matrix[:, ch] = 0
            
            # Create category info dict
            category_info = {stim_idx: categories.get(stim_idx, 'unknown') 
                           for stim_idx in stim_order}
            
            # Emit results with normalization flag
            self.firing_rate_ready.emit(
                firing_rate_matrix, 
                stim_order, 
                category_info,
                category_boundaries
            )
            
            elapsed = time.perf_counter() - start_time
            self.processing_time.emit('firing_rate', elapsed * 1000)
            
            # Log normalization status
            norm_status = "normalized" if normalize else "raw"
            self.log_message.emit(f"Calculated {norm_status} firing rate matrix in {elapsed*1000:.1f}ms")
            
        except Exception as e:
            self.log_message.emit(f"Error calculating firing rate matrix: {e}")

    def _smooth_psth(self, data: np.ndarray, bin_size: int, method: str = 'boxcar') -> np.ndarray:
        """Apply smoothing to PSTH data"""
        if bin_size <= 1:
            return data
        
        if method == 'boxcar':
            kernel = np.ones(bin_size) / bin_size
        elif method == 'gaussian':
            x = np.arange(bin_size) - bin_size // 2
            kernel = np.exp(-x**2 / (2 * (bin_size/4)**2))
            kernel /= kernel.sum()
        elif method == 'hamming':
            kernel = signal.windows.hamming(bin_size)
            kernel /= kernel.sum()
        else:
            kernel = np.ones(bin_size) / bin_size
        
        if len(data.shape) == 2:
            smoothed = signal.convolve2d(
                data, 
                kernel.reshape(-1, 1), 
                mode='same', 
                boundary='symm'
            )
        else:
            smoothed = signal.convolve(data, kernel, mode='same')
        
        return smoothed
    
    def stop(self):
        """Stop the worker"""
        self._is_running = False

# =============================================================================
# Main GUI Application - Simplified
# =============================================================================

class RealTimeNeuralGUI(QMainWindow):
    """Main GUI application with simple epoch saving control"""
    
    log_message_signal = pyqtSignal(str)
    
    def __init__(self, config: SystemConfig):
        super().__init__()
        self.config = config
        self.setWindowIcon(QIcon('icon.png')) 
        self.shutdown_manager = ShutdownManager()
        self.is_acquiring = False
        
        # Load stimulus info
        self.stim_info = None
        self.categories = []
        self.category_lookup = None
        self.load_stim_info()
     
        # Initialize components
        self.producers = {}
        self.consumers = {}
        self.queues = {}
        self.control_queue = mp.Queue()
        self.manager = Manager()
        self.psth_dict = self.manager.dict()
        self.psth_lock = mp.Lock()
        
        self.analysis_pipeline = None

        # Epoch saving control
        self.epoch_save_enabled = mp.Value('b', False)  # Shared boolean
        
        # Register manager with shutdown manager
        self.shutdown_manager.register_resource('mp_manager', self.manager)
        
        # Create monitor
        self.monitor = SystemMonitor(self.config)
        
        # Category colors
        self.category_colors = self._generate_category_colors()

        # Setup worker thread
        self.display_worker = None
        self.display_thread = None
        self._setup_display_worker()
        
        # Initialize PSTH dict
        self._initialize_psth_dict()
        
        # Store raw data for reprocessing
        self._last_raw_psth_data = None
        
        # Setup UI
        self.init_ui()
        self.connect_signals()
        
        # Register cleanup
        self.shutdown_manager.register_cleanup(self._cleanup_gui, priority=100)
        
        # Setup timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.setInterval(self.config.gui_update_interval_ms)
        self.shutdown_manager.register_timer(self.update_timer)
        
        # Show startup message
        mode_str = "Simulation" if self.config.simulation_mode else "Production"
        self.append_log(f"System initialized in {mode_str} mode")
        self.append_log("Ready to start acquisition")

    def _generate_category_colors(self):
        """Generate maximally contrasting colors for categories"""
        
        if not self.categories:
            return {}
        
        n = len(self.categories)
        colors = {}
        
        # Special cases for small numbers of categories with hand-picked optimal colors
        optimal_palettes = {
            1: ['#FF0000'],  # Red
            2: ['#FF0000', '#00FFFF'],  # Red, Cyan (complementary)
            3: ['#FF0000', '#00FF00', '#0000FF'],  # RGB primaries
            4: ['#FF0000', '#00FF00', '#0000FF', '#FFD700'],  # RGBY
            5: ['#FF0000', '#00FF00', '#0000FF', '#FFD700', '#FF00FF'],  # RGBY + Magenta
            6: ['#FF0000', '#00FF00', '#0000FF', '#FFD700', '#FF00FF', '#00FFFF'],  # + Cyan
        }
        
        if n in optimal_palettes:
            # Use pre-defined optimal colors for small sets
            for category, color in zip(self.categories, optimal_palettes[n]):
                colors[category] = color
        else:
            # For larger sets, use HSV color space distribution
            # Golden angle in turns (approx 137.5°) for optimal spacing
            golden_angle = 0.618033988749895
            
            # Start hue
            hue = 0.0
            
            for i, category in enumerate(self.categories):
                # For many categories, vary saturation and value too
                if n <= 12:
                    # Fewer categories: use full saturation and brightness
                    saturation = 1.0
                    value = 1.0
                else:
                    # Many categories: create more variation
                    # Cycle through 3 levels of saturation/value
                    level = i % 3
                    if level == 0:
                        saturation, value = 1.0, 0.9
                    elif level == 1:
                        saturation, value = 0.8, 1.0
                    else:
                        saturation, value = 0.9, 0.75
                
                # Convert HSV to RGB to hex
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255),
                    int(rgb[1] * 255),
                    int(rgb[2] * 255)
                )
                colors[category] = hex_color
                
                # Update hue using golden angle for maximum separation
                if n <= 12:
                    # Even distribution for small sets
                    hue = (i + 1) / n
                else:
                    # Golden angle distribution for large sets
                    hue = (hue + golden_angle) % 1.0
        
        return colors

    def _setup_display_worker(self):
        """Setup the display worker thread"""
        self.display_worker = DisplayWorker(self.config, self.category_colors)
        self.display_thread = QThread()
        
        self.display_worker.moveToThread(self.display_thread)
        
        self.display_worker.psth_display_ready.connect(self._update_psth_visualization)
        self.display_worker.dprime_display_ready.connect(self._update_dprime_visualization)
        self.display_worker.firing_rate_ready.connect(self._update_firing_rate_visualization)
        self.display_worker.log_message.connect(self.append_log)
        self.display_worker.processing_time.connect(self._monitor_processing_time)
        
        self.shutdown_manager.register_cleanup(self._cleanup_display_worker, priority=5)
        
        self.display_thread.start()
    
    def _cleanup_display_worker(self):
        """Clean shutdown of display worker"""
        if self.display_worker:
            self.display_worker.stop()
        if self.display_thread and self.display_thread.isRunning():
            self.display_thread.quit()
            self.display_thread.wait(1000)

    def _initialize_psth_dict(self):
        """Safely initialize PSTH dictionary"""
        try:
            with self.psth_lock:
                self.psth_dict.clear()
                self.psth_dict['data'] = {}
                self.psth_dict['counts'] = {}
                self.psth_dict['categories'] = {}
                self.psth_dict['last_update'] = time.time()
        except Exception as e:
            self.append_log(f"Error initializing PSTH dict: {e}")

    def load_stim_info(self):
        """Load stimulus information with validation"""
        stim_path = Path(self.config.image_info_path)
        
        if not stim_path.exists():
            QMessageBox.warning(
                self,
                "Stimulus Info Missing",
                f"Stimulus info file '{self.config.image_info_path}' not found.\n\n"
                "Category-specific analyses will use default categories.\n"
                "To use custom categories, create a TSV file with columns:\n"
                "Idx (or Index), FileName, Category (or FOB)"
            )
            
            self.categories = ['F', 'O', 'B', 'C']
            self.append_log("WARNING: Using default categories (F, O, B, C)")
            return
        
        try:
            self.stim_info = pd.read_csv(stim_path, sep='\t')
            
            required_cols = {'Idx', 'Index', 'FileName'}
            category_cols = {'Category', 'FOB'}
            
            has_index = bool(required_cols & set(self.stim_info.columns))
            has_category = bool(category_cols & set(self.stim_info.columns))
            
            if not has_index:
                raise ValueError("Missing index column (Idx or Index)")
            
            if not has_category:
                raise ValueError("Missing category column (Category or FOB)")
            
            cat_col = 'FOB' if 'FOB' in self.stim_info.columns else 'Category'
            self.categories = self.stim_info[cat_col].unique().tolist()
            self.category_lookup = {}
            for _, row in self.stim_info.iterrows():
                idx = int(row.get('Idx', row.get('Index', -1)))
                category = str(row.get('FOB', row.get('Category', 'unknown')))
                self.category_lookup[idx] = category
            self.append_log(f"Loaded {len(self.stim_info)} stimuli with {len(self.categories)} categories")
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Stimulus Info Error",
                f"Error loading '{self.config.image_info_path}':\n{str(e)}\n\n"
                "Using default categories."
            )
            self.categories = ['F', 'O', 'B', 'C']
            self.stim_info = None
            self.category_lookup = None
            self.append_log(f"ERROR loading stim_info: {e}")
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Real-time Neural Analysis System v5")
        self.setGeometry(100, 100, 1800, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_widget = self.create_left_panel()
        splitter.addWidget(left_widget)
        
        right_widget = self.create_right_panel()
        splitter.addWidget(right_widget)
        
        splitter.setSizes([1400, 400])
        
        main_layout.addWidget(splitter)
    
    def create_left_panel(self) -> QWidget:
        """Create left panel with controls and visualizations"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        self.tab_widget = QTabWidget()
        
        psth_widget = self.create_psth_widget()
        self.tab_widget.addTab(psth_widget, "PSTH Viewer")
        
        dprime_widget = self.create_dprime_widget()
        self.tab_widget.addTab(dprime_widget, "D-prime Analysis")
        
        firing_rate_widget = self.create_firing_rate_widget()
        self.tab_widget.addTab(firing_rate_widget, "Firing Rate Matrix")

        layout.addWidget(self.tab_widget)
        
        log_widget = self.create_log_widget()
        layout.addWidget(log_widget)
        
        layout.setStretch(0, 1)
        layout.setStretch(1, 6)
        layout.setStretch(2, 2)
        
        widget.setLayout(layout)
        return widget
    
    def create_control_panel(self) -> QWidget:
        """Create main control panel with epoch save path display"""
        """Create control panel with epoch saving button"""
        panel = QGroupBox("System Control")
        layout = QVBoxLayout()
        
        # Row 1: Mode and basic controls
        row1 = QHBoxLayout()
        
        self.mode_checkbox = QCheckBox("Simulation Mode")
        self.mode_checkbox.setChecked(self.config.simulation_mode)
        self.mode_checkbox.stateChanged.connect(self.on_mode_changed)
        
        self.start_btn = QPushButton("Start Acquisition")
        self.start_btn.clicked.connect(self.start_acquisition)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        
        self.stop_btn = QPushButton("Stop Acquisition")
        self.stop_btn.clicked.connect(self.stop_acquisition)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        
        self.clear_btn = QPushButton("Clear Buffers")
        self.clear_btn.clicked.connect(self.clear_buffers)
        self.clear_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        
        # Add epoch record button
        self.epoch_record_btn = QPushButton("Enable Epoch Record")
        self.epoch_record_btn.clicked.connect(self.toggle_epoch_recording)
        self.epoch_record_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
        
        row1.addWidget(self.mode_checkbox)
        row1.addWidget(self.start_btn)
        row1.addWidget(self.stop_btn)
        row1.addWidget(self.clear_btn)
        row1.addWidget(self.epoch_record_btn)
        row1.addStretch()
        row1.addWidget(self.status_label)
        
        # Row 2: Simulation parameters
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Simulation Parameters:"))
        
        row2.addWidget(QLabel("Event Rate (Hz):"))
        self.event_rate_spin = QDoubleSpinBox()
        self.event_rate_spin.setRange(0.1, 10.0)
        self.event_rate_spin.setSingleStep(0.5)
        self.event_rate_spin.setValue(self.config.sim_event_rate_hz)
        self.event_rate_spin.valueChanged.connect(self.on_event_rate_changed)
        row2.addWidget(self.event_rate_spin)
        
        row2.addWidget(QLabel("Sync Rate (Hz):"))
        self.sync_rate_spin = QDoubleSpinBox()
        self.sync_rate_spin.setRange(0.1, 10.0)
        self.sync_rate_spin.setSingleStep(0.5)
        self.sync_rate_spin.setValue(self.config.sim_sync_rate_hz)
        self.sync_rate_spin.valueChanged.connect(self.on_sync_rate_changed)
        row2.addWidget(self.sync_rate_spin)
        
        # Add epoch status label
        self.epoch_status_label = QLabel("Epochs: 0 saved")
        self.epoch_status_label.setStyleSheet("QLabel { color: blue; }")
        row2.addWidget(self.epoch_status_label)
        
        row2.addStretch()
        
        # Row 3: Channel selector
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Display Channel:"))
        
        self.channel_selector = QSpinBox()
        self.channel_selector.setRange(0, self.config.neural_channels - 1)
        self.channel_selector.setValue(0)
        self.channel_selector.valueChanged.connect(self.on_channel_changed)
        row3.addWidget(self.channel_selector)
        row3.addWidget(QLabel("|"))  # Separator
        self.epoch_save_path_label = QLabel("Save Path: Not Recording")
        self.epoch_save_path_label.setFont(QFont("Courier", 9))
        self.epoch_save_path_label.setStyleSheet("QLabel { color: #666; }")
        self.epoch_save_path_label.setToolTip("Current epoch save directory")
        row3.addWidget(self.epoch_save_path_label)
        
        self.open_save_folder_btn = QPushButton("📁")
        self.open_save_folder_btn.setToolTip("Open save folder")
        self.open_save_folder_btn.setMaximumWidth(30)
        self.open_save_folder_btn.clicked.connect(self.open_save_folder)
        self.open_save_folder_btn.setEnabled(False)
        row3.addWidget(self.open_save_folder_btn)
        row3.addStretch()
        
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        
        panel.setLayout(layout)
        return panel  
    
    def create_psth_widget(self) -> QWidget:
        """Create PSTH visualization widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Smoothing Bin Size (ms):"))
        
        self.smoothing_bin_spin = QSpinBox()
        self.smoothing_bin_spin.setRange(1, 100)
        self.smoothing_bin_spin.setSingleStep(5)
        self.smoothing_bin_spin.setValue(10)
        self.smoothing_bin_spin.valueChanged.connect(self.on_smoothing_changed)
        control_layout.addWidget(self.smoothing_bin_spin)
        
        control_layout.addWidget(QLabel("Smoothing Type:"))
        
        self.smoothing_type_combo = QComboBox()
        self.smoothing_type_combo.addItems(['boxcar', 'gaussian', 'hamming'])
        self.smoothing_type_combo.setCurrentText('boxcar')
        self.smoothing_type_combo.currentTextChanged.connect(self.on_smoothing_changed)
        control_layout.addWidget(self.smoothing_type_combo)
        
        control_layout.addStretch()

        self.psth_plot = pg.PlotWidget(title="Category-Averaged PSTH")
        self.psth_plot.setLabel('left', 'Firing Rate', units='Hz')
        self.psth_plot.setLabel('bottom', 'Time', units='ms')
        self.psth_plot.addLegend()
        
        self.psth_curves = {}
        
        layout.addLayout(control_layout)
        layout.addWidget(self.psth_plot)
        widget.setLayout(layout)
        return widget
    
    def create_dprime_widget(self) -> QWidget:
        """Create d-prime analysis widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        control_layout = QHBoxLayout()
        
        self.contrast_label = QLabel("No contrast defined")
        control_layout.addWidget(self.contrast_label)
        
        self.define_contrast_btn = QPushButton("Define Contrast")
        self.define_contrast_btn.clicked.connect(self.define_contrast)
        control_layout.addWidget(self.define_contrast_btn)
        
        control_layout.addWidget(QLabel("Time Window (ms):"))
        
        self.time_start_selector = QSpinBox()
        self.time_start_selector.setRange(-1000, 1000)
        self.time_start_selector.setValue(50)
        control_layout.addWidget(self.time_start_selector)
        
        control_layout.addWidget(QLabel("to"))
        
        self.time_end_selector = QSpinBox()
        self.time_end_selector.setRange(-1000, 2000)
        self.time_end_selector.setValue(250)
        control_layout.addWidget(self.time_end_selector)
        
        self.calculate_dprime_btn = QPushButton("Calculate D-prime")
        self.calculate_dprime_btn.clicked.connect(self.calculate_dprime)
        control_layout.addWidget(self.calculate_dprime_btn)
        
        control_layout.addStretch()
        
        threshold_layout = QHBoxLayout()
        
        threshold_layout.addWidget(QLabel("D-prime Threshold:"))
        
        self.dprime_threshold_spin = QDoubleSpinBox()
        self.dprime_threshold_spin.setRange(0.0, 5.0)
        self.dprime_threshold_spin.setSingleStep(0.1)
        self.dprime_threshold_spin.setValue(1.0)
        self.dprime_threshold_spin.setDecimals(2)
        self.dprime_threshold_spin.valueChanged.connect(self.on_dprime_threshold_changed)
        threshold_layout.addWidget(self.dprime_threshold_spin)
        
        threshold_layout.addWidget(QLabel("Show Grid:"))
        
        self.dprime_grid_checkbox = QCheckBox()
        self.dprime_grid_checkbox.setChecked(True)
        self.dprime_grid_checkbox.stateChanged.connect(self.on_dprime_grid_changed)
        threshold_layout.addWidget(self.dprime_grid_checkbox)
        
        threshold_layout.addStretch()

        self.dprime_plot = pg.PlotWidget(title="D-prime Across Channels")
        self.dprime_plot.setLabel('left', "D-prime")
        self.dprime_plot.setLabel('bottom', 'Channel')
        
        self.dprime_bar = pg.BarGraphItem(x=[], height=[], width=0.8, brush='b')
        self.dprime_plot.addItem(self.dprime_bar)
        
        self.dprime_threshold_line = pg.InfiniteLine(
            pos=1.0, 
            angle=0, 
            pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            label='Threshold',
            labelOpts={'position': 0.95, 'color': 'r'}
        )
        self.dprime_plot.addItem(self.dprime_threshold_line)
        
        self.dprime_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Setup hover functionality
        self.setup_dprime_hover()
        self.dprime_summary_label = QLabel("Top channels: calculating...")
        self.dprime_summary_label.setFont(QFont("Courier", 10))
    
        layout.addLayout(control_layout)
        layout.addLayout(threshold_layout)
        layout.addWidget(self.dprime_plot)
        layout.addWidget(self.dprime_summary_label)
        
        widget.setLayout(layout)
        return widget

    def create_firing_rate_widget(self) -> QWidget:
        """Create mean firing rate matrix visualization widget with normalization"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Control panel
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("Time Window (ms):"))
        
        self.fr_time_start = QSpinBox()
        self.fr_time_start.setRange(-1000, 1000)
        self.fr_time_start.setValue(50)
        control_layout.addWidget(self.fr_time_start)
        
        control_layout.addWidget(QLabel("to"))
        
        self.fr_time_end = QSpinBox()
        self.fr_time_end.setRange(-1000, 2000)
        self.fr_time_end.setValue(250)
        control_layout.addWidget(self.fr_time_end)
        
        # ADD NORMALIZATION CHECKBOX
        self.fr_normalize_checkbox = QCheckBox("Normalize per channel")
        self.fr_normalize_checkbox.setChecked(True)
        self.fr_normalize_checkbox.setToolTip("Z-score normalize each channel across stimuli")
        control_layout.addWidget(self.fr_normalize_checkbox)
        self.fr_normalize_checkbox.stateChanged.connect(self.on_normalize_changed)
        
        self.fr_calculate_btn = QPushButton("Calculate Firing Rates")
        self.fr_calculate_btn.clicked.connect(self.calculate_firing_rates)
        control_layout.addWidget(self.fr_calculate_btn)
        
        control_layout.addWidget(QLabel("Color Range:"))
        
        control_layout.addWidget(QLabel("Min:"))
        self.fr_vmin_spin = QDoubleSpinBox()
        self.fr_vmin_spin.setRange(-5, 5)  # Changed range for normalized values
        self.fr_vmin_spin.setValue(-2)  # Default for z-scores
        self.fr_vmin_spin.setSingleStep(0.5)
        self.fr_vmin_spin.valueChanged.connect(self.update_firing_rate_colormap)
        control_layout.addWidget(self.fr_vmin_spin)
        
        control_layout.addWidget(QLabel("Max:"))
        self.fr_vmax_spin = QDoubleSpinBox()
        self.fr_vmax_spin.setRange(-5, 5)  # Changed range for normalized values
        self.fr_vmax_spin.setValue(2)  # Default for z-scores
        self.fr_vmax_spin.setSingleStep(0.5)
        self.fr_vmax_spin.valueChanged.connect(self.update_firing_rate_colormap)
        control_layout.addWidget(self.fr_vmax_spin)
        
        self.fr_auto_scale_btn = QPushButton("Auto Scale")
        self.fr_auto_scale_btn.clicked.connect(self.auto_scale_firing_rate)
        control_layout.addWidget(self.fr_auto_scale_btn)
        
        control_layout.addStretch()
        
        # Create plot widget with image item
        self.fr_plot = pg.PlotWidget(title="Mean Firing Rate Matrix (Normalized)")
        self.fr_plot.setLabel('left', 'Stimulus Index')
        self.fr_plot.setLabel('bottom', 'Channel')
        
        # Create image item for the heatmap
        self.fr_image = pg.ImageItem()
        self.fr_plot.addItem(self.fr_image)
        
        # Setup colormap (RdBu_r)
        self.fr_colormap = self._create_rdbu_colormap()
        self.fr_image.setLookupTable(self.fr_colormap)
        
        # Store for category boundaries
        self.category_lines = []
        self.fr_matrix_data = None
        self.fr_stim_order = []
        self.fr_category_info = {}
        self.fr_is_normalized = True  # Track normalization state
        
        # Add hover label
        self.fr_hover_label = QLabel("Hover over matrix to see values")
        self.fr_hover_label.setFont(QFont("Courier", 10))
        
        # Setup hover proxy
        self.fr_plot.scene().sigMouseMoved.connect(self.on_fr_mouse_moved)
        
        layout.addLayout(control_layout)
        layout.addWidget(self.fr_plot)
        layout.addWidget(self.fr_hover_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_log_widget(self) -> QWidget:
        """Create system log widget"""
        widget = QGroupBox("System Log")
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        
        layout.addWidget(self.log_text)
        widget.setLayout(layout)
        return widget
    
    def create_right_panel(self) -> QWidget:
        """Create right panel with monitoring"""
        panel = QGroupBox("System Monitor")
        layout = QVBoxLayout()
        
        grid = QGridLayout()
        
        self.queue_labels = {}
        self.queue_bars = {}
        
        grid.addWidget(QLabel("Queue Status:"), 0, 0, 1, 2)
        
        for i, name in enumerate(['ttl', 'udp', 'sync', 'neural']):
            label = QLabel(f"{name.upper()} Queue: 0")
            bar = QProgressBar()
            bar.setMaximum(100)
            
            self.queue_labels[name] = label
            self.queue_bars[name] = bar
            
            grid.addWidget(label, i+1, 0)
            grid.addWidget(bar, i+1, 1)
        
        grid.addWidget(QLabel("Event Rates:"), 5, 0, 1, 2)
        
        self.rate_labels = {}
        for i, name in enumerate(['ttl', 'udp', 'sync', 'neural']):
            label = QLabel(f"{name.upper()} Rate: 0.0 Hz")
            self.rate_labels[name] = label
            grid.addWidget(label, i+6, 0, 1, 2)
        
        grid.addWidget(QLabel("System Resources:"), 10, 0, 1, 2)
        
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_bar = QProgressBar()
        grid.addWidget(self.cpu_label, 11, 0)
        grid.addWidget(self.cpu_bar, 11, 1)
        
        self.memory_label = QLabel("Memory: 0%")
        self.memory_bar = QProgressBar()
        grid.addWidget(self.memory_label, 12, 0)
        grid.addWidget(self.memory_bar, 12, 1)
        
        self.data_rate_label = QLabel("Data Rate: 0.0 MB/s")
        grid.addWidget(self.data_rate_label, 13, 0, 1, 2)
        
        layout.addLayout(grid)
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel

    def _create_rdbu_colormap(self):
        """Create RdBu_r colormap for pyqtgraph"""
        # Get matplotlib RdBu_r colormap
        cmap = mpl.colormaps['RdBu_r']
        
        # Sample the colormap
        n_colors = 256
        colors = []
        for i in range(n_colors):
            rgba = cmap(i / (n_colors - 1))
            # Convert to 0-255 range for pyqtgraph
            colors.append([int(rgba[j] * 255) for j in range(4)])
        
        return np.array(colors, dtype=np.ubyte)

    def calculate_firing_rates(self):
        """Calculate mean firing rates for matrix display with normalization"""
        if self.display_worker and self._last_raw_psth_data:
            time_window = (self.fr_time_start.value(), self.fr_time_end.value())
            normalize = self.fr_normalize_checkbox.isChecked()
            
            QTimer.singleShot(0, lambda: self.display_worker.calculate_firing_rate_matrix(
                self._last_raw_psth_data, 
                time_window,
                normalize  # Pass normalization flag
            ))
            
            # Update plot title based on normalization
            title = "Mean Firing Rate Matrix (Normalized)" if normalize else "Mean Firing Rate Matrix (Raw)"
            self.fr_plot.setTitle(title)
            
            # Adjust color scale defaults based on normalization
            if normalize and not hasattr(self, '_fr_manual_scaled'):
                self.fr_vmin_spin.setValue(-2)
                self.fr_vmax_spin.setValue(2)
            elif not normalize and not hasattr(self, '_fr_manual_scaled'):
                self.fr_vmin_spin.setValue(0)
                self.fr_vmax_spin.setValue(100)
                
        else:
            self.append_log("No PSTH data available for firing rate calculation")

    def update_firing_rate_colormap(self):
        """Update the color scale of the firing rate matrix"""
        if hasattr(self, 'fr_matrix_data') and self.fr_matrix_data is not None:
            vmin = self.fr_vmin_spin.value()
            vmax = self.fr_vmax_spin.value()
            
            # Set levels for the image item
            self.fr_image.setLevels([vmin, vmax])

    def update_epoch_save_path_display(self, save_path: str = None):
        """Update the epoch save path label"""
        if save_path:
            self.current_epoch_save_path = save_path
            display_path = save_path
            if len(display_path) > 100:
                # Truncate long paths
                display_path = "..." + display_path[-97:]
            
            self.epoch_save_path_label.setText(f"Save Path: {display_path}")
            self.epoch_save_path_label.setStyleSheet("QLabel { color: #2e7d32; font-weight: bold; }")
            self.epoch_save_path_label.setToolTip(f"Full path: {save_path}")
            self.open_save_folder_btn.setEnabled(True)
        else:
            self.current_epoch_save_path = None
            self.epoch_save_path_label.setText("Save Path: Not Recording")
            self.epoch_save_path_label.setStyleSheet("QLabel { color: #666; }")
            self.epoch_save_path_label.setToolTip("Current epoch save directory")
            self.open_save_folder_btn.setEnabled(False)

    def auto_scale_firing_rate(self):
        """Auto-scale the firing rate color range"""
        if hasattr(self, 'fr_matrix_data') and self.fr_matrix_data is not None:
            vmin = np.percentile(self.fr_matrix_data, 5)
            vmax = np.percentile(self.fr_matrix_data, 95)
            
            self.fr_vmin_spin.setValue(vmin)
            self.fr_vmax_spin.setValue(vmax)

    def connect_signals(self):
        """Connect all signals"""
        self.log_message_signal.connect(self.append_log)
        self.monitor.stats_updated.connect(self.update_monitor_display)

    def on_mode_changed(self, state):
        """Handle mode change"""
        self.config.simulation_mode = self.mode_checkbox.isChecked()
        mode_str = "Simulation" if self.config.simulation_mode else "Production"
        self.append_log(f"Mode changed to: {mode_str}")
        
        self.event_rate_spin.setEnabled(self.config.simulation_mode)
        self.sync_rate_spin.setEnabled(self.config.simulation_mode)

    def on_epoch_save_toggled(self, state):
        """Handle epoch save checkbox toggle"""
        enabled = state == 2  # Qt.Checked
        
        if self.analysis_pipeline:
            self.analysis_pipeline.epoch_save_enabled.value = enabled
            
            if enabled:
                # Create epoch manager if needed and update path display
                if self.analysis_pipeline.epoch_manager is None:
                    from pathlib import Path
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    session_dir = Path(self.config.save_path) / f"epochs_{timestamp}"
                    
                    # Display the path
                    display_path = str(session_dir)
                    if len(display_path) > 60:
                        # Truncate long paths
                        display_path = "..." + display_path[-57:]
                    
                    self.epoch_save_path_label.setText(f"Save Path: {display_path}")
                    self.epoch_save_path_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
                    self.epoch_save_path_label.setToolTip(f"Full path: {session_dir}")
                    self.open_save_folder_btn.setEnabled(True)
                    
                    self.append_log(f"Epoch saving enabled: {session_dir}")
                else:
                    # Already have a manager, get its path
                    save_path = self.analysis_pipeline.epoch_manager.get_save_path()
                    display_path = save_path
                    if len(display_path) > 60:
                        display_path = "..." + display_path[-57:]
                    
                    self.epoch_save_path_label.setText(f"Save Path: {display_path}")
                    self.epoch_save_path_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
                    self.epoch_save_path_label.setToolTip(f"Full path: {save_path}")
                    self.open_save_folder_btn.setEnabled(True)
            else:
                self.epoch_save_path_label.setText("Save Path: Not Recording")
                self.epoch_save_path_label.setStyleSheet("color: #666;")
                self.epoch_save_path_label.setToolTip("Current epoch save directory")
                self.open_save_folder_btn.setEnabled(False)
                self.append_log("Epoch saving disabled")
        else:
            self.epoch_save_checkbox.setChecked(False)
            self.append_log("Cannot toggle epoch saving: acquisition not started")

    def on_event_rate_changed(self, value):
        """Handle event rate change"""
        self.config.sim_event_rate_hz = value
        if 'sglx' in self.producers:
            self.producers['sglx'].set_event_rate(value)
        if 'udp' in self.producers:
            self.producers['udp'].set_event_rate(value)
    
    def on_sync_rate_changed(self, value):
        """Handle sync rate change"""
        self.config.sim_sync_rate_hz = value
        if 'sglx' in self.producers:
            self.producers['sglx'].set_sync_rate(value)
    
    def on_channel_changed(self, channel: int):
        """Handle channel selection change"""
        if self.display_worker:
            self.display_worker.set_channel(channel)
            for curve in list(self.psth_curves.values()):
                self.psth_plot.removeItem(curve)
            self.psth_curves.clear()
            if self._last_raw_psth_data:
                QTimer.singleShot(0, lambda: self.display_worker.process_psth_display(self._last_raw_psth_data))
    
    def on_smoothing_changed(self):
        """Handle smoothing parameter change"""
        if self.display_worker:
            bin_size_ms = self.smoothing_bin_spin.value()
            method = self.smoothing_type_combo.currentText()
            self.display_worker.set_smoothing(bin_size_ms, method)
            if self._last_raw_psth_data:
                QTimer.singleShot(0, lambda: self.display_worker.process_psth_display(self._last_raw_psth_data))
    
    def on_dprime_threshold_changed(self, value: float):
        """Handle d-prime threshold change"""
        self.dprime_threshold_line.setPos(value)
        
        if not hasattr(self, 'dprime_neg_threshold_line'):
            self.dprime_neg_threshold_line = pg.InfiniteLine(
                pos=-value,
                angle=0,
                pen=pg.mkPen('r', width=1, style=pg.QtCore.Qt.PenStyle.DotLine),
                label='-Threshold',
                labelOpts={'position': 0.05, 'color': 'r'}
            )
            self.dprime_plot.addItem(self.dprime_neg_threshold_line)
        else:
            self.dprime_neg_threshold_line.setPos(-value)
        
        if self.display_worker:
            self.display_worker.set_dprime_threshold(value)
            if self._last_raw_psth_data and self.display_worker.current_contrast:
                QTimer.singleShot(0, lambda: self.display_worker.calculate_dprime(self._last_raw_psth_data))
    
    def on_fr_mouse_moved(self, pos):
        """Handle mouse movement over firing rate matrix"""
        if not hasattr(self, 'fr_matrix_data') or self.fr_matrix_data is None:
            return
        
        # Convert scene position to view position
        mouse_point = self.fr_plot.plotItem.vb.mapSceneToView(pos)
        x = int(mouse_point.x())
        y = int(mouse_point.y())
        
        # Check bounds
        n_stim, n_chan = self.fr_matrix_data.shape
        if 0 <= x < n_chan and 0 <= y < n_stim:
            value = self.fr_matrix_data[y, x]
            
            # Get stimulus info
            if y < len(self.fr_stim_order):
                stim_idx = self.fr_stim_order[y]
                category = self.fr_category_info.get(stim_idx, 'unknown')
                
                # Format value based on normalization
                if hasattr(self, 'fr_is_normalized') and self.fr_is_normalized:
                    value_str = f"z-score: {value:.2f}"
                else:
                    value_str = f"Firing Rate: {value:.1f} Hz"
                
                self.fr_hover_label.setText(
                    f"Channel: {x:3d} | Stim: {stim_idx:3d} ({category}) | {value_str}"
                )
            else:
                self.fr_hover_label.setText(
                    f"Channel: {x:3d} | Stim Row: {y:3d} | Value: {value:.2f}"
                )

    def open_save_folder(self):
        """Open the current save folder in system file explorer"""
        if hasattr(self, 'current_epoch_save_path') and self.current_epoch_save_path:
            import platform
            import subprocess
            from pathlib import Path
            
            save_path = Path(self.current_epoch_save_path)
            if save_path.exists():
                system = platform.system()
                try:
                    if system == "Windows":
                        subprocess.run(["explorer", str(save_path)])
                    elif system == "Darwin":  # macOS
                        subprocess.run(["open", str(save_path)])
                    else:  # Linux and others
                        subprocess.run(["xdg-open", str(save_path)])
                    
                    self.append_log(f"Opened save folder: {save_path}")
                except Exception as e:
                    self.append_log(f"Failed to open folder: {e}")
            else:
                self.append_log(f"Save folder does not exist yet: {save_path}")
        else:
            self.append_log("No save folder path available")

    def _update_epoch_info(self, epoch_count: int):
        """Update epoch counter display with save count"""
        self.epochs_received = epoch_count
        self.epochs_label.setText(f"Epochs: {epoch_count}")
        
        # Update save count if saving is enabled
        if (self.analysis_pipeline and 
            self.analysis_pipeline.epoch_manager and 
            self.epoch_save_checkbox.isChecked()):
            
            saved_count = self.analysis_pipeline.epoch_manager.epochs_saved
            self.epochs_label.setText(f"Epochs: {epoch_count} (Saved: {saved_count})")

    @pyqtSlot(object, list, dict, dict)
    def _update_firing_rate_visualization(self, matrix_data: np.ndarray, 
                                        stim_order: list, 
                                        category_info: dict,
                                        category_boundaries: dict):
        """Update firing rate matrix visualization"""
        self.fr_matrix_data = matrix_data
        self.fr_stim_order = stim_order
        self.fr_category_info = category_info
        self.fr_is_normalized = self.fr_normalize_checkbox.isChecked()
        
        # Set the image data
        # Transpose to match expected orientation (stimuli on Y, channels on X)
        self.fr_image.setImage(matrix_data.T)
        
        # Clear old category lines
        for line in self.category_lines:
            self.fr_plot.removeItem(line)
        self.category_lines.clear()
        
        # Add category boundary lines
        for category, (start, end) in category_boundaries.items():
            if start > 0:  # Don't draw line at the very top
                line = pg.InfiniteLine(
                    pos=start - 0.5,
                    angle=0,
                    pen=pg.mkPen('black', width=1, style=pg.QtCore.Qt.PenStyle.DotLine)
                )
                self.fr_plot.addItem(line)
                self.category_lines.append(line)
        
        # Update axis labels with category info
        n_stim, n_chan = matrix_data.shape
        
        # Set axis ranges
        self.fr_plot.setXRange(0, n_chan, padding=0)
        self.fr_plot.setYRange(0, n_stim, padding=0)
        
        # Update title
        norm_str = "Normalized " if self.fr_is_normalized else ""
        self.fr_plot.setTitle(
            f"{norm_str}Mean Firing Rate Matrix ({n_stim} stimuli × {n_chan} channels)"
        )
        
        # Auto-scale if first time
        if not hasattr(self, '_fr_autoscaled'):
            self.auto_scale_firing_rate()
            self._fr_autoscaled = True

    @pyqtSlot(int)
    def on_dprime_grid_changed(self, state: int):
        """Toggle grid visibility on d-prime plot"""
        show_grid = self.dprime_grid_checkbox.isChecked()
        self.dprime_plot.showGrid(x=show_grid, y=show_grid, alpha=0.3)
    
    def toggle_epoch_recording(self):
        """Toggle epoch recording on/off"""
        if self.epoch_save_enabled.value:
            # Currently enabled, so disable it
            self.epoch_save_enabled.value = False
            self.epoch_record_btn.setText("Enable Epoch Record")
            self.epoch_record_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
            self.update_epoch_save_path_display(None)
            self.append_log("Epoch recording DISABLED")
        else:
            # Currently disabled, so enable it
            self.epoch_save_enabled.value = True
            self.epoch_record_btn.setText("Disable Epoch Record")
            self.epoch_record_btn.setStyleSheet("QPushButton { background-color: #673AB7; color: white; }")
            
            # Show waiting message - path will be updated when first epoch is processed
            self.epoch_save_path_label.setText("Save Path: Waiting for first epoch...")
            self.epoch_save_path_label.setStyleSheet("QLabel { color: #1976d2; font-style: italic; }")
            self.open_save_folder_btn.setEnabled(False)
            
            self.append_log("Epoch recording ENABLED - path will be created on first epoch")
    
    def define_contrast(self):
        """Open contrast definition dialog"""
        dialog = ContrastDialog(self.categories, self)
        if dialog.exec():
            contrast = dialog.contrast
            
            self.contrast_label.setText(
                f"A: {', '.join(contrast.positive_categories)} | " 
                f"B: {', '.join(contrast.negative_categories)}"
            )
            
            if self.display_worker:
                self.display_worker.set_contrast(contrast)
                if self._last_raw_psth_data:
                    QTimer.singleShot(0, lambda: self.display_worker.calculate_dprime(self._last_raw_psth_data))
    
    def calculate_dprime(self):
        """Manual d-prime calculation trigger"""
        if self.display_worker:
            self.display_worker.set_dprime_window(
                self.time_start_selector.value(),
                self.time_end_selector.value()
            )
            
            if self._last_raw_psth_data:
                QTimer.singleShot(0, lambda: self.display_worker.calculate_dprime(self._last_raw_psth_data))
            else:
                self.append_log("No PSTH data available for d-prime calculation")

    # Enhanced D-prime methods
    def setup_dprime_hover(self):
        """Setup hover functionality for d-prime plot"""
        # Create a text item for displaying values
        self.dprime_hover_text = pg.TextItem(
            text="",
            color='w',
            fill=(0, 0, 0, 128),
            anchor=(0, 1)
        )
        self.dprime_plot.addItem(self.dprime_hover_text)
        self.dprime_hover_text.hide()
        
        # Connect mouse move event
        self.dprime_plot.scene().sigMouseMoved.connect(self.on_dprime_mouse_moved)
        
        # Store current d-prime values for hover
        self.current_dprime_values = None

    def on_dprime_mouse_moved(self, pos):
        """Handle mouse movement over d-prime plot"""
        if not hasattr(self, 'current_dprime_values') or self.current_dprime_values is None:
            if hasattr(self, 'dprime_hover_text'):
                self.dprime_hover_text.hide()
            return
        
        # Convert scene position to view position
        mouse_point = self.dprime_plot.plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x()
        
        # Find nearest channel
        channel = int(round(x))
        
        if 0 <= channel < len(self.current_dprime_values):
            dprime = self.current_dprime_values[channel]
            
            # Update hover text
            self.dprime_hover_text.setText(
                f"Ch {channel}: {dprime:.3f}"
            )
            self.dprime_hover_text.setPos(channel, dprime)
            self.dprime_hover_text.show()
        else:
            self.dprime_hover_text.hide()

    def on_normalize_changed(self):
        """Handle normalization checkbox state change"""
        normalize = self.fr_normalize_checkbox.isChecked()
        
        # Update spin box ranges based on normalization
        if normalize:
            self.fr_vmin_spin.setRange(-5, 5)
            self.fr_vmax_spin.setRange(-5, 5)
            if not hasattr(self, '_fr_manual_scaled'):
                self.fr_vmin_spin.setValue(-2)
                self.fr_vmax_spin.setValue(2)
        else:
            self.fr_vmin_spin.setRange(-100, 500)
            self.fr_vmax_spin.setRange(0, 500)
            if not hasattr(self, '_fr_manual_scaled'):
                self.fr_vmin_spin.setValue(0)
                self.fr_vmax_spin.setValue(100)
        
        # Recalculate if we have data
        if hasattr(self, '_last_raw_psth_data') and self._last_raw_psth_data:
            self.calculate_firing_rates()

    def update_dprime_summary(self, d_primes: np.ndarray):
        """Update summary of high d-prime channels"""
        if d_primes is None or len(d_primes) == 0:
            return
        
        threshold = self.dprime_threshold_spin.value()
        
        # Find channels above threshold
        high_channels = np.where(np.abs(d_primes) >= threshold)[0]
        
        if len(high_channels) > 0:
            # Sort by absolute d-prime value
            sorted_idx = high_channels[np.argsort(np.abs(d_primes[high_channels]))[::-1]]
            
            # Create summary text
            top_5 = sorted_idx[:5]
            summary_parts = []
            for ch in top_5:
                summary_parts.append(f"Ch{ch}:{d_primes[ch]:.2f}")
            
            summary = f"Top channels (|d'|≥{threshold:.1f}): " + ", ".join(summary_parts)
            
            if len(high_channels) > 5:
                summary += f" ... ({len(high_channels)} total)"
        else:
            summary = f"No channels with |d'| ≥ {threshold:.1f}"
        
        self.dprime_summary_label.setText(summary)

    def _create_producers(self):
        """Create producer threads"""
        if self.config.simulation_mode:
            self.producers['sglx'] = SimulatedSpikeGLXProducer(
                self.config, self.queues['ttl'], self.queues['neural']
            )
            self.producers['udp'] = SimulatedUDPProducer(
                self.config, self.queues['udp'], self.stim_info
            )
        else:
            self.producers['sglx'] = RealSpikeGLXProducer(
                self.config, self.queues['ttl'], self.queues['neural']
            )
            self.producers['udp'] = RealUDPProducer(
                self.config, self.queues['udp']
            )
        
        # Register with names for better debugging
        for name, producer in self.producers.items():
            self.shutdown_manager.register_thread(producer, name=f"producer_{name}")

    
    def _create_consumers(self):
        """Create consumer processes"""
        self.consumers['sync'] = SynchronizationConsumer(
            self.config, self.queues['ttl'], self.queues['udp'],
            self.queues['sync'], self.category_lookup
        )
        
        # Pass simplified parameters to analysis pipeline
        self.analysis_pipeline = AnalysisPipeline(
            self.config, self.queues['sync'], self.queues['neural'],
            self.psth_dict, self.psth_lock, self.epoch_save_enabled
        )
        self.consumers['analysis'] = self.analysis_pipeline
        self.consumers['analysis'].control_queue = self.control_queue

        # Register with names for better debugging
        for name, consumer in self.consumers.items():
            self.shutdown_manager.register_process(consumer, name=f"consumer_{name}")


    def start_acquisition(self):
        """Start acquisition"""
        if self.is_acquiring:
            return
        
        if not self._is_manager_valid():
            self.append_log("Manager invalid, recreating...")
            self._recreate_manager()

        self.is_acquiring = True
        self.append_log("Starting acquisition...")
        
        for name in ['ttl', 'udp', 'sync', 'neural']:
            self.queues[name] = mp.Queue()
            self.monitor.register_queue(name, self.queues[name])
        
        self._create_producers()
        self._create_consumers()
        
        for producer in self.producers.values():
            producer.start()
        for consumer in self.consumers.values():
            consumer.start()
        
        self.monitor.start_monitoring()
        self.update_timer.start()
        
        self._update_ui_state(running=True)
        
        self.append_log("Acquisition started successfully")

    def stop_acquisition(self):
        """Stop acquisition"""
        if not self.is_acquiring:
            return
        # Prevent re-entry
        if hasattr(self, '_stop_in_progress') and self._stop_in_progress:
            self.append_log("Stop already in progress...")
            return
        
        self._stop_in_progress = True
        self.is_acquiring = False
        self.append_log("Stopping acquisition...")
        
        try:
            # Step 1: Stop GUI timers first (prevents new updates)
            self.append_log("Stopping timers...")
            if hasattr(self, 'update_timer') and self.update_timer.isActive():
                self.update_timer.stop()
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.stop_monitoring()
            
            # Step 2: Stop producers (data sources)
            self.append_log("Stopping producers...")
            for name, producer in list(self.producers.items()):
                try:
                    if hasattr(producer, 'stop'):
                        producer.stop()
                    if hasattr(producer, 'running'):
                        producer.running = False
                    # Unregister from shutdown manager
                    self.shutdown_manager.unregister_thread(producer)
                except Exception as e:
                    self.append_log(f"Warning: Error stopping producer {name}: {e}")
            
            # Brief wait for producers to stop sending data
            time.sleep(0.3)
            
            # Step 3: Clear queues to prevent consumers from processing old data
            self.append_log("Clearing queues...")
            
            for name, queue_obj in list(self.queues.items()):
                try:
                    # Clear with a timeout to prevent hanging
                    cleared = 0
                    start_time = time.time()
                    while not queue_obj.empty() and time.time() - start_time < 0.5:
                        try:
                            queue_obj.get_nowait()
                            cleared += 1
                        except:
                            break
                    if cleared > 0:
                        self.append_log(f"  Cleared {cleared} items from {name} queue")
                except Exception as e:
                    self.append_log(f"  Warning: Could not clear {name} queue: {e}")
            
            # Step 4: Stop consumers (data processors)
            self.append_log("Stopping consumers...")
            for name, consumer in list(self.consumers.items()):
                try:
                    # Set running flag to False first
                    if hasattr(consumer, 'running'):
                        consumer.running = False
                    # Then call stop method
                    if hasattr(consumer, 'stop'):
                        consumer.stop()
                    # Unregister from shutdown manager
                    self.shutdown_manager.unregister_process(consumer)
                except Exception as e:
                    self.append_log(f"Warning: Error stopping consumer {name}: {e}")
            
            # Step 5: Wait for threads to finish (with timeout)
            self.append_log("Waiting for threads to finish...")
            for name, producer in list(self.producers.items()):
                try:
                    if hasattr(producer, 'is_alive') and producer.is_alive():
                        producer.join(timeout=1.0)
                        if producer.is_alive():
                            self.append_log(f"  Warning: Producer {name} did not stop cleanly")
                except Exception as e:
                    self.append_log(f"  Warning: Error joining producer {name}: {e}")
            
            # Step 6: Terminate processes (with timeout)
            self.append_log("Terminating processes...")
            for name, consumer in list(self.consumers.items()):
                try:
                    if hasattr(consumer, 'is_alive') and consumer.is_alive():
                        consumer.terminate()
                        consumer.join(timeout=1.0)
                        if consumer.is_alive():
                            self.append_log(f"  Force killing consumer {name}")
                            consumer.kill()
                            consumer.join(timeout=0.5)
                except Exception as e:
                    self.append_log(f"  Warning: Error terminating consumer {name}: {e}")
            
            # Step 7: Final cleanup
            self.append_log("Final cleanup...")
            
            # Clear references
            self.analysis_pipeline = None
            self.producers.clear()
            self.consumers.clear()
            
            # Close queues properly
            for name, queue_obj in list(self.queues.items()):
                try:
                    if hasattr(queue_obj, 'close'):
                        queue_obj.close()
                    if hasattr(queue_obj, 'join_thread'):
                        queue_obj.join_thread()
                except:
                    pass
            self.queues.clear()
            
            # Update UI state
            self._update_ui_state(running=False)
            
            # Reset epoch save path display if it was active
            if hasattr(self, 'epoch_save_path_label'):
                if self.epoch_save_enabled.value:
                    # Keep the path visible but update status
                    current_text = self.epoch_save_path_label.text()
                    if "Save Path:" in current_text:
                        path_part = current_text.replace("Save Path: ", "")
                        self.epoch_save_path_label.setText(f"Save Path (Stopped): {path_part}")
                        self.epoch_save_path_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
            
            self.append_log("Acquisition stopped successfully")
            
        except Exception as e:
            self.append_log(f"Error during stop: {e}")
            import traceback
            self.append_log(f"Traceback: {traceback.format_exc()}")
        finally:
            self._stop_in_progress = False
    
    @pyqtSlot()
    def update_displays(self):
        """Update all displays including epoch status"""
        # Update queue sizes
        for name in ['ttl', 'udp', 'sync', 'neural']:
            try:
                if name in self.queues and self.queues[name]:
                    queue_size = self.queues[name].qsize()
                    self.monitor.metrics[f'{name}_queue_size'] = queue_size
            except:
                self.monitor.metrics[f'{name}_queue_size'] = 0
        
        psth_data_copy = None
        if self.is_acquiring and self.psth_dict and 'data' in self.psth_dict:
            try:
                if self.psth_lock.acquire(timeout=0.002):
                    try:
                        psth_data_copy = {
                            'data': dict(self.psth_dict.get('data', {})),
                            'counts': dict(self.psth_dict.get('counts', {})),
                            'categories': dict(self.psth_dict.get('categories', {})),
                            'epoch_status': dict(self.psth_dict.get('epoch_status', {})),
                            'last_update': self.psth_dict.get('last_update', 0)
                        }
                    finally:
                        self.psth_lock.release()
                else:
                    return
            except Exception as e:
                self.append_log(f"Error in update_displays: {e}")
                return
        
        if psth_data_copy:
            self._last_raw_psth_data = psth_data_copy
            
            # Update epoch status display
            epoch_status = psth_data_copy.get('epoch_status', {})
            if epoch_status:
                saved_count = epoch_status.get('saved_count', 0)
                save_path = epoch_status.get('save_path', '')
                if save_path and self.epoch_save_enabled.value:
                    current_tooltip = self.epoch_save_path_label.toolTip()
                    if not current_tooltip.endswith(save_path):
                        self.update_epoch_save_path_display(save_path)
                # Update the epoch count label
                self.epoch_status_label.setText(f"Epochs: {saved_count} saved")
                if save_path:
                    self.epoch_status_label.setToolTip(f"Saved to: {save_path}")
            
            if self.display_worker:
                QTimer.singleShot(0, lambda: self.display_worker.process_psth_display(psth_data_copy))
                
                if self.display_worker.current_contrast:
                    QTimer.singleShot(0, lambda: self.display_worker.calculate_dprime(psth_data_copy))
    
    @pyqtSlot(dict, object, str)
    def _update_psth_visualization(self, display_data: dict, time_points: np.ndarray, title: str):
        """Update PSTH plot"""
        for cat, data in display_data.items():
            if cat not in self.psth_curves:
                self.psth_curves[cat] = self.psth_plot.plot(
                    time_points, data['data'],
                    pen=pg.mkPen(data['color'], width=2),
                    name=f"{cat} (n={data['count']})"
                )
            else:
                self.psth_curves[cat].setData(time_points, data['data'])
                self.psth_curves[cat].opts['name'] = f"{cat} (n={data['count']})"
        
        self.psth_plot.setTitle(title)

    @pyqtSlot(object, list, list)
    def _update_dprime_visualization(self, d_primes: np.ndarray, channels: list, colors: list):
        """Update d-prime plot with proper colors"""
        # Store for hover
        self.current_dprime_values = d_primes
        # Convert colors to pyqtgraph format
        brushes = [pg.mkBrush(color) for color in colors]
        # Update bar graph
        self.dprime_bar.setOpts(x=channels, height=d_primes, brushes=brushes)
        # Update title with statistics
        n_significant = np.sum(np.abs(d_primes) >= self.dprime_threshold_spin.value())
        self.dprime_plot.setTitle(
            f"D-prime Across Channels ({n_significant}/{len(d_primes)} "
            f"channels above threshold={self.dprime_threshold_spin.value():.1f})"
        )
        # Update summary
        self.update_dprime_summary(d_primes)
        
    @pyqtSlot(str, float)
    def _monitor_processing_time(self, process_type: str, time_ms: float):
        """Monitor display processing performance"""
        if time_ms > 200:
            if not hasattr(self, '_last_perf_warning'):
                self._last_perf_warning = {}
            
            current_time = time.time()
            last_warning = self._last_perf_warning.get(process_type, 0)
            
            if current_time - last_warning > 10.0:
                self.append_log(f"Warning: {process_type} processing took {time_ms:.1f}ms")
                self._last_perf_warning[process_type] = current_time

    @pyqtSlot(dict)
    def update_monitor_display(self, metrics: dict):
        """Update monitor display with system metrics"""
        for name in ['ttl', 'udp', 'sync', 'neural']:
            size = int(metrics.get(f'{name}_queue_size', 0))
            rate = metrics.get(f'{name}_rate', 0)
            
            if name in self.queue_labels:
                self.queue_labels[name].setText(f"{name.upper()} Queue: {size}")
                self.queue_bars[name].setValue(min(size, 100))
            
            if name in self.rate_labels:
                self.rate_labels[name].setText(f"{name.upper()} Rate: {rate:.1f} Hz")
        
        cpu = int(metrics.get('cpu_percent', 0))
        mem = int(metrics.get('memory_percent', 0))
        data_rate = metrics.get('neural_data_rate_mb', 0)
        
        self.cpu_label.setText(f"CPU: {cpu}%")
        self.cpu_bar.setValue(cpu)
        
        self.memory_label.setText(f"Memory: {mem}%")
        self.memory_bar.setValue(mem)
        
        self.data_rate_label.setText(f"Data Rate: {data_rate:.1f} MB/s")
    
    def clear_buffers(self):
        """Clear all buffers and queues"""
        reply = QMessageBox.question(
            self,
            "Clear Buffers",
            "This will clear all data buffers and queues.\nPSTH and d-prime data will be lost.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._safe_clear_queues()
                
                if self._is_manager_valid():
                    self._safe_clear_psth_dict()
                
                self._clear_ui_plots()
                if hasattr(self, 'fr_matrix_data'):
                    try:
                        # Clear data structures
                        self.fr_matrix_data = None
                        self.fr_stim_order = []
                        self.fr_category_info = {}
                        
                        # Clear the image
                        if hasattr(self, 'fr_image'):
                            self.fr_image.clear()
                        
                        # Clear category boundary lines
                        if hasattr(self, 'category_lines'):
                            for line in self.category_lines[:]:  # Use slice to avoid modification during iteration
                                try:
                                    self.fr_plot.removeItem(line)
                                except:
                                    pass
                            self.category_lines.clear()
                        
                        # Reset hover label
                        if hasattr(self, 'fr_hover_label'):
                            self.fr_hover_label.setText("Hover over matrix to see values")
                        
                        # Reset plot title
                        if hasattr(self, 'fr_plot'):
                            norm_str = "Normalized " if (hasattr(self, 'fr_normalize_checkbox') and 
                                                        self.fr_normalize_checkbox.isChecked()) else ""
                            self.fr_plot.setTitle(f"{norm_str}Mean Firing Rate Matrix")
                        
                        # Reset autoscale flag
                        if hasattr(self, '_fr_autoscaled'):
                            delattr(self, '_fr_autoscaled')
                        
                        # Reset manual scale flag
                        if hasattr(self, '_fr_manual_scaled'):
                            delattr(self, '_fr_manual_scaled')
                            
                    except Exception as e:
                        self.append_log(f"Warning: Error clearing firing rate matrix: {e}")
                
                if self.is_acquiring and hasattr(self, 'control_queue'):
                    try:
                        self.control_queue.put('reset')
                        self.append_log("Sent reset command to analysis pipeline")
                    except Exception as e:
                        self.append_log(f"Error sending reset command: {e}")
                
                if self.display_worker:
                    self._cleanup_display_worker()
                    self._setup_display_worker()
                    self.append_log("Display worker reset complete")
                
                # Reset monitor counters
                for name in self.monitor.counters:
                    self.monitor.counters[name] = 0
                self.monitor.last_counters = self.monitor.counters.copy()
                self.monitor.metrics.clear()
                
                # Reset display
                for name in ['ttl', 'udp', 'sync', 'neural']:
                    if name in self.queue_labels:
                        self.queue_labels[name].setText(f"{name.upper()} Queue: 0")
                        self.queue_bars[name].setValue(0)
                    if name in self.rate_labels:
                        self.rate_labels[name].setText(f"{name.upper()} Rate: 0.0 Hz")
                self.cpu_label.setText("CPU: 0%")
                self.cpu_bar.setValue(0)
                self.memory_label.setText("Memory: 0%")
                self.memory_bar.setValue(0)
                self.data_rate_label.setText("Data Rate: 0.0 MB/s")

                # Reset display worker state
                if self.display_worker:
                    self.display_worker.set_channel(self.channel_selector.value())
                
                self.append_log("All buffers cleared successfully")
                
            except Exception as e:
                self.append_log(f"Error during clear: {e}")
                QMessageBox.warning(
                    self,
                    "Clear Error",
                    f"Error clearing buffers: {str(e)}\nSome data may not have been cleared."
                )

    def _safe_clear_queues(self):
        """Safely clear all queues"""
        for name, queue in list(self.queues.items()):
            if queue is None:
                continue
            
            try:
                count = 0
                start_time = time.time()
                timeout = 1.0
                
                while not queue.empty() and (time.time() - start_time) < timeout:
                    try:
                        queue.get_nowait()
                        count += 1
                    except:
                        break
                
                if count > 0:
                    self.append_log(f"Cleared {name} queue ({count} items)")
                    
            except Exception as e:
                self.append_log(f"Warning: Could not clear {name} queue: {e}")

    def _safe_clear_psth_dict(self):
        """Safely clear PSTH dictionary"""
        try:
            if self.psth_lock.acquire(timeout=0.5):
                try:
                    if self.psth_dict is not None:
                        try:
                            self.psth_dict.clear()
                            self.psth_dict['data'] = {}
                            self.psth_dict['counts'] = {}
                            self.psth_dict['categories'] = {}
                            self.psth_dict['last_update'] = time.time()
                        except (BrokenPipeError, EOFError, ConnectionError) as e:
                            self.append_log("PSTH dict not accessible (manager connection lost)")
                        except Exception as e:
                            self.append_log(f"Could not clear PSTH dict: {e}")
                finally:
                    self.psth_lock.release()
            else:
                self.append_log("Warning: Could not acquire PSTH lock for clearing")
                
        except Exception as e:
            self.append_log(f"Error accessing PSTH lock: {e}")

    def _is_manager_valid(self):
        """Check if the multiprocessing manager is still valid"""
        if not hasattr(self, 'manager') or self.manager is None:
            return False
        
        try:
            test = self.manager.Value('i', 0)
            return True
        except:
            return False

    def _recreate_manager(self):
        """Recreate the multiprocessing manager safely"""
        try:
            if hasattr(self, 'manager') and self.manager:
                try:
                    self.manager.shutdown()
                except:
                    pass
            
            self.manager = Manager()
            self.psth_dict = self.manager.dict()
            self.psth_lock = mp.Lock()
            
            self.shutdown_manager.resources['mp_manager'] = self.manager
            
            self._initialize_psth_dict()
            
            self.append_log("Manager recreated successfully")
            
        except Exception as e:
            self.append_log(f"Failed to recreate manager: {e}")
            raise

    def _clear_ui_plots(self):
        """Clear all UI plots"""
        try:
            if hasattr(self, 'psth_plot'):
                self.psth_plot.clear()
                self.psth_plot.addLegend()
                self.psth_curves.clear()
            
            if hasattr(self, 'dprime_bar'):
                self.dprime_bar.setOpts(x=[], height=[])
                
            # Clear the cached data
            self._last_raw_psth_data = None
                
        except Exception as e:
            self.append_log(f"Warning: Error clearing plots: {e}")
    
    def _update_ui_state(self, running: bool):
        """Update UI state based on acquisition status"""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        
        if running:
            self.status_label.setText("Status: Running")
            self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        else:
            self.status_label.setText("Status: Idle")
            self.status_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")

    @pyqtSlot(str)
    def append_log(self, message: str):
        """Append message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(f"[{timestamp}] {message}")
        else:
            print(f"[{timestamp}] {message}")
        
        logger.info(message)

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event"""
        print("GUI closeEvent triggered")
        
        # Stop acquisition first
        if self.is_acquiring:
            self.stop_acquisition()
        
        # Give processes time to stop
        time.sleep(0.5)
        
        success = self.shutdown_manager.shutdown(force=True)
        
        if success:
            event.accept()
            QApplication.quit()
        else:
            event.accept()

    def _cleanup_gui(self):
        """GUI-specific cleanup callback"""
        if self._is_manager_valid():
            try:
                self.manager.shutdown()
            except:
                pass
        
        try:
            self.log_message_signal.disconnect()
        except:
            pass

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point"""
    import signal as sig_module
    import atexit
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    global_shutdown = ShutdownManager()
    
    def emergency_cleanup():
        global_shutdown.shutdown(force=True)
        # Force exit after cleanup
        import os
        os._exit(0)
    
    atexit.register(emergency_cleanup)

    def signal_handler(sig, frame):
        print("SIGINT received, shutting down...")
        global_shutdown.shutdown(force=True)
        # Force exit after cleanup
        import os
        os._exit(0)
    
    sig_module.signal(sig_module.SIGINT, signal_handler)

    config_path = "config.yaml"
    if Path(config_path).exists():
        config = SystemConfig.from_yaml(config_path)
        print(f"Loaded configuration from {config_path}")
    else:
        config = SystemConfig()
        config.to_yaml(config_path)
        print(f"Created default configuration file: {config_path}")
        print("Please review and adjust settings as needed")
    
    window = RealTimeNeuralGUI(config)
    global_shutdown.active_processes = window.shutdown_manager.active_processes
    global_shutdown.active_threads = window.shutdown_manager.active_threads
    
    window.show()  
    
    try:
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Application error: {e}")
        emergency_cleanup()

if __name__ == "__main__":
    main()