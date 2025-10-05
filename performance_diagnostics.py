#!/usr/bin/env python
"""
SpikeGLX Performance Diagnostics and Configuration Tool
Run this to diagnose and fix Fetch Too Late errors
"""

import time
import psutil
import socket
import subprocess
import platform
from pathlib import Path

def diagnose_system():
    """Run comprehensive system diagnostics"""
    print("=" * 60)
    print("SPIKEGLX PERFORMANCE DIAGNOSTICS")
    print("=" * 60)
    
    # 1. System Information
    print("\n1. SYSTEM INFORMATION:")
    print(f"   Platform: {platform.platform()}")
    print(f"   Processor: {platform.processor()}")
    print(f"   CPU Count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"   CPU Frequency: {psutil.cpu_freq().current:.0f} MHz")
    print(f"   Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # 2. Python Performance
    print("\n2. PYTHON PERFORMANCE:")
    # Test basic loop performance
    start = time.perf_counter()
    for _ in range(10000000):
        pass
    loop_time = time.perf_counter() - start
    print(f"   Basic loop (10M iterations): {loop_time*1000:.1f} ms")
    
    # Test numpy performance
    try:
        import numpy as np
        data = np.random.randn(1000000)
        start = time.perf_counter()
        _ = data.astype(np.int16)
        conv_time = time.perf_counter() - start
        print(f"   Numpy conversion (1M samples): {conv_time*1000:.1f} ms")
    except ImportError:
        print("   Numpy not available")
    
    # 3. Network Latency (if using remote SpikeGLX)
    print("\n3. NETWORK DIAGNOSTICS:")
    sglx_host = input("   Enter SpikeGLX host IP (default: 127.0.0.1): ").strip() or "127.0.0.1"
    
    if sglx_host != "127.0.0.1":
        try:
            # Ping test
            if platform.system() == "Windows":
                result = subprocess.run(['ping', '-n', '10', sglx_host], 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(['ping', '-c', '10', sglx_host], 
                                      capture_output=True, text=True)
            
            # Parse ping results
            output = result.stdout
            if "Average" in output or "avg" in output:
                print(f"   Ping results to {sglx_host}:")
                for line in output.split('\n'):
                    if "Average" in line or "avg" in line or "loss" in line:
                        print(f"     {line.strip()}")
        except Exception as e:
            print(f"   Could not ping {sglx_host}: {e}")
    else:
        print("   Local connection (no network latency)")
    
    # 4. Process Priority
    print("\n4. PROCESS PRIORITY:")
    try:
        current_process = psutil.Process()
        print(f"   Current priority: {current_process.nice()}")
        if platform.system() == "Windows":
            print("   Recommendation: Run as Administrator for real-time priority")
        else:
            print("   Recommendation: Run with nice -20 for highest priority")
    except:
        print("   Could not get process priority")
    
    # 5. Disk I/O Performance
    print("\n5. DISK I/O PERFORMANCE:")
    test_file = Path("test_performance.tmp")
    data = np.random.randn(10000000).astype(np.int16)  # ~20MB
    
    # Write test
    start = time.perf_counter()
    np.save(test_file, data)
    write_time = time.perf_counter() - start
    write_speed = (data.nbytes / (1024**2)) / write_time
    print(f"   Write speed: {write_speed:.1f} MB/s")
    
    # Read test
    start = time.perf_counter()
    _ = np.load(test_file)
    read_time = time.perf_counter() - start
    read_speed = (data.nbytes / (1024**2)) / read_time
    print(f"   Read speed: {read_speed:.1f} MB/s")
    
    # Cleanup
    test_file.unlink()
    
    # 6. Recommendations
    print("\n6. PERFORMANCE RECOMMENDATIONS:")
    print("=" * 60)
    
    recommendations = []
    
    # CPU recommendations
    if psutil.cpu_freq().current < 2000:
        recommendations.append("CPU frequency is low. Check power settings.")
    
    # RAM recommendations
    if psutil.virtual_memory().available / (1024**3) < 4:
        recommendations.append("Low available RAM. Close unnecessary applications.")
    
    # Python performance
    if loop_time > 0.5:
        recommendations.append("Python performance is slow. Consider upgrading Python or hardware.")
    
    # Disk I/O
    if write_speed < 50:
        recommendations.append("Slow disk write speed. Consider SSD or different save location.")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   System performance appears adequate.")
    
    return recommendations


def generate_optimized_config(performance_mode='balanced'):
    """Generate optimized configuration file"""
    
    config_content = f"""# Optimized SpikeGLX Configuration
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

# Performance mode: {performance_mode}
performance_mode: {performance_mode}

# Operation mode
simulation_mode: false

# SpikeGLX connection
sglx_host: "127.0.0.1"
sglx_port: 4142

# Network settings
udp_ip: "127.0.0.1"
udp_port: 33433

# Data acquisition
neural_channels: 384
neural_sample_rate: 30000
ttl_sample_rate: 10593
ttl_sync_bit: 0
ttl_event_bit: 6

# Performance settings for {performance_mode} mode
"""
    
    if performance_mode == 'fast':
        config_content += """
# Fast mode - Maximum throughput
performance:
  worker_processes: 6
  queue_max_size: 2000
  fetch_size_neural: 5000
  fetch_size_ttl: 2000
  min_sleep_time: 0
  max_fetch_time_ms: 200

# Larger buffers
ring_buffer_size_mb: 2000
max_trials_per_category: 2000

# Faster GUI updates
gui_update_interval_ms: 1000
monitor_update_interval_ms: 1000
"""
    
    elif performance_mode == 'safe':
        config_content += """
# Safe mode - Conservative settings
performance:
  worker_processes: 2
  queue_max_size: 500
  fetch_size_neural: 500
  fetch_size_ttl: 200
  min_sleep_time: 2
  max_fetch_time_ms: 50

# Smaller buffers
ring_buffer_size_mb: 500
max_trials_per_category: 500

# Slower GUI updates
gui_update_interval_ms: 2000
monitor_update_interval_ms: 2000
"""
    
    else:  # balanced
        config_content += """
# Balanced mode - Standard settings
performance:
  worker_processes: 4
  queue_max_size: 1000
  fetch_size_neural: 1000
  fetch_size_ttl: 400
  min_sleep_time: 0.5
  max_fetch_time_ms: 100

# Standard buffers
ring_buffer_size_mb: 1000
max_trials_per_category: 1000

# Standard GUI updates
gui_update_interval_ms: 500
monitor_update_interval_ms: 500
"""
    
    config_content += """
# Synchronization
sync_epsilon_ms: 150.0
sync_skew_window_size: 50

# Processing
downsample_factor: 30
psth_window_ms: [-50, 300]
spike_threshold: -4

# File paths
image_info_path: "stim_info.tsv"
save_path: "./data"

# D-prime defaults
dprime_defaults:
  contrast_threshold: 0.5
  time_window_ms: [50, 250]

# Logging
logging:
  level: 'INFO'
  file_rotation: true
  max_log_size_mb: 100
"""
    
    return config_content


def apply_system_optimizations():
    """Apply system-level optimizations"""
    print("\n7. APPLYING SYSTEM OPTIMIZATIONS:")
    print("=" * 60)
    
    system = platform.system()
    
    if system == "Windows":
        print("Windows optimizations:")
        print("1. Set process priority to HIGH:")
        print("   - Open Task Manager")
        print("   - Find python.exe")
        print("   - Right-click > Set Priority > High")
        print("")
        print("2. Disable Windows Defender real-time scanning for data folder")
        print("")
        print("3. Set Power Plan to High Performance:")
        print("   powercfg /setactive SCHEME_MIN")
        
    elif system == "Linux":
        print("Linux optimizations:")
        print("1. Increase process priority:")
        print("   sudo nice -n -20 python RealTimeGUIv4z.py")
        print("")
        print("2. Disable CPU frequency scaling:")
        print("   sudo cpupower frequency-set -g performance")
        print("")
        print("3. Increase network buffer sizes:")
        print("   sudo sysctl -w net.core.rmem_max=134217728")
        print("   sudo sysctl -w net.core.wmem_max=134217728")
        
    elif system == "Darwin":  # macOS
        print("macOS optimizations:")
        print("1. Increase process priority:")
        print("   sudo nice -n -20 python RealTimeGUIv4z.py")
        print("")
        print("2. Disable App Nap for Python:")
        print("   defaults write NSGlobalDomain NSAppSleepDisabled -bool YES")


def main():
    """Main diagnostic routine"""
    print("\nSPIKEGLX PERFORMANCE DIAGNOSTIC TOOL")
    print("=====================================\n")
    
    # Run diagnostics
    issues = diagnose_system()
    
    # Determine performance mode
    if issues:
        if len(issues) > 2:
            mode = 'safe'
            print("\n⚠️  Multiple performance issues detected. Recommending SAFE mode.")
        else:
            mode = 'balanced'
            print("\n⚠️  Some performance issues detected. Recommending BALANCED mode.")
    else:
        mode = 'fast'
        print("\n✓ System performance looks good. Recommending FAST mode.")
    
    # Generate configuration
    print(f"\nGenerating optimized configuration for {mode.upper()} mode...")
    config = generate_optimized_config(mode)
    
    # Save configuration
    config_file = f"config_optimized_{mode}.yaml"
    with open(config_file, 'w') as f:
        f.write(config)
    print(f"Configuration saved to: {config_file}")
    
    # Show system optimizations
    apply_system_optimizations()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Apply the system optimizations above")
    print(f"2. Copy {config_file} to config.yaml")
    print("3. Restart the GUI application")
    print("4. Monitor the performance logs")
    print("=" * 60)


if __name__ == "__main__":
    main()