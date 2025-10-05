# Real-time Neural Analysis GUI System
## Installation and Usage Guide

### System Requirements

- Python 3.8+
- Windows/Linux with sufficient RAM (16GB+ recommended)
- SpikeGLX running on the same or networked machine
- MonkeyLogic or similar behavioral control system

### Installation

1. **Install Required Dependencies**
```bash
pip install numpy scipy pandas pyyaml
pip install PyQt6 pyqtgraph
pip install multiprocessing-logging
```

2. **Install SpikeGLX Python SDK**
   - Copy the `sglx_pkg` folder from SpikeGLX installation to your project directory
   - Ensure the C++ SDK DLLs are accessible

3. **Prepare Configuration Files**
   - Edit `config.yaml` with your system parameters
   - Prepare `image_info.tsv` with columns: `Idx`, `FileName`, `Category`

### Quick Start

1. **Start SpikeGLX**
   - Launch SpikeGLX and start recording
   - Note the IP address and port (default: 127.0.0.1:4142)

2. **Configure MonkeyLogic**
   - Set UDP output to send messages to port 33433
   - Message format: `"timestamp=%s, value=%d, name=%s\n"`

3. **Launch the GUI**
```bash
python realtime_neural_gui.py
```

4. **Start Acquisition**
   - Click "Start Acquisition" button
   - Monitor the system log for status updates
   - Select channels to view PSTH
   - Perform d-prime analysis as needed

### Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐
│  SpikeGLX API   │     │   UDP Listener  │
│   (Producer I)  │     │  (Producer II)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    ┌────────────────────────────┐
    │  Synchronization Consumer  │
    │  (Dynamic Clock Matching)  │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │    Analysis Pipeline       │
    │  • Epoching                │
    │  • Spike Detection         │
    │  • Downsampling            │
    │  • PSTH Aggregation        │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │      PyQt6 GUI             │
    │  • Real-time PSTH          │
    │  • D-prime Analysis        │
    │  • System Monitoring       │
    └────────────────────────────┘
```

### Key Features

#### 1. **Dual-Producer Architecture**
- **Neural/TTL Producer**: Continuously fetches data from SpikeGLX
- **UDP Producer**: Listens for stimulus events from behavioral software

#### 2. **Dynamic Clock Synchronization**
- Automatically corrects for clock drift between hardware and PC
- Maintains sliding window of clock offsets for robust matching
- Configurable tolerance window (default: 50ms)

#### 3. **Efficient Processing Pipeline**
- Zero-copy shared memory for high-throughput data transfer
- Lock-protected concurrent updates to PSTH matrices
- NumPy-optimized downsampling (30x reduction)

#### 4. **Real-time Visualizations**
- Multi-condition PSTH viewer with channel selection
- On-demand d-prime sensitivity analysis
- Live system status monitoring

### Extending the System

#### Adding New Analysis Modules

Create a new analysis class inheriting from `mp.Process`:

```python
class CustomAnalyzer(mp.Process):
    def __init__(self, config, input_queue, output_queue):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
    
    def run(self):
        while True:
            data = self.input_queue.get()
            result = self.analyze(data)
            self.output_queue.put(result)
    
    def analyze(self, data):
        # Your custom analysis here
        pass
```

#### Adding New Visualizations

Add a new tab to the GUI:

```python
def create_custom_widget(self) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout()
    
    # Create your custom plot
    self.custom_plot = pg.PlotWidget(title="Custom Analysis")
    layout.addWidget(self.custom_plot)
    
    widget.setLayout(layout)
    return widget

# In init_ui():
custom_widget = self.create_custom_widget()
self.tab_widget.addTab(custom_widget, "Custom Analysis")
```

### Performance Optimization

1. **Adjust Buffer Sizes**
   - Increase `ring_buffer_size_mb` for longer recording sessions
   - Tune `queue_max_size` based on processing speed

2. **Optimize Processing**
   - Increase `worker_processes` for more parallel analysis
   - Adjust `downsample_factor` to balance time resolution vs. performance

3. **Memory Management**
   - Monitor shared memory usage
   - Implement circular buffer overflow handling
   - Add automatic data saving for long sessions

### Troubleshooting

#### Connection Issues
- Verify SpikeGLX is running and accessible
- Check firewall settings for UDP port
- Confirm IP addresses in config.yaml

#### Synchronization Problems
- Increase `sync_epsilon_ms` if events aren't matching
- Check TTL channel configuration
- Verify UDP message format from behavioral software

#### Performance Issues
- Reduce `neural_channels` if processing can't keep up
- Increase `downsample_factor` for faster processing
- Check CPU usage and adjust `worker_processes`

### Data Format Specifications

#### Image Info File (TSV)
```
Idx	FileName	Category
0	stim001.png	face
1	stim002.png	object
2	stim003.png	face
...
```

#### UDP Message Format
```
timestamp=1234567890, value=42, name=stim042.png
```

#### Output Data Structure
- PSTH matrices: `(categories, time_bins, channels)`
- D-prime values: `(channels,)` array
- Trial metadata: JSON with event timing and stimulus info

### API Reference

#### Key Classes

- `SpikeGLXProducer`: Handles data acquisition from SpikeGLX
- `UDPEventProducer`: Receives stimulus events via UDP
- `SynchronizationConsumer`: Matches TTL and UDP events
- `AnalysisPipeline`: Processes neural data through multiple stages
- `DPrimeAnalyzer`: Calculates sensitivity metrics
- `RealTimeNeuralGUI`: Main GUI application

#### Configuration Parameters

See `config.yaml` for all configurable parameters with descriptions.

### Best Practices

1. **Start Small**: Test with a subset of channels first
2. **Monitor Logs**: Check system log for warnings and errors
3. **Save Periodically**: Implement automatic data checkpointing
4. **Validate Synchronization**: Verify TTL-UDP matching accuracy
5. **Benchmark Performance**: Profile critical sections for optimization

### Support and Contributing

For issues or feature requests, consider:
- Adding comprehensive error handling
- Implementing data persistence
- Creating unit tests for critical components
- Documenting experimental protocols

### License

This implementation is provided as a framework for research purposes.
Adapt and extend according to your experimental needs.