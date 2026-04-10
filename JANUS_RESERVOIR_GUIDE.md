# Janus Compute Reservoir Integration Guide

This guide explains how to use the **Compute Reservoir** to run Janus models on hardware with low VRAM.

## How it Works
The `JanusLayerReservoir` (implemented in `janus_reservoir.py`) intercepts calls to individual transformer blocks in the `Avus` model. 
- **Charging:** During the first inference, the reservoir calculates each layer's output and saves it to the `compute_storage/` directory on your disk.
- **Discharging:** On subsequent calls with the same input, the reservoir retrieves the results from disk instead of re-calculating them on the GPU.
- **VRAM Savings:** This reduces peak VRAM usage because intermediate activation tensors are offloaded to disk rather than being held in memory.

## Integration in `avus.py`
I have already integrated the reservoir into your `avus.py`. You can now initialize the `Avus` model with an optional `use_reservoir=True` flag:

```python
from avus import Avus, AvusConfig

config = AvusConfig.from_file("config_avus_1b.json")
# Enable the reservoir to save VRAM
model = Avus(config, use_reservoir=True)
```

## Running the Demo
To see the reservoir in action and verify VRAM offloading, run the provided test script:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 janus_vram_test.py
```

## Performance & Storage
- **Storage:** The cached compute is stored in `compute_storage/`. You can clear this at any time by calling `model.reservoir.clear()`.
- **Speed:** The first pass (charging) will be slightly slower due to disk I/O. Subsequent passes (discharging) will be significantly faster and use less memory.
- **VRAM Impact:** This is most effective for large batch sizes or extremely deep models where intermediate activations would normally cause an "Out of Memory" (OOM) error.

## Key Files
- `janus_reservoir.py`: The core reservoir logic for Janus.
- `janus_vram_test.py`: A benchmark script to demonstrate VRAM offloading.
- `avus.py`: The updated model architecture with reservoir support.
