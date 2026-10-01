# DDR4-PCIe-4-vs-DDR5-PCIe-5-for-CUDA-training

This repository compares training on DDR4/PCIe 4 and DDR5/PCIe 5 workstations using RTX PRO 6000 WS GPUs. `train_DDP.py` supports one or two GPUs; `train_PP.py` uses two pipeline stages. The plot report cumulative training-step time to validation loss ≤ 3.28, excluding validation and warm-up.

<img width="3416" height="2972" alt="workstation-training-comparisons" src="https://github.com/user-attachments/assets/79fa1fde-5fdd-49bc-9ddf-f49fff74a15f" />

## Running the benchmarks

1. Start up a container with the given image (PyTorch 2.10.0 + CUDA 12.8):

https://cloud.vast.ai/?ref_id=195884&creator_id=195884&name=anywinter4079%2Fpytorch%3A2.10.0-cu128

2. Clone and `cd` into this repository:

```
git clone https://github.com/Any-Winter-4079/DDR4-PCIe-4-vs-DDR5-PCIe-5-for-CUDA-training.git && \
cd DDR4-PCIe-4-vs-DDR5-PCIe-5-for-CUDA-training
```

3. Download the dataset:

```
python data/fineweb-npy.py
```

4. Run, e.g., for pipeline parallelism:

```
torchrun --standalone --nproc_per_node=2 train_PP.py
```
