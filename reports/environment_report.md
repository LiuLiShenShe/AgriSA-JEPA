# Environment Report

## Host

- OS: Ubuntu 24.04.4 LTS; kernel `6.17.0-40-generic`.
- GPUs: 2 x NVIDIA RTX A6000, 49,140 MiB each.
- NVIDIA driver: `590.48.01`; driver reports CUDA support 13.1.
- Tested GPU: physical device 1, selected with `--devices cuda:1`; it was idle before the runs. The PyTorch build uses CUDA 12.8.
- Filesystem available during setup: about 429 GiB on `/` and 581 GiB on `/data`.
- AgriField archive: 11,900,356,365 bytes on `/data`; dataset remains outside the Git worktree.

## Isolated Python environment

The project uses a Conda environment at the ignored path `.venv/`, with `sys.prefix` resolving to the repository's `.venv`. It does not use system site packages.

| Package/runtime | Tested version |
| --- | --- |
| Python | 3.13.16 |
| PyTorch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| submitit | 1.5.4 |

`torch.cuda.is_available()` returned `True`; PyTorch reported CUDA 12.8 and two visible A6000 GPUs. `python -m pip check` completed with `No broken requirements found.` The tested package versions are recorded in `environment/requirements-lock.txt`.

The host's base Python was 3.13.12, with its existing PyTorch/torchvision installation left unchanged. All project package installation and execution used `.venv`; no system or base-environment package was upgraded or removed.

## Recreate

From a fresh clone on Linux with Conda and network access:

```bash
conda create --yes --prefix .venv python=3.13 pip
conda run --prefix .venv python -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r environment/requirements-lock.txt
conda run --prefix .venv python -m pip check
```

The lock records the tested Python package resolution, including the CUDA 12.8 PyTorch wheels; it is not a cross-platform Conda lockfile. The host still needs a compatible NVIDIA driver and a working CUDA-capable GPU for the GPU smoke run.

## References

- Official I-JEPA repository: https://github.com/facebookresearch/ijepa
- Conda project-prefix environments: https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
- PyTorch CUDA 12.8 wheel index used for the tested build: https://download.pytorch.org/whl/cu128
