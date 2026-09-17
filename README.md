# COMP3710 Lab 2

`lab2_solution.py` contains runnable implementations for the locally reproducible parts of the Lab 2 sheet.

## Run

From this folder:

```powershell
python lab2_solution.py --part dft
python lab2_solution.py --part lfw --epochs 10
python lab2_solution.py --part unet --data-root "C:/Users/icha/Downloads/keras_png_slices_data/keras_png_slices_data" --epochs 10
```

Install the required packages in the selected VS Code Python environment if needed:

```powershell
python -m pip install numpy matplotlib scikit-learn tensorflow pillow
```

## Included

- Part 1: square-wave Fourier reconstruction, direct DFT, NumPy FFT comparison, and spectrum plot.
- Part 2: LFW PCA/eigenfaces and Random Forest baseline.
- Part 3.1: two-layer CNN face classifier.
- Supplied PNG dataset: binary U-Net training, Dice metric, test evaluation, and prediction visualisation.

The supplied PNG data is suitable for segmentation experiments, but it is not the LFW or CIFAR-10 dataset named in the lab sheet.

## Not locally reproducible

The DAWNBench CIFAR-10 target requires Rangpur GPU timing and the OASIS VAE/UNet/GAN tasks require the cluster dataset at `/home/groups/comp3710/`. Those results must be trained and demonstrated on Rangpur; this Windows workspace cannot honestly produce those cluster measurements.

Run training with a small epoch count first to verify the environment, then increase `--epochs` for reported experiments. Do not report example metrics until the corresponding cells have actually been executed on your data.
