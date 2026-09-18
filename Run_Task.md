# COMP3710 Lab 2 Demo Runbook

The recognition section is graded as follows:

- **Easy: 3/7 maximum:** Task 1, VAE.
- **Medium: 5/7 maximum:** Tasks 1 and 2, VAE plus multi-class U-Net.
- **Hard: 7/7 maximum:** Tasks 1, 2, and 3, VAE plus U-Net plus GAN.

The commands below must be run on Rangpur using the real OASIS data. A command that only completes locally, or code that has not produced results, is not evidence for full marks.

## 1. Rangpur setup

```bash
ssh s5022035@rangpur.compute.eait.uq.edu.au
cd ~/comp3710_lab2
source venv/bin/activate
mkdir -p demo_outputs gan_checkpoints logs
ls /home/groups/comp3710/OASIS
```

If the repository is not present yet:

```bash
git clone https://github.com/icha211/comp3710_lab2.git
cd comp3710_lab2
source venv/bin/activate
```

The expected data layout is:

```text
/home/groups/comp3710/OASIS/
  keras_png_slices_train/case_*.png
  keras_png_slices_validate/case_*.png
  keras_png_slices_test/case_*.png
  keras_png_slices_seg_train/seg_*.png
  keras_png_slices_seg_validate/seg_*.png
  keras_png_slices_seg_test/seg_*.png
```

## 2. Easy: VAE

```bash
python lab2_solution.py --part vae \
  --data-root /home/groups/comp3710/OASIS \
  --epochs 10 --latent-dim 2 --output-dir demo_outputs
```

Show these files during the demo:

```text
demo_outputs/vae/vae_latent_manifold.png
demo_outputs/vae/vae_loss.csv
demo_outputs/vae/encoder.keras
demo_outputs/vae/decoder.keras
```

Explain the encoder, reparameterisation trick, reconstruction loss, KL loss, and how the 2D latent grid is decoded into the manifold image.

## 3. Medium: multi-class U-Net

```bash
python lab2_solution.py --part oasis-unet \
  --data-root /home/groups/comp3710/OASIS \
  --epochs 30 --num-classes 4 --output-dir demo_outputs
```

Show these files and terminal output:

```text
demo_outputs/oasis_unet/test_dsc.txt
demo_outputs/oasis_unet/segmentation_predictions.png
demo_outputs/oasis_unet/oasis_unet.keras
demo_outputs/oasis_unet/unet_training_loss.csv
```

The final test output must show mean DSC above `0.9` for labels 1, 2, and 3. The training objective combines softened class-balanced categorical cross-entropy with foreground Dice loss, and the best validation checkpoint is used for test evaluation. Explain that the network has four softmax channels, the masks are one-hot encoded, and predictions are converted back to labels with `argmax` for discrete DSC.

For the live inference demonstration, use the saved model or the test predictions and show an input MRI, ground-truth mask, and predicted mask.

## 4. Hard: GAN

```bash
python lab2_solution.py --part gan \
  --data-root /home/groups/comp3710/OASIS \
  --epochs 20 --latent-dim 128 --output-dir demo_outputs
```

Show these files:

```text
demo_outputs/gan/generated_epoch_001.png
demo_outputs/gan/generated_epoch_020.png
demo_outputs/gan/generated_final.png
demo_outputs/gan/gan_losses.csv
demo_outputs/gan/gan_loss_curves.png
gan_checkpoints/
```

Explain the generator, discriminator, alternating updates, `tanh` output scaling, binary cross-entropy losses, fixed noise used to compare epochs, and how the samples were checked for realism and mode collapse. The instructor decides whether the generated brains are realistic enough for full GAN marks.

## 5. Evidence checklist

Before the demo, confirm that:

- VAE output was produced from the real OASIS data.
- The VAE manifold image is saved and readable.
- U-Net DSC is above `0.9` for every foreground label.
- U-Net predictions are from the held-out test split.
- U-Net input, ground truth, and prediction images are saved.
- GAN samples show brain structure and variation rather than mode collapse.
- GAN loss values and generated images are saved across epochs.
- The GitHub repository contains the README, source code, and these results or clearly documented result locations.

Do not claim a metric until its command has actually produced it.

## Earlier lab components

These commands are separate from the VAE/U-Net/GAN 7-mark recognition breakdown:

```bash
python lab2_solution.py --part dft
python lab2_solution.py --part dft-tensor
python lab2_solution.py --part lfw --epochs 10
python lab2_solution.py --part eigenfaces
python lab2_solution.py --part dawnbench --epochs 5 --mixed-precision
```

The DAWNBench timing and accuracy target must be measured inside a GPU Slurm job. For a GPU run, submit a suitable job script, then inspect the job-specific files rather than assuming a fixed Slurm job number:

```bash
sbatch train_job.sh
squeue -u "$USER"
cat logs/train_<job_id>.out
cat logs/train_<job_id>.err
```
