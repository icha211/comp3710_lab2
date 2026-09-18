## Actual Results From My Runs (Real Numbers, Use These Live)

## Mark Status Summary

### Verified / Fulfilled

- **Earlier DFT component:** completed and verified locally and on Rangpur CPU.
- **Tensor DFT GPU comparison:** completed in Slurm job `598416` on an NVIDIA A100.
- **LFW PCA plus Random Forest/CNN:** completed; CNN test accuracy was `72.7%`.
- **Manual eigenfaces:** completed; `60.6%` accuracy, with 195 correct out of 322 test faces.
- **Binary practice U-Net:** completed with prediction visualisation, but this is not the assessed OASIS multi-class task.
- **Recognition Task 1, VAE:** fulfilled. Slurm job `598512` completed on an NVIDIA A100 and saved the 2D manifold, loss CSV, encoder, and decoder.

### Partial / Not Yet Full Marks

- **Recognition Task 2, OASIS multi-class U-Net:** implemented and executed successfully on an A100. The latest recorded DSC values were Label 1 `0.7994`, Label 2 `0.8290`, and Label 3 `0.8682`. These are below the required `>0.9` for every foreground label, so Task 2 is partial rather than full.
- **Recognition Task 3, GAN:** implemented with checkpoint and evidence-saving support, but no completed OASIS GAN run or realism evidence has been recorded.
- **DAWNBench CIFAR-10/ResNet-18:** implemented, but not yet run or verified on the required Rangpur GPU setup.

### Current Mark Claim

- **Easy recognition level:** supported by the fulfilled VAE, maximum `3/7` recognition-task marks.
- **Medium recognition level:** not yet supported because all three U-Net foreground DSC values must exceed `0.9`.
- **Hard recognition level:** not yet supported because Medium must be achieved and the GAN must show realistic, varied OASIS brain slices with training evidence.

Only claim a higher level after the corresponding measured artifacts and logs exist.

These are genuine outputs from commands I actually executed on this machine. Use these instead of guessing, and point to the matching saved figure in `lab2/`.

### DFT Run

Command: `python lab2_solution.py --part dft`

```text
Naive DFT: 7.751078 s
NumPy FFT: 0.019670 s
Results agree: True
```

Saved figures: [lab2/Figure_1(dft).png](lab2/Figure_1(dft).png) and [lab2/Figure_2(dft).png](lab2/Figure_2(dft).png)

Rangpur venv confirmation, same command: `python lab2_solution.py --part dft`

```text
Naive DFT: 3.667617 s
NumPy FFT: 0.007678 s
Results agree: True
```

### LFW Run (10 epochs)

Command: `python lab2_solution.py --part lfw --epochs 10`

```text
PCA + Random Forest accuracy: 0.5528
CNN test metrics (loss, accuracy): [0.7431, 0.7267]
```

Saved figure: [lab2/Figure_1(lfw_epochs 10).png](lab2/Figure_1(lfw_epochs%2010).png)

### Manual Eigenfaces Run

Command: `python lab2_solution.py --part eigenfaces`

```text
Total Testing: 322
Total Correct: 195
Accuracy: 0.6055900621118012
```

Saved figures: [lab2/Figure_1(eigenfaces).png](lab2/Figure_1(eigenfaces).png) and [lab2/Figure_2(eigenfaces).png](lab2/Figure_2(eigenfaces).png)

### Tensor DFT Runs (Local CPU And Rangpur Login CPU)

Command: `python lab2_solution.py --part dft-tensor`

```text
N=256:  CPU tensor naive DFT 0.0998s | GPU unavailable | tf.signal.fft 0.0025s
N=512:  CPU tensor naive DFT 0.0068s | GPU unavailable | tf.signal.fft 0.0019s
N=1024: CPU tensor naive DFT 0.0187s | GPU unavailable | tf.signal.fft 0.0002s
N=2048: CPU tensor naive DFT 0.0374s | GPU unavailable | tf.signal.fft 0.0002s
```

Rangpur venv confirmation, same command: `python lab2_solution.py --part dft`

```text
Naive DFT: 3.667617 s
NumPy FFT: 0.007678 s
Results agree: True
```

### U-Net Run (10 epochs)

Command: `python lab2_solution.py --part unet --data-root "C:/Users/icha/Downloads/keras_png_slices_data/keras_png_slices_data" --epochs 10`

Saved figure: [lab2/Figure_3(unet mri).png](lab2/Figure_3(unet%20mri).png)

## Gap Analysis: What The Task Sheet Requires vs What `lab2_solution.py` Actually Does

Updated after adding the missing pieces below. Be ready to state which of these you have **actually run** versus only **implemented but not yet executed** — implemented-but-unrun is still an honest, lesser claim than "done and verified."

### Part 1, DFT (1 Mark) — Done and verified on CPU and A100 GPU

- NumPy square wave, Fourier reconstruction, naive DFT vs FFT timing: **done and verified** in `run_dft` (real output already captured above).
- TF tensor-based `naive_dft_tf` plus the CPU-vs-GPU-vs-`tf.signal.fft` sweep across sizes 256/512/1024/2048: **run successfully as Slurm job 598416 on Rangpur's A100 GPU**. TensorFlow explicitly created `GPU:0` with about 38 GB memory. At $N=2048$, the direct tensor DFT took 0.0346 s on CPU and 0.0060 s on GPU, while `tf.signal.fft` took 0.0026 s. This ordering is expected for the larger input: FFT is fastest because it uses $O(N\log N)$ work; the vectorised GPU direct DFT is next because the A100 parallelises its $O(N^2)$ matrix work; CPU direct DFT is slowest. Small-size timings are dominated by TensorFlow/GPU startup and transfer overhead, so they are not monotonic and should not be used alone to infer scaling.

Rangpur A100 evidence (Slurm job 598416):

```text
N=256:  CPU tensor naive DFT 0.0119s | GPU tensor naive DFT 0.0707s | tf.signal.fft 0.0582s
N=512:  CPU tensor naive DFT 0.0061s | GPU tensor naive DFT 0.0028s | tf.signal.fft 0.0029s
N=1024: CPU tensor naive DFT 0.0132s | GPU tensor naive DFT 0.0111s | tf.signal.fft 0.0160s
N=2048: CPU tensor naive DFT 0.0346s | GPU tensor naive DFT 0.0060s | tf.signal.fft 0.0026s
```

### Part 2, Eigenfaces (1 Mark) — Done and verified

- The original `run_lfw` still uses `sklearn.PCA`, kept because it has verified real results (55.3% Random Forest and 72.7% CNN accuracy in the latest run).
- Added and executed `run_eigenfaces_manual`: manual `np.linalg.svd` PCA, eigenface gallery, compactness plot, and `RandomForestClassifier(max_features=150)`. It correctly classified 195 of 322 test faces for **60.6% accuracy**.

### Part 3.1, CNN Classifier (1 Mark) — Done and verified

Unchanged, matches spec. The latest recorded 10-epoch run reached **72.7% test accuracy** with test loss **0.7431**.

### Part 3.2, DAWNBench CIFAR-10/ResNet-18 (4 Marks) — Now implemented, not yet run

Added `build_resnet18` (CIFAR-style 3x3 stem, 4 stages of 2 residual blocks each = 18 layers) and `run_dawnbench`, with an optional `--mixed-precision` flag. Run via `python lab2_solution.py --part dawnbench --epochs 5 --mixed-precision`. **Not yet run anywhere.** The >90%/94% accuracy and ~360-second targets are cluster-GPU claims and must be measured on Rangpur's A100 before you can honestly report a number.

### Part 4, Recognition Tasks (7 Marks) — Task 1 fulfilled; Task 2 improvement run pending

- **Task 1, VAE — FULFILLED:** Slurm job `598512` completed with `COMPLETED` and exit code `0:0` in 57 seconds on node `a100-9`. TensorFlow detected `GPU:0` as an NVIDIA A100. The run completed one epoch with loss `17064.3125` and saved `vae_latent_manifold.png`, `vae_loss.csv`, `encoder.keras`, and `decoder.keras` under `demo_outputs/vae`. This is sufficient evidence for Task 1, including the required 2D manifold visualisation.
- **Task 2, OASIS multi-class U-Net — PARTIAL, improvement pending:** Slurm job `598598` completed on an NVIDIA A100 and saved `oasis_unet.keras`, `segmentation_predictions.png`, `test_dsc.txt`, and `unet_training_loss.csv`. The held-out test DSC values were Label 1 `0.7861`, Label 2 `0.8050`, and Label 3 `0.8350`. The training objective has now been improved to combine categorical cross-entropy with foreground Dice loss, and a longer run should be submitted. These original values prove the pipeline and inference work, but they do not satisfy the required `>0.9` DSC for every foreground label.
- **Task 3, GAN:** implemented with a DCGAN-style generator/discriminator, alternating updates, fixed-noise sample grids, per-epoch generated images, loss CSV/plot, and checkpoints. No completed OASIS realism evidence has been recorded yet.
