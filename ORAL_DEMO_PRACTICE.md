# COMP3710 Lab 2 Oral Demo Practice

Use this as a speaking guide for a 3 minute presentation, then as a question bank for follow-up questions. The goal is to show that the work runs, that you understand the theory behind it, and that you can explain your own implementation choices.

## What The Tutor Is Assessing

Based on the course clarification, prepare for all of these areas:

- Theory: explain PCA, CNNs, U-Net, GANs, DFT, loss functions, metrics, and the normal architecture of each model.
- Code ownership: explain the data preprocessing, model layers, training loop, validation, testing, and outputs in code you wrote or adapted with AI.
- Practical evidence: show the compulsory outputs requested in the task sheet, such as training curves, an inference visualisation, and reported metrics.
- Rangpur competence: be able to log in, submit a Slurm job, find its output and error logs, and explain how you would train or resume a model there.
- Critical judgement: identify limitations or bugs in a standard implementation and make a small justified modification when asked.

Do not describe every line from memory. Be ready to explain the purpose, input, output, and design reason for every major function. A good answer follows this pattern: "The input is ..., this step does ..., it produces ..., and this is needed because ...."

## Current Verified Status

The following earlier components have real evidence and can be presented as completed:

- NumPy DFT and Fourier reconstruction: verified locally and on Rangpur CPU.
- Tensor DFT CPU/GPU comparison: verified in Slurm job `598416` on an A100.
- LFW PCA plus Random Forest/CNN: verified, including the recorded CNN accuracy of 72.7%.
- Manual SVD eigenfaces: verified at 60.6% accuracy on 195 of 322 test faces.
- Binary practice U-Net: verified with prediction visualisation, but this is not the assessed OASIS task.

The three assessed recognition tasks are currently at different evidence levels:

- VAE: **fulfilled for Task 1**. Slurm job `598512` completed successfully on an NVIDIA A100 with TensorFlow GPU support, and the manifold and model artifacts were saved.
- OASIS multi-class U-Net: **completed with partial evidence**. Slurm job `598598` produced the trained model, prediction visualisation, loss history, and per-label DSC results, but the three DSC values did not reach the required `>0.9` threshold.
- GAN: implemented with checkpoints and saved-image support, but no completed OASIS run or realism evidence has been recorded.

Therefore, the current evidence supports Task 1 and completed partial work for Task 2, but it does not yet support the full Medium `5/7` level or the full `7/7` level.

## Full Oral Transcript: Start To Finish

Use this as one continuous explanation. Replace bracketed wording only when you have a new measured result.

### Opening: What The Rubric Requires

"This project has two related groups of work. The earlier lab components cover DFT, PCA and face recognition, CNN classification, eigenfaces, and a binary practice U-Net. The recognition section in the task sheet is graded separately: Task 1 is the VAE, Task 2 is the OASIS multi-class U-Net, and Task 3 is the GAN. Easy requires Task 1 for a maximum of 3 marks, Medium requires Tasks 1 and 2 for a maximum of 5 marks, and Hard requires all three for a maximum of 7 marks.

For every task, I need more than code. I need to show that the code runs on the correct data, explain the model and loss, report results on the correct split, and show the required visual evidence. I keep training, validation, and test data separate. The validation data helps monitor development, while the test data is used only for final evaluation. This avoids claiming performance from data used to tune the model."

### Dataset And Reproducible Workflow

"The OASIS data is stored on Rangpur at `/home/groups/comp3710/OASIS`. It contains train, validate, and test PNG image folders, plus corresponding segmentation-mask folders. My command-line program uses `--part` to select one experiment, `--data-root` to locate the dataset, and `--epochs` to control training. For long jobs I submit a Slurm script requesting an A100, activate the virtual environment, load CUDA, write stdout and stderr logs, and save artifacts under `demo_outputs`. The login node is only the control terminal; the actual training runs on an allocated compute node."

### Task 1: VAE Concept, Code, And Result

"A variational autoencoder learns a compact probabilistic representation of images. The encoder receives an MRI slice and produces a latent mean and log variance. The reparameterisation trick samples a latent vector while preserving differentiability:

$$
z = \mu + \exp(0.5\log\sigma^2)\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
$$

The decoder maps the latent vector back to an image. Training balances reconstruction quality with a KL-divergence regulariser. The reconstruction term makes the output resemble the input, while the KL term keeps the latent distribution close to a standard normal distribution so I can sample meaningful points.

In the code, `build_vae` creates the convolutional encoder and transpose-convolution decoder. `run_vae` loads the OASIS images, streams them in batches to avoid loading the complete dataset into RAM, performs the custom gradient-tape training step, records loss, and samples a two-dimensional grid from the decoder. The grid becomes the required latent manifold visualisation.

I ran the VAE on Slurm job `598512`. It completed successfully on an NVIDIA A100 with TensorFlow GPU support. The one-epoch loss was `17064.3125`. The saved evidence is `vae_latent_manifold.png`, `vae_loss.csv`, `encoder.keras`, and `decoder.keras` in `demo_outputs/vae`. This fulfils Task 1 and supports the Easy level maximum of 3 out of 7."

### Task 2: Multi-class U-Net Concept, Code, And Result

"A U-Net performs pixel-level segmentation. Unlike classification, which predicts one label for an image, segmentation predicts one class for every pixel. The encoder uses convolution and pooling to learn context. The bottleneck contains compressed high-level features. The decoder upsamples to the original resolution. Skip connections concatenate encoder features with decoder features so boundaries and spatial detail are not lost.

For OASIS there are four mutually exclusive labels, so the final layer has four softmax channels. The masks are one-hot encoded and the training objective uses categorical cross-entropy plus a foreground Dice loss. The foreground Dice term is important because background pixels can dominate ordinary accuracy. At inference, `argmax` selects one class per pixel. I then calculate discrete DSC separately for labels 1, 2, and 3:

$$
DSC_c = \frac{2|A_c \cap B_c|}{|A_c|+|B_c|}.
$$

The code path is `load_oasis_multiclass_split` to load and remap masks, `unet_model_multiclass` to build the four-channel network, and `run_oasis_unet` to train, save the model, predict the held-out test set in small batches, calculate DSC, and save the prediction grid.

The completed Slurm run produced `oasis_unet.keras`, `segmentation_predictions.png`, `test_dsc.txt`, and `unet_training_loss.csv`. The first measured DSC values were Label 1 `0.7861`, Label 2 `0.8050`, and Label 3 `0.8350`. This proves the OASIS pipeline, A100 training, test inference, discrete evaluation, and artifact saving work. However, the task requires every foreground label to exceed `0.9`, so this run is partial Task 2 evidence rather than full Medium credit."

### Task 3: GAN Concept, Code, And Required Result

"A GAN contains a generator and a discriminator. The generator maps a random latent vector to a synthetic 2D MRI slice. The discriminator receives either a real or generated slice and predicts whether it is real. The discriminator learns to distinguish the two distributions, while the generator learns to fool the discriminator. They are trained in alternating updates.

In the code, `build_gan_generator` uses a dense layer, reshape, and transpose convolutions to increase resolution. `build_gan_discriminator` uses convolutional downsampling and a final real-versus-fake output. `run_gan` scales the images to match the generator's `tanh` output, calculates binary cross-entropy losses, updates both networks with gradient tapes, saves checkpoints, and generates fixed-noise image grids after each epoch.

For full Task 3 evidence I must show `generated_epoch_001.png`, a later epoch grid, `generated_final.png`, `gan_losses.csv`, `gan_loss_curves.png`, and the checkpoint directory. I must inspect the images for realistic brain structure and variation, because loss values alone do not prove realism. I must also check for mode collapse, where different latent vectors produce nearly identical images. The instructor judges whether the generated brains are realistic enough for full GAN credit."

### Marks And Honest Final Position

"At this point, Task 1 is fulfilled. Task 2 has a completed working run but its DSC values are below the required threshold. Task 3 is implemented but still needs a completed OASIS run and visual realism evidence. Therefore I can honestly claim the VAE task and partial U-Net work, but not the full 7 out of 7 yet.

To claim Medium, I need a new U-Net run where Labels 1, 2, and 3 are all above `0.9`. To claim Hard, I also need the GAN image grids, loss evidence, checkpoints, and convincing evidence of realistic, varied brains. I will report the measured result rather than converting high pixel accuracy or low training loss into a claim that the rubric does not support."

### Closing Summary

"The main idea across the project is matching the model contract to the task. DFT converts signals into frequency components. PCA reduces image dimensionality before classical classification. CNNs learn spatial features for image labels. A U-Net predicts a class at every pixel. A VAE learns a smooth latent image manifold. A GAN learns to generate new images through competition between a generator and discriminator. My code exposes each experiment as a reproducible command, saves the evidence, and evaluates the metric required by the task sheet."

## Important Scope Check Before The Demo

The local `lab2_solution.py` contains the assessed OASIS VAE, multi-class U-Net, and GAN implementations, but implementation alone is not evidence of a mark. Only report a recognition-task result after the corresponding OASIS command completes and its required outputs are saved.

Say this honestly if asked:

"This local script was useful for debugging the workflow and explaining the core methods. The OASIS and DAWNBench requirements need their specified dataset and Rangpur environment. I will only claim results for a task once I have run and saved the corresponding experiment there."

This distinction matters: the local U-Net uses one sigmoid output, binary cross-entropy, and a continuous Dice metric. The OASIS requirement uses discrete labels $0, 1, 2, 3$ and requires a standard **discrete Dice score for each foreground label** $1$, $2$, and $3$ on the held-out test set. A binary result does not prove the OASIS multi-class target.

## 30 Second Opening

"For Lab 2, I implemented runnable experiments for Fourier analysis, face recognition, CNN classification, and U-Net segmentation in `lab2_solution.py`. The script is split into explicit command-line parts so each experiment can be run independently. My focus was to connect the theory to practical outputs: reconstructing and analysing a square wave with Fourier methods, comparing classical machine learning with a CNN on LFW faces, and training a U-Net style model for binary medical image segmentation."

## What To Demonstrate Live

Run these from the project folder.

```powershell
python lab2_solution.py --part dft
python lab2_solution.py --part lfw --epochs 1
python lab2_solution.py --part unet --data-root "C:/Users/icha/Downloads/keras_png_slices_data/keras_png_slices_data" --epochs 1
```

For the real demo, use a small epoch count if time is limited. If you have already trained longer runs, explain the longer results and only run a quick smoke test live.

## 3 Minute Presentation Structure

### 1. Task Rundown

Say:

"The code has three main runnable sections. The DFT section creates a square wave, reconstructs it from odd Fourier harmonics, compares a direct DFT implementation with NumPy FFT, and plots the frequency spectrum. The LFW section loads face images, applies PCA for dimensionality reduction, trains a Random Forest baseline, then trains a small CNN. The U-Net section loads paired image and mask PNG files, trains a segmentation model, evaluates it with Dice coefficient, and visualises predictions."

Point to:

- `square_wave`, `square_wave_fourier`, `naive_dft`, and `run_dft`
- `run_lfw`
- `load_png_split`, `dice_coefficient`, `unet_model`, and `run_unet`

### 2. Fourier And DFT Explanation

Say:

"A square wave can be approximated by adding odd sine harmonics. More harmonics make the approximation sharper, but near discontinuities we still see ringing, which is the Gibbs phenomenon. I implemented a direct DFT to show the mathematical definition: each frequency bin is computed by summing every sample multiplied by a complex exponential. This is $O(N^2)$, while FFT computes the same transform much faster, around $O(N \log N)$."

Key formula:

$$
X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}
$$

Expected follow-up answer:

"The spectrum has peaks at odd harmonics because the square wave is made from odd sine components. The amplitude drops roughly with $1/k$, where $k$ is the harmonic number."

### 3. PCA And Random Forest Explanation

Say:

"The LFW images are high-dimensional, so I flatten each image and use PCA to project it into a lower-dimensional feature space. PCA finds directions of maximum variance, which are often called eigenfaces in the face recognition context. A Random Forest then classifies those PCA features. This gives a classical machine learning baseline before using a neural network."

Expected follow-up answer:

"PCA is useful because it reduces noise and computation while preserving major variation in the data. The trade-off is that PCA is unsupervised, so the directions of highest variance are not always the directions that best separate identities."

### 4. CNN Explanation

Say:

"The CNN takes the face image as a 2D input instead of flattening it immediately. Convolutional layers learn local filters such as edges, textures, and facial features. Max pooling reduces spatial size and adds some translation tolerance. The dense layers then use the learned features for classification. The final softmax outputs one probability per person."

Expected follow-up answer:

"The model uses sparse categorical cross-entropy because the labels are integer class IDs, not one-hot vectors. Accuracy is suitable here because this is a multi-class classification task."

### 5. U-Net Segmentation Explanation

Say:

"The U-Net model is for image segmentation, so the output is an image-sized mask rather than one class label. The encoder compresses the image into deeper features, and the decoder upsamples back to the original resolution. Skip connections concatenate encoder features with decoder features, which helps recover spatial detail. The final sigmoid predicts the probability that each pixel belongs to the target mask."

Expected follow-up answer:

"Dice coefficient measures overlap between the predicted mask and ground truth mask. It is common for segmentation because accuracy can be misleading when the foreground is much smaller than the background."

Key formula:

$$
Dice = \frac{2|A \cap B|}{|A| + |B|}
$$

### 6. Workflow And Ownership

Say:

"I structured the solution as a command-line runner rather than one long notebook-style script. That makes each part reproducible and easier to test separately. The README documents the run commands, dependencies, and which tasks are locally reproducible. For fair AI use, I can explain each function, why each library is used, and what assumptions or limitations remain."

Mention limitations honestly:

- The Windows workspace can run the local experiments.
- Cluster-only tasks such as DAWNBench timing or OASIS cluster data cannot be honestly claimed from this local machine unless they were actually run on Rangpur or the required cluster dataset.
- Reported metrics should only be claimed after the corresponding command has actually been executed.

## Marking Scheme Checklist

### Functional Code: 20%

- Can I run each implemented part from the command line?
- Can I show plots or printed metrics for each part?
- Can I explain what inputs and outputs each section has?
- Can I identify which tasks are not locally reproducible and why?

### Understanding And Ownership: 40%

- Can I explain Fourier harmonics and why square waves need odd harmonics?
- Can I explain direct DFT versus FFT complexity?
- Can I explain PCA and why it is used before Random Forest?
- Can I explain CNN layers, activation functions, loss, and evaluation?
- Can I explain U-Net skip connections and Dice coefficient?
- Can I explain my own code without reading it line by line?

### Programming Practice And Documentation: 20%

- Is the code split into functions with clear responsibilities?
- Are dependencies listed in `requirements.txt` or the README?
- Are command-line arguments used safely for selecting experiments?
- Are errors clear, such as missing U-Net masks or missing data root?
- If AI helped, can I describe what I checked and what I personally understand?

### Summary And Context: 20%

- Can I summarise what was completed in under 3 minutes?
- Can I connect each task to the course learning objectives?
- Can I explain what the results mean in computer science or AI terms?
- Can I discuss limitations, assumptions, and next improvements?

## Likely Demonstrator Questions

### General Code Questions

**Q: Why did you use command-line arguments?**

A: To make each experiment independently runnable. Training sections can take time, so requiring `--part` prevents accidentally running everything.

**Q: Why are TensorFlow imports inside functions instead of at the top?**

A: TensorFlow is only needed for the neural network sections. Keeping the import local lets the DFT section run even if TensorFlow is not installed.

**Q: What does `astype(np.float32)` do?**

A: It converts arrays to 32-bit floating point, which is standard for numerical and neural network work because it uses less memory than 64-bit floats and is usually enough precision.

**Q: How would you improve the project if you had more time?**

A: I would save plots and trained model outputs, add fixed seeds for TensorFlow, add a test mode with smaller samples, log metrics to files, and run the cluster-only tasks in the correct environment.

## Recognition Task Question Bank

### VAE Questions

**Q: What is the purpose of a VAE?**

A: A VAE learns a compact probabilistic latent representation of MRI images and can decode latent samples into new brain-slice images. Unlike an ordinary autoencoder, it regularises the latent space so nearby points produce meaningful outputs.

**Q: What does the encoder output?**

A: It outputs a latent mean, `z_mean`, and a latent log variance, `z_log_var`. These define the Gaussian distribution from which the latent vector is sampled.

**Q: Why use the reparameterisation trick?**

A: Direct random sampling would block gradient propagation. I write the sample as $z = \mu + \sigma\epsilon$, where $\epsilon$ is random noise, so gradients can still flow through the mean and variance.

**Q: What are the two VAE loss terms?**

A: Reconstruction loss measures how close the decoded image is to the input. KL divergence regularises the latent distribution toward a standard normal distribution. The total loss is reconstruction loss plus KL loss.

**Q: How did you create the manifold?**

A: Because the latent dimension is 2, I created a grid of coordinates between -2.5 and 2.5, decoded every coordinate, and tiled the resulting images into one manifold figure.

**Q: What evidence proves your VAE task is complete?**

A: Slurm job `598512` completed with exit code `0:0` on an NVIDIA A100. The saved evidence is `demo_outputs/vae/vae_latent_manifold.png`, `vae_loss.csv`, `encoder.keras`, and `decoder.keras`. The manifold image is the required visualisation.

**Q: What does the VAE loss value prove?**

A: It proves that training completed and produced a measurable objective, but the manifold visualisation is also necessary to judge whether the latent space produces meaningful images. A loss value alone does not prove realism.

### OASIS Multi-class U-Net Questions

**Q: Why is this U-Net different from the binary practice U-Net?**

A: The practice U-Net has one sigmoid output for foreground versus background. The OASIS U-Net has four softmax channels because each pixel belongs to one of four mutually exclusive labels.

**Q: Why use softmax?**

A: Softmax converts the four output scores at each pixel into probabilities that sum to one. `argmax` then chooses the most likely class for the final discrete segmentation map.

**Q: Why one-hot encode the masks?**

A: Each integer mask label becomes a four-element vector. This matches the four-channel softmax output and makes categorical cross-entropy appropriate.

**Q: Why is pixel accuracy not enough?**

A: Background pixels can dominate the image. A model can have high accuracy while missing smaller foreground tissues. The task therefore requires a separate discrete DSC for labels 1, 2, and 3.

**Q: How is DSC calculated?**

A: For each label, I compare the hard predicted mask with the ground-truth mask using $DSC = 2|A \cap B|/(|A|+|B|)$. I average the per-image scores over the held-out test set.

**Q: What does the completed U-Net run prove?**

A: It proves that OASIS data loading, the four-class architecture, one-hot training, A100 execution, held-out inference, discrete DSC evaluation, model saving, and prediction visualisation all work.

**Q: What were your latest U-Net DSC values?**

A: The latest recorded values were Label 1 `0.8201`, Label 2 `0.8588`, and Label 3 `0.8842`. They improved over the earlier run, but they are still below the required `>0.9` for every foreground label.

**Q: Can you claim full Task 2 marks?**

A: No. I can claim a completed and working pipeline with partial evidence, but full Task 2 credit requires all three foreground DSC values to exceed `0.9`.

**Q: Why did you add Dice and Tversky-style losses?**

A: Cross-entropy and accuracy can be dominated by background pixels. Foreground Dice and Tversky-style terms focus optimization on tissue overlap and missed foreground pixels. This is a justified improvement, but it must be evaluated using a new held-out test result.

**Q: What U-Net evidence should you show?**

A: Show the Slurm log proving A100 execution, `test_dsc.txt`, `segmentation_predictions.png`, `oasis_unet.keras`, and `unet_training_loss.csv`. Also explain that the test set was not used for training.

### GAN Questions

**Q: What are the two GAN networks?**

A: The generator maps a random latent vector to a synthetic MRI slice. The discriminator receives real or generated slices and predicts whether each is real.

**Q: How are the networks trained?**

A: The discriminator learns from real targets of one and fake targets of zero. The generator is trained so its generated images receive the target one from the discriminator. These updates alternate for every batch.

**Q: Why use `tanh` in the generator?**

A: The real images are scaled to the range [-1, 1], matching the generator's `tanh` output. Consistent scaling helps stabilize adversarial training.

**Q: What is mode collapse?**

A: Mode collapse occurs when the generator produces very similar images for different latent vectors. The samples may look superficially realistic but lack diversity.

**Q: Why save fixed-noise grids?**

A: The same latent vectors are decoded after each epoch, so changes in the generated images reflect training rather than different random inputs.

**Q: What proves the GAN task is complete?**

A: I need a completed OASIS run, generator checkpoints, per-epoch generated grids, a final generated grid, loss values and curves, and visual evidence that the slices are realistic and varied. The instructor judges realism, so losses alone are insufficient.

**Q: Can you currently claim full GAN marks?**

A: No. The GAN implementation and artifact-saving code exist, but I have not yet produced a completed OASIS GAN run and realism evidence.

### Evidence And Marks Questions

**Q: What is currently fulfilled?**

A: The VAE task is fulfilled with a completed A100 run and saved 2D manifold. Earlier DFT, tensor DFT, LFW/CNN, eigenfaces, and binary practice U-Net results are also verified.

**Q: What is still missing for Medium?**

A: The OASIS U-Net must produce DSC above `0.9` for labels 1, 2, and 3 on the held-out test set. The current values are below that threshold.

**Q: What is still missing for Hard?**

A: Medium must first be achieved, then the GAN must produce realistic, varied OASIS brain slices with saved losses, generated grids, and checkpoints.

**Q: Why can’t you claim a mark from implemented code?**

A: The rubric requires functional results and evidence. Code structure shows the intended method, but only a completed run on the required data can prove the metric and visualisation requirements.

**Q: What would you show if a job fails during the demo?**

A: I would show the last verified log and explain the failure accurately. I would identify whether it was a cluster resource issue, environment issue, memory issue, or model issue rather than claiming an unverified result.

## Expanded Oral Questions: Parts 1-4

Use these longer answers when the demonstrator asks you to explain a concept in detail.

### Part 1: Fourier Transform And DFT

**Q: What is a Fourier transform?**

A: A Fourier transform changes the representation of a signal from the time or spatial domain into the frequency domain. Instead of asking how the signal changes over time, I ask which sinusoidal frequencies are present and how strongly each contributes. A large magnitude at a frequency means that frequency is important in the signal. For sampled data, the discrete Fourier transform, or DFT, is the finite version of this idea:

$$
X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}.
$$

Here, $x_n$ is a signal sample, $k$ is a frequency bin, and $X_k$ is the complex coefficient for that bin. The magnitude $|X_k|$ describes the strength of the frequency and the phase describes its alignment.

**Q: What is the difference between DFT and FFT?**

A: The DFT is the mathematical transform. My `naive_dft` function computes it directly with two loops: one over frequency bins and one over samples. That costs $O(N^2)$. The FFT is an efficient algorithm for computing the same DFT, using structure and reuse in the calculation to achieve approximately $O(N\log N)$. FFT changes the computation speed, not the mathematical result.

**Q: What does your Fourier graph show for 1, 3, 5, 20, or 50 harmonics?**

A: The original square wave has sharp transitions, but a finite sum of smooth sine waves can only approximate those transitions. With one harmonic, the graph looks like a single sine wave and is a poor approximation. With 3 or 5 odd harmonics, the flat sections become more square-like and the transitions become sharper. With 20 and 50 harmonics, the approximation is much closer to the square wave. Near each discontinuity there is still overshoot and ringing, called the Gibbs phenomenon. Adding more harmonics narrows the ringing region but does not completely remove the overshoot.

The code uses odd harmonics because the symmetric square wave has only odd sine terms. The coefficient decreases approximately as $1/k$, so higher harmonics add fine detail but contribute less amplitude.

**Q: What does the frequency spectrum graph tell you?**

A: The spectrum has strong peaks at the fundamental frequency and its odd multiples. That matches the Fourier-series construction: the square wave was built from odd sine harmonics. The peak magnitudes decrease as the harmonic number increases. This graph verifies that the time-domain square wave contains the expected frequency components.

**Q: What does the naive DFT code do?**

A: For each output frequency bin, it multiplies every input sample by a complex exponential for that frequency and adds the contributions. The complex exponential acts like a reference sinusoid. If the signal contains that frequency, the terms add constructively and the magnitude is large; otherwise they largely cancel. The nested loops make the implementation clear but computationally expensive.

**Q: What are the three timing methods and which is fastest?**

A: The three methods are direct tensor DFT on the CPU, direct tensor DFT on the GPU, and the built-in TensorFlow FFT. For larger inputs, the usual order is:

1. Fastest: `tf.signal.fft`, because it uses approximately $O(N\log N)$ work.
2. Second: tensor DFT on the A100 GPU, because the GPU parallelises the matrix operations, although the algorithm is still $O(N^2)$.
3. Slowest: tensor DFT on the CPU, because it performs the same quadratic work with much less parallel hardware.

For example, at $N=2048$ in the recorded A100 run, CPU tensor DFT was `0.0346 s`, GPU tensor DFT was `0.0060 s`, and FFT was `0.0026 s`.

**Q: Why can the GPU be slower for small N?**

A: Small inputs may not contain enough work to keep the GPU busy. GPU kernel-launch, memory-transfer, and initialization overhead can be larger than the computation itself. CPU execution may also benefit from lower launch overhead. As $N$ grows, the parallel matrix work dominates the overhead and the GPU advantage becomes clearer. This is why individual small-size timings can be non-monotonic.

### Parts 2 And 3.1: PCA, Random Forest, And CNN

**Q: What is PCA and why use it?**

A: Principal component analysis is an unsupervised dimensionality-reduction method. It finds orthogonal directions that capture the greatest variance in the training data. I use PCA because flattened face images have many pixel features, and a smaller feature representation can reduce computation, noise, and storage. The trade-off is that PCA preserves variance without using identity labels, so the highest-variance directions are not guaranteed to be the best class-separating directions.

**Q: Explain the PCA process step by step.**

A: First, I flatten each grayscale image into one feature vector. Second, I fit PCA on the training vectors only, so test information does not leak into the model. Third, PCA centres the training data around its mean face and finds orthogonal principal directions. Fourth, I keep a selected number of components and project both training and test images into that lower-dimensional coordinate system. Finally, the classifier uses those projected coordinates instead of the original pixels.

**Q: What does the compactness plot mean?**

A: The compactness plot shows cumulative explained variance against the number of principal components. It tells me how much of the variation in the training faces is retained as I keep more components. A steep early rise means a small number of components captures a lot of information. When the curve flattens, additional components provide diminishing returns. It helps justify a component count such as 150 because I can see the information retained versus the feature dimension.

**Q: What is a Random Forest?**

A: A Random Forest is an ensemble of decision trees. Each tree learns rules for separating classes, while random subsets of data and features make the trees diverse. Their predictions are combined, usually by majority vote for classification. In this project, the forest receives PCA face features and predicts the person's identity. It provides a classical non-neural baseline.

**Q: Why was PCA plus Random Forest less accurate than the CNN?**

A: The PCA projection is label-blind: it preserves general image variance, not necessarily the variation that separates people. The Random Forest then works on this compressed representation. The CNN learns convolutional features directly from the images and optimizes them for the identity labels, so it can learn task-specific facial patterns. In my recorded run, PCA plus Random Forest reached `55.28%`, while the CNN reached `72.67%` test accuracy. The comparison also has experimental differences, so I describe it as evidence that the CNN performed better in this run, not as a universal rule.

**Q: What is a CNN?**

A: A convolutional neural network learns spatial features from grid-shaped data such as images. A convolution applies learned local filters across the image. Early filters can detect edges and textures; deeper filters combine them into more complex patterns. Pooling reduces spatial resolution and computation while retaining strong local responses. Dense layers combine the learned features, and softmax produces one probability per identity.

**Q: Explain how your CNN is trained.**

A: The images are split into training and test sets, and a validation portion is taken from the training data. The grayscale channel dimension is added before the images enter the model. The model performs a forward pass, producing class probabilities. Sparse categorical cross-entropy compares those probabilities with integer identity labels. Adam updates the weights using gradients from the loss. This repeats over batches and epochs. Validation accuracy monitors generalization, and the untouched test set is evaluated after training.

**Q: What do ReLU, pooling, dropout, and softmax do?**

A: ReLU, defined as $\max(0,x)$, adds non-linearity and keeps positive activations. Max pooling reduces spatial dimensions and retains the strongest local response. Dropout randomly disables some units during training to reduce overfitting. Softmax converts the final class scores into probabilities that sum to one.

### Part 3.2: ResNet-18 And DAWNBench

**Q: What is ResNet-18?**

A: ResNet-18 is an 18-layer residual convolutional network. It contains a convolutional stem followed by four stages of residual blocks. Each residual block learns a transformation and adds the original input through a shortcut connection. The final global average pooling and dense layer produce class probabilities. My CIFAR-style version uses a small 3-by-3 stem without the large-image initial max-pooling layer.

**Q: How is ResNet-18 trained?**

A: CIFAR-10 images are normalized to the range 0 to 1 and split into training and test data. For each batch, the network performs a forward pass, computes sparse categorical cross-entropy against the integer class labels, and backpropagates gradients through the convolutional and residual layers. Adam updates the weights. Validation is monitored during `fit`, and final test loss and accuracy are measured afterward. Mixed precision can reduce memory use and accelerate matrix operations on the A100, while the final output remains float32 for stable probabilities.

**Q: What are the skip connections in ResNet-18?**

A: A residual skip connection carries the block input directly to the block output. If the dimensions match, it is an identity path. If the number of filters or spatial size changes, a 1-by-1 convolution with the appropriate stride projects the shortcut to the new shape. The main path and shortcut are added before the final ReLU.

**Q: How is a ResNet skip connection different from a U-Net skip connection?**

A: ResNet adds the shortcut feature map element-by-element inside a block. U-Net concatenates encoder features with decoder features at matching spatial scales. ResNet skips help optimization and gradient flow through depth. U-Net skips primarily preserve high-resolution spatial information needed for accurate pixel boundaries.

**Q: What is the difference between ResNet-18 and the CNN face classifier?**

A: Both use convolution and pooling-like spatial processing, but the face CNN is a small sequential classifier with two convolution layers and dense layers. ResNet-18 is deeper and uses residual blocks, allowing information and gradients to pass through shortcut paths. ResNet-18 is designed for CIFAR-10 image classification, while the face CNN is designed for the smaller LFW identity task.

**Q: What is DAWNBench evidence?**

A: I would need a completed Rangpur GPU run showing the ResNet-18 training time and final CIFAR-10 accuracy, then compare them with the task targets. My current documentation says this component is implemented but not yet run, so I must not claim its timing or accuracy yet.

### Part 4: Uses And Model Selection

**Q: What are practical uses of a VAE?**

A: VAEs can learn compact representations, generate new samples, interpolate between images, detect unusual examples, and support semi-supervised learning. For MRI data, the latent space can help explore variations in brain appearance, but generated images still need clinical and visual validation.

**Q: What are practical uses of a U-Net?**

A: U-Net is useful for medical image segmentation, cell segmentation, satellite-image masks, and other pixel-level labeling problems. It is especially useful when precise boundaries matter because skip connections retain fine spatial information.

**Q: What are practical uses of a GAN?**

A: GANs can generate synthetic training examples, support data augmentation, learn image distributions, and create realistic images. In medical imaging, synthetic data must be checked carefully for artifacts, privacy concerns, and mode collapse.

**Q: When would you choose PCA, a CNN, a VAE, or a U-Net?**

A: I would choose PCA when dimensionality reduction or a compact classical feature representation is needed. I would choose a CNN for supervised image classification. I would choose a VAE for a structured latent space and probabilistic generation. I would choose a U-Net when the output must label every pixel rather than assign one label to the whole image.

### DFT Questions

**Q: What is the difference between time domain and frequency domain?**

A: Time domain shows how the signal changes over time. Frequency domain shows which frequencies are present and their strengths.

**Q: Why does the direct DFT take longer than FFT?**

A: Direct DFT uses two nested loops over all samples, so it performs about $N^2$ operations. FFT reuses structure in the computation and reduces the cost to about $N \log N$.

**Q: Why do you compare `naive_dft` with `np.fft.fft`?**

A: It validates that my direct implementation matches a trusted library implementation, while also showing the performance difference.

**Q: What does `np.fft.fftfreq` give you?**

A: It gives the frequency value associated with each FFT bin, based on the sampling interval.

### LFW And PCA Questions

**Q: Why split into training and testing data?**

A: Training data fits the model. Testing data estimates performance on unseen examples, which is needed to check generalisation.

**Q: Why use stratified splitting?**

A: It keeps class proportions similar in the train and test sets, which matters when identities may have different numbers of images.

**Q: What does `PCA(n_components=...)` control?**

A: It controls how many principal components are kept. More components preserve more information but increase computation and may include more noise.

**Q: Why use a Random Forest baseline?**

A: It gives a non-neural comparison point. If the CNN performs better, I can explain that learned spatial features may help compared with PCA features.

### CNN Questions

**Q: What does a convolutional layer learn?**

A: It learns small spatial filters that detect patterns such as edges, corners, textures, and later more complex features.

**Q: Why use ReLU?**

A: ReLU introduces non-linearity and is computationally simple. Without non-linear activation functions, stacked layers would collapse into a mostly linear model.

**Q: Why use dropout?**

A: Dropout randomly disables some activations during training, which can reduce overfitting by preventing the model from relying too heavily on specific neurons.

**Q: What would overfitting look like in your plots?**

A: Training accuracy would keep improving while validation accuracy stalls or gets worse.

### U-Net Questions

**Q: What is the difference between classification and segmentation?**

A: Classification predicts one label for an image. Segmentation predicts a label for each pixel.

**Q: Why does U-Net use skip connections?**

A: Downsampling loses spatial detail. Skip connections pass high-resolution encoder features to the decoder so the output mask can be more precise.

**Q: Why is the final activation sigmoid?**

A: This is binary segmentation, so each pixel needs a probability between 0 and 1 for foreground membership.

**Q: Why use binary cross-entropy?**

A: Each pixel is treated as a binary prediction: foreground or background.

**Q: Why threshold predictions at 0.5 for display?**

A: The model outputs probabilities. Thresholding converts them into a binary mask for visual comparison with the ground truth.

## Code Walkthrough Prompts

Practice explaining the code as a flow from the first line that runs to the final output. The demonstrator may ask you to point at a function and explain why it exists, what input it receives, what output it creates, and how that helps reach the lab goal.

## Full Code Flow From Start To Goal

### 1. Program Entry Point

Start at the bottom of the file:

```python
if __name__ == "__main__":
	main()
```

Say:

"This means the script only runs `main()` when the file is executed directly from the terminal. If another file imports functions from this file, it does not automatically start training or plotting. That is good software practice because it makes the code reusable."

Then move to `main()`:

```python
def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--part", choices=("dft", "lfw", "unet"), required=True)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--data-root", type=Path)
```

Say:

"The program starts by building a command-line interface. The `--part` argument is required, so the user must choose which lab section to run. This avoids running all experiments at once, which would be slow and confusing. `--epochs` controls training length for neural network parts. `--data-root` is only needed for U-Net because that section must load local PNG image and mask folders."

Then explain the branch:

```python
if args.part == "dft":
	run_dft()
elif args.part == "lfw":
	run_lfw(args.epochs)
elif args.data_root is None:
	parser.error("--data-root is required for --part unet")
else:
	run_unet(args.data_root, args.epochs)
```

Say:

"This is the main control flow. It sends execution to the correct experiment. The U-Net path checks for `--data-root` before training, because without the dataset path it cannot load images or masks. This connects to the goal of making the work reproducible: every experiment has a clear command and clear inputs."

### 2. DFT Flow: Signal Theory To Frequency Output

Command:

```powershell
python lab2_solution.py --part dft
```

Code path:

```text
main() -> run_dft() -> square_wave()
				  -> square_wave_fourier()
				  -> naive_dft()
				  -> np.fft.fft()
				  -> plots and timing output
```

Step 1: `run_dft()` creates sample points.

```python
sample_count = 2048
duration = 1.0
frequency = 1.0
times = np.linspace(0.0, duration, sample_count, endpoint=False)
```

Say:

"The code creates 2048 evenly spaced time samples across one second. `endpoint=False` prevents duplicating the start of the next period. The goal is to create a clean sampled signal that can be analysed with Fourier methods."

Step 2: `square_wave()` creates the original wave.

```python
return np.sign(np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)
```

Say:

"This generates a sine wave and then takes its sign. Positive sine values become 1, negative values become -1, so the smooth sine becomes a square wave. The result is converted to `float32` for efficient numerical work."

Step 3: `square_wave_fourier()` reconstructs the wave from harmonics.

```python
for harmonic_index in range(harmonics):
	harmonic = 2 * harmonic_index + 1
	result += np.sin(2 * np.pi * harmonic * f0 * t) / harmonic
return (4 / np.pi) * result
```

Say:

"This loop adds only odd harmonics: 1, 3, 5, 7, and so on. Each harmonic is divided by its harmonic number, so higher frequencies contribute less. Multiplying by $4/\pi$ gives the correct square-wave Fourier series scale. As the number of harmonics increases, the reconstruction becomes closer to the original square wave."

Connect to theory:

"This demonstrates that a complex-looking discontinuous signal can be represented as a sum of simpler sine waves. The ringing near sharp jumps is expected and is called the Gibbs phenomenon."

Step 4: `naive_dft()` computes the direct DFT.

```python
for frequency_bin in range(sample_count):
	for sample_index in range(sample_count):
		result[frequency_bin] += signal[sample_index] * np.exp(
			-2j * np.pi * frequency_bin * sample_index / sample_count
		)
```

Say:

"The outer loop chooses a frequency bin. The inner loop sums the contribution of every time sample to that frequency. The complex exponential is the basis wave for that frequency. This directly implements the DFT formula, so it is easy to understand mathematically, but it is slow because it uses nested loops."

Goal reached:

"The DFT section reaches the goal by showing the square wave in time, showing how Fourier harmonics reconstruct it, timing direct DFT against FFT, checking they agree, and plotting the magnitude spectrum. The expected result is that FFT is much faster, while both methods produce matching frequency results."

### 3. LFW Flow: Images To Classical ML And CNN Classification

Command:

```powershell
python lab2_solution.py --part lfw --epochs 1
```

Code path:

```text
main() -> run_lfw(epochs)
		-> fetch_lfw_people()
		-> train_test_split()
		-> PCA features
		-> RandomForestClassifier
		-> CNN input preparation
		-> TensorFlow model training
		-> test evaluation and accuracy plot
```

Step 1: Load the LFW face dataset.

```python
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
images = faces.images.astype("float32")
labels = faces.target
```

Say:

"This loads labelled face images where each target is a person identity. `min_faces_per_person=70` keeps only people with enough examples, which makes training and testing more reliable. `resize=0.4` reduces image size so the experiment is faster."

Step 2: Split into train and test sets.

```python
x_train, x_test, y_train, y_test = train_test_split(
	images, labels, test_size=0.25, random_state=42, stratify=labels
)
```

Say:

"The model learns from the training set and is evaluated on the test set. The split is stratified so each person is represented proportionally in both sets. `random_state=42` makes the split reproducible."

Step 3: Flatten images and use PCA.

```python
pca = PCA(n_components=min(150, x_train.shape[0] - 1), svd_solver="randomized", random_state=42)
x_train_pca = pca.fit_transform(x_train.reshape(len(x_train), -1))
x_test_pca = pca.transform(x_test.reshape(len(x_test), -1))
```

Say:

"PCA needs each image as a feature vector, so the 2D image is flattened. PCA learns the main directions of variation from the training data only, then applies the same transformation to the test data. This avoids leaking test information into training. The reduced features are similar to eigenface-style representations."

Step 4: Train and evaluate Random Forest.

```python
forest = RandomForestClassifier(
	n_estimators=150, max_depth=15, max_features="sqrt", random_state=42, n_jobs=-1
)
forest.fit(x_train_pca, y_train)
forest_predictions = forest.predict(x_test_pca)
```

Say:

"The Random Forest is a classical machine learning baseline. It trains many decision trees and combines their predictions. I use the PCA features rather than raw pixels to reduce dimensionality and noise. The printed accuracy and classification report show how well this baseline recognises people."

Step 5: Prepare images for CNN.

```python
x_train_cnn = x_train[..., np.newaxis]
x_test_cnn = x_test[..., np.newaxis]
```

Say:

"TensorFlow convolutional layers expect a channel dimension. The LFW images are grayscale, so `np.newaxis` changes the shape from height by width to height by width by 1. This tells the CNN each image has one channel."

Step 6: Build the CNN.

```python
model = tf.keras.Sequential(
	[
		tf.keras.layers.Input((height, width, 1)),
		tf.keras.layers.Conv2D(32, 3, activation="relu"),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(32, 3, activation="relu"),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation="relu"),
		tf.keras.layers.Dropout(0.3),
		tf.keras.layers.Dense(class_count, activation="softmax"),
	]
)
```

Say:

"The CNN learns spatial features directly from images. Convolution layers learn filters, pooling reduces image size, flatten converts feature maps into a vector, dense layers perform classification, dropout helps reduce overfitting, and softmax outputs a probability for each identity."

Step 7: Compile, train, evaluate, and plot.

```python
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train_cnn, y_train, validation_split=0.15, epochs=epochs, batch_size=32)
print("CNN test metrics:", model.evaluate(x_test_cnn, y_test, verbose=0))
```

Say:

"The loss is sparse categorical cross-entropy because the labels are integer class IDs. The validation split monitors performance during training. The final test evaluation estimates performance on unseen images. The accuracy plot helps reveal whether the model is learning or overfitting."

Goal reached:

"The LFW section reaches the goal by comparing a classical PCA plus Random Forest pipeline with a CNN pipeline. This shows two different approaches to image classification: hand-designed dimensionality reduction followed by a traditional classifier, versus learned spatial features in a neural network."

### 4. U-Net Flow: PNG Images To Segmentation Masks

Command:

```powershell
python lab2_solution.py --part unet --data-root "C:/Users/icha/Downloads/keras_png_slices_data/keras_png_slices_data" --epochs 1
```

Code path:

```text
main() -> run_unet(data_root, epochs)
		-> load_png_split(train/validate/test)
		-> unet_model(input_shape)
		-> dice_coefficient metric
		-> model training
		-> test evaluation
		-> prediction visualisation
```

Step 1: Load train, validate, and test splits.

```python
x_train, y_train = load_png_split(data_root, "train")
x_validate, y_validate = load_png_split(data_root, "validate")
x_test, y_test = load_png_split(data_root, "test")
```

Say:

"The U-Net section needs paired images and segmentation masks. The train set teaches the model, the validation set monitors learning during training, and the test set evaluates final performance."

Step 2: `load_png_split()` finds image and mask folders.

```python
image_dir = root / f"keras_png_slices_{split}"
mask_dir = root / f"keras_png_slices_seg_{split}"
image_paths = sorted(image_dir.glob("case_*.png"))
```

Say:

"For each split, the code expects one folder of input images and one folder of segmentation masks. Sorting the image paths makes the loading order deterministic."

Step 3: Pair each image with its mask.

```python
mask_path = mask_dir / image_path.name.replace("case_", "seg_", 1)
if not mask_path.exists():
	raise FileNotFoundError(f"Missing mask for {image_path.name}")
```

Say:

"Each case image should have a matching segmentation image. If a mask is missing, the program stops with a clear error instead of silently training on incorrect pairs. This is important because segmentation depends on exact image-mask alignment."

Step 4: Convert PNGs to normalised arrays.

```python
images.append(np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0)
masks.append(np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0)
return np.asarray(images)[..., np.newaxis], np.asarray(masks)[..., np.newaxis]
```

Say:

"Images are converted to grayscale, changed to float arrays, and normalised from 0 to 255 into 0 to 1. The final `np.newaxis` adds the single grayscale channel dimension required by Keras."

Step 5: Build the U-Net model.

```python
inputs = tf.keras.Input(input_shape)
```

Say:

"The model input shape comes from the training images, so the network matches the dataset shape rather than hard-coding it."

Then explain the convolution block:

```python
def block(tensor, filters):
	tensor = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(tensor)
	tensor = tf.keras.layers.BatchNormalization()(tensor)
	tensor = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(tensor)
	return tensor
```

Say:

"This helper block applies two convolutions with ReLU. Batch normalisation helps stabilise training. `padding="same"` keeps the spatial size consistent inside the block, which is useful for connecting encoder and decoder features."

Step 6: Explain encoder, bottleneck, decoder.

```python
encoder1 = block(inputs, 32)
pooled1 = tf.keras.layers.MaxPooling2D()(encoder1)
encoder2 = block(pooled1, 64)
pooled2 = tf.keras.layers.MaxPooling2D()(encoder2)
bottleneck = block(pooled2, 128)
```

Say:

"The encoder gradually reduces spatial resolution while increasing the number of filters. This lets the model learn more abstract features. The bottleneck is the deepest part of the network, where the model has compressed information about the whole image."

Then:

```python
decoder2 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(bottleneck)
decoder2 = tf.keras.layers.Concatenate()([decoder2, encoder2])
decoder2 = block(decoder2, 64)
decoder1 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same")(decoder2)
decoder1 = tf.keras.layers.Concatenate()([decoder1, encoder1])
decoder1 = block(decoder1, 32)
```

Say:

"The decoder upsamples back toward the original image size. The concatenate operations are skip connections. They combine detailed encoder features with decoder features, helping the model produce sharper masks."

Step 7: Explain the output layer.

```python
outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(decoder1)
```

Say:

"The output has one channel because this is binary segmentation. Sigmoid turns each pixel output into a probability between 0 and 1."

Step 8: Explain Dice coefficient.

```python
intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
return tf.reduce_mean((2.0 * intersection + 1e-6) / (denominator + 1e-6))
```

Say:

"Dice measures overlap between prediction and ground truth. Multiplying `y_true` and `y_pred` gives the intersection. The denominator is the total predicted and true mask area. The small `1e-6` prevents division by zero and makes the metric stable for empty or near-empty masks."

Step 9: Compile, train, test, and visualise.

```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[dice_coefficient])
history = model.fit(
	x_train, y_train, validation_data=(x_validate, y_validate), epochs=epochs, batch_size=16
)
print("Test metrics:", model.evaluate(x_test, y_test, verbose=0))
predictions = model.predict(x_test[:4], verbose=0)
```

Say:

"Binary cross-entropy trains each pixel as foreground or background. Dice is used as an interpretable segmentation metric. After training, the model evaluates on the test set and predicts masks for the first four test images."

Final visualisation:

```python
axes[row, 0].imshow(x_test[row, ..., 0], cmap="gray")
axes[row, 1].imshow(y_test[row, ..., 0], cmap="gray")
axes[row, 2].imshow(predictions[row, ..., 0] > 0.5, cmap="gray")
```

Say:

"The visualisation shows the input image, the ground truth mask, and the predicted mask. The prediction is thresholded at 0.5 to convert probabilities into a binary mask. This reaches the practical goal of checking not just a number, but whether the segmentation output visually matches the target."

Goal reached:

"The U-Net section reaches the goal by loading paired medical images and masks, training a pixel-level model, measuring overlap with Dice, and displaying predictions. This connects deep learning theory to a practical segmentation task."

## Full Demo Speaking Script

Use this when you want a complete practice run.

"I start the program from the command line by choosing a required `--part`. This sends the program into one of three flows: DFT, LFW classification, or U-Net segmentation. I designed it this way so each experiment can be run separately and reproduced with a clear command.

For the DFT part, the code samples a one-second signal, creates a square wave, reconstructs it using odd Fourier harmonics, and then compares my direct DFT implementation with NumPy FFT. The direct DFT follows the mathematical formula by summing every sample against every frequency bin, so it is understandable but slow at $O(N^2)$. FFT gives the same result much faster, around $O(N \log N)$. This part reaches the goal by showing both the theoretical Fourier reconstruction and the practical computational advantage of FFT.

For the LFW part, the code loads labelled face images and splits them into train and test sets. First it uses PCA to reduce each flattened image into a smaller feature vector, then trains a Random Forest classifier as a classical machine learning baseline. After that, it trains a CNN directly on the image shape. The CNN uses convolution and pooling to learn spatial features, then dense layers and softmax for identity classification. This part reaches the goal by comparing traditional feature extraction with neural network feature learning.

For the U-Net part, the code loads image-mask pairs from train, validation, and test folders. It checks that every image has a matching mask, normalises the PNGs, and adds a channel dimension for Keras. The U-Net has an encoder that downsamples, a bottleneck that learns compressed features, and a decoder that upsamples. Skip connections bring spatial detail from the encoder into the decoder. The output is a sigmoid probability mask, trained with binary cross-entropy and evaluated with Dice coefficient. This part reaches the goal by producing pixel-level segmentation predictions and comparing them with ground truth masks.

Overall, the flow of the code goes from command-line selection, to data creation or loading, to model or algorithm execution, to evaluation and visualisation. That matches the lab objective because it turns the theoretical material into working computational experiments that I can run, inspect, explain, and critique."

## Function By Function Quick Reference

| Function | Input | Main job | Output or result | Why it matters |
| --- | --- | --- | --- | --- |
| `square_wave` | Time samples and frequency | Converts a sine wave into -1/1 values | Square-wave samples | Creates the signal for Fourier analysis |
| `square_wave_fourier` | Time samples, frequency, harmonic count | Adds odd sine harmonics | Approximate square wave | Demonstrates Fourier series reconstruction |
| `naive_dft` | 1D signal | Computes DFT with nested loops | Complex frequency bins | Shows the direct mathematical transform |
| `run_dft` | No argument | Runs plots, timing, DFT, FFT comparison | Figures and printed timings | Completes the signal-processing demo |
| `run_lfw` | Epoch count | Trains PCA plus Random Forest and CNN | Accuracy, report, plots | Completes the face-classification demo |
| `load_png_split` | Dataset root and split name | Loads image-mask PNG pairs | Image and mask arrays | Prepares segmentation data correctly |
| `dice_coefficient` | True and predicted masks | Measures mask overlap | Mean Dice score | Evaluates segmentation quality |
| `unet_model` | Input image shape | Builds encoder-decoder segmentation model | Keras model | Defines the segmentation architecture |
| `run_unet` | Dataset root and epoch count | Trains and evaluates U-Net | Metrics and prediction plots | Completes the segmentation demo |
| `main` | Command-line arguments | Chooses which experiment to run | Calls selected part | Makes the project reproducible and organised |

## Short Final Summary

"Overall, this lab helped me connect theory with implementation. The DFT section shows how signals can be represented by frequencies and why FFT matters computationally. The face recognition section compares classical dimensionality reduction and machine learning with CNN feature learning. The U-Net section shows how deep learning can be adapted from image classification to pixel-level prediction. I can run the implemented parts, explain the code structure, and describe the limitations of results that require cluster resources."

## Actual Results From My Runs (Real Numbers, Use These Live)

These are genuine outputs from commands I actually executed on this machine. Use these instead of guessing, and point to the matching saved figure in `lab2/`.

### DFT Run

Command: `python lab2_solution.py --part dft`

```text
Naive DFT: 7.751078 s
NumPy FFT: 0.019670 s
Results agree: True
```

Saved figures: [lab2/Figure_1(dft).png](lab2/Figure_1(dft).png) and [lab2/Figure_2(dft).png](lab2/Figure_2(dft).png)

Say:

"On this machine, the direct DFT took about 7.75 seconds for 2048 samples, while NumPy's FFT took about 0.02 seconds, roughly 400 times faster. Both agree numerically, which confirms my direct implementation is correct; the FFT is only faster, not different in result. That is the practical demonstration of $O(N^2)$ versus $O(N\log N)$: doubling the sample count would roughly quadruple the naive DFT time but barely change the FFT time. Figure 1 shows the square wave and its harmonic reconstruction, and Figure 2 shows the frequency spectrum with peaks at the odd harmonics."

Rangpur venv confirmation, same command: `python lab2_solution.py --part dft`

```text
Naive DFT: 3.667617 s
NumPy FFT: 0.007678 s
Results agree: True
```

Say:

"I also confirmed this NumPy DFT part actually runs inside the fresh `venv` on Rangpur's `login0` node, not just locally. Naive DFT took about 3.67 seconds there versus 0.0077 seconds for FFT, roughly 478 times faster, and the results still agree numerically. The absolute times differ from my Windows machine because the CPUs are different, but the qualitative result is the same: FFT wins by two to three orders of magnitude, and both implementations compute the identical transform. This is one concrete piece of evidence that my Rangpur Python environment, NumPy, and the repository itself are correctly set up before I attempt anything GPU- or OASIS-related there."

### LFW Run (10 epochs)

Command: `python lab2_solution.py --part lfw --epochs 10`

```text
PCA + Random Forest accuracy: 0.5528
CNN test metrics (loss, accuracy): [0.7431, 0.7267]
```

Saved figure: [lab2/Figure_1(lfw_epochs 10).png](lab2/Figure_1(lfw_epochs%2010).png)

Say:

"The PCA plus Random Forest baseline reached 55.3% accuracy on 322 held-out images. Its classification report shows strong class imbalance: George W. Bush has 133 test images and recall 1.00, while Ariel Sharon and Hugo Chavez have recall 0.00. Accuracy alone therefore hides weak minority-class recognition. After 10 epochs, the CNN reached 72.7% test accuracy with test loss 0.7431, improving by 17.4 percentage points over the baseline. In the saved curve, training accuracy rises from about 40% to 77%, while validation accuracy rises from about 39% to 77%. The two curves remain close and validation briefly exceeds training, which is possible because dropout is active during training but disabled during validation. There is no persistent widening gap, so this run shows learning without strong evidence of overfitting. The final test accuracy is lower than the final validation accuracy, which is normal because they are different samples and the validation subset influenced development monitoring."

Follow-up you should be ready for:

"Why did Random Forest fail on minority classes?" — "With only 18-27 training-adjacent images for some identities versus 133 for George W. Bush, the PCA-plus-tree pipeline has little signal to learn minority decision boundaries, and accuracy alone hides this; the per-class precision/recall table exposes it."

"Why is CNN better here?" — "The CNN learns task-specific convolutional features end-to-end instead of relying on PCA's variance-based, label-blind projection, and it directly optimizes the classification objective."

### Manual Eigenfaces Run

Command: `python lab2_solution.py --part eigenfaces`

```text
Total Testing: 322
Total Correct: 195
Accuracy: 0.6055900621118012
```

Saved figures: [lab2/Figure_1(eigenfaces).png](lab2/Figure_1(eigenfaces).png) and [lab2/Figure_2(eigenfaces).png](lab2/Figure_2(eigenfaces).png)

Say:

"For the exact manual PCA version, I centred the training faces, applied NumPy SVD, retained 150 principal components, projected train and test images into that space, and trained a Random Forest. It correctly classified 195 of 322 test images, giving 60.6% accuracy. This is about 5.3 percentage points better than the sklearn-PCA baseline run, although the comparison is not perfectly controlled because the manual run uses a non-stratified split and different Random Forest feature settings. The eigenface gallery visualises the leading principal directions, not individual people. Bright and dark regions show correlated pixel changes involving illumination, face outline, eyes, nose, and mouth. The compactness curve rises quickly: approximately 75% of variance is captured by 20 components, about 90% by roughly 85 components, and about 95% by 150. This shows diminishing returns: later components add progressively less information. The class report still shows imbalance, including zero recall for Ariel Sharon, so 60.6% overall accuracy does not mean every identity is recognised equally well."

Follow-up you should be ready for:

"What is an eigenface?" — "It is a principal-component direction reshaped into the original image dimensions. A face is represented by weights describing how much of each eigenface it contains."

"Why centre the data first?" — "PCA should model variation around the mean face. Without centring, the first direction would be dominated by the overall average intensity rather than differences between faces."

"Why keep 150 components?" — "The compactness plot shows that 150 components retain about 95% of the training variance while reducing each face from all original pixels to 150 features."

### Tensor DFT Runs (Local CPU And Rangpur Login CPU)

Command: `python lab2_solution.py --part dft-tensor`

```text
N=256:  CPU tensor naive DFT 0.0998s | GPU unavailable | tf.signal.fft 0.0025s
N=512:  CPU tensor naive DFT 0.0068s | GPU unavailable | tf.signal.fft 0.0019s
N=1024: CPU tensor naive DFT 0.0187s | GPU unavailable | tf.signal.fft 0.0002s
N=2048: CPU tensor naive DFT 0.0374s | GPU unavailable | tf.signal.fft 0.0002s
```

Say:

"The tensor DFT path runs successfully on this Windows machine, but TensorFlow reports no supported native-Windows GPU, so I cannot claim a GPU comparison from this run. For every tested size, `tf.signal.fft` is faster than the matrix-based direct DFT. The direct method forms an $N$ by $N$ transform matrix, so its work and memory grow quadratically, whereas FFT exploits structure for approximately $O(N\log N)$ work. The 256-point direct timing is unusually high because it includes TensorFlow tracing and first-call warm-up; this is why the raw one-off timings are not monotonic. For a fair benchmark I would warm up each function, repeat it several times, report the median, synchronise GPU execution, and run the GPU measurements on Rangpur."

Rangpur login-node command: `python lab2_solution.py --part dft-tensor`

```text
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
N=256:  CPU tensor naive DFT 0.0161s | GPU unavailable | tf.signal.fft 0.0012s
N=512:  CPU tensor naive DFT 0.0037s | GPU unavailable | tf.signal.fft 0.0001s
N=1024: CPU tensor naive DFT 0.0069s | GPU unavailable | tf.signal.fft 0.0002s
N=2048: CPU tensor naive DFT 0.0186s | GPU unavailable | tf.signal.fft 0.0002s
```

Say:

"I also ran the command successfully in my `comp3710` environment on Rangpur's `login0` node. This verifies that the repository, Python environment, NumPy, and TensorFlow installation work on Rangpur. The login node reported `CUDA_ERROR_NO_DEVICE`, which is expected because logging into Rangpur does not automatically allocate a GPU. These are therefore Rangpur CPU timings, not GPU timings. The built-in FFT remained much faster than the direct matrix DFT. The first 256-point measurement is again affected by TensorFlow initialisation and warm-up, so I should not interpret the individual one-off values as a clean scaling curve. To obtain valid GPU timings, I must submit the same command as a Slurm job requesting one GPU and confirm that TensorFlow lists a GPU inside that allocated job."

If asked why Rangpur showed no GPU:

"I ran this test on the shared login node, which is intended for setup and job submission rather than GPU computation. Access to a GPU is provided only inside a scheduled Slurm job after requesting a GPU resource."

### U-Net Run (10 epochs)

Command: `python lab2_solution.py --part unet --data-root "C:/Users/icha/Downloads/keras_png_slices_data/keras_png_slices_data" --epochs 10`

Saved figure: [lab2/Figure_3(unet mri).png](lab2/Figure_3(unet%20mri).png)

Say:

"This figure shows four rows, each with three columns: the input MRI slice, the ground truth mask, and the model's predicted mask thresholded at 0.5, after 10 epochs of training. The ground truth mask is the smooth white brain-tissue region segmented from the dark skull and background. My predicted masks correctly capture the overall brain shape and location in every row, so the model has learned the coarse foreground-versus-background boundary. However, the predicted masks are visibly more fragmented and speckled than the ground truth: the smooth white matter and grey matter regions in the ground truth become a patchier, holey pattern in the prediction, especially in the interior of the brain. That tells me the model gets the outer boundary largely right but has not learned the finer internal structure as cleanly, likely because 10 epochs and a fairly small encoder-decoder is not enough to resolve fine-grained texture, and because this is a binary sigmoid segmentation task so it can only separate foreground from background, not distinguish tissue types within the brain. This is the binary practice pipeline: one sigmoid output channel trained with binary cross-entropy and evaluated with the continuous Dice coefficient, which is different from the OASIS task's four-class discrete Dice requirement."

If you still have the terminal output, state the actual printed test loss/Dice number from `model.evaluate` instead of only this qualitative description.

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

Say:

"This Slurm job was allocated an NVIDIA A100, and TensorFlow confirmed it created `GPU:0` with about 38 GB of usable memory. At 2048 samples, the direct tensor DFT took 0.0346 seconds on the CPU and 0.0060 seconds on the GPU, so GPU parallelism made the same quadratic calculation about 5.8 times faster. TensorFlow's FFT was fastest at 0.0026 seconds because it reduces the algorithmic work from $O(N^2)$ to approximately $O(N\log N)$. For small inputs, startup, kernel-launch, and transfer overhead can be larger than the useful computation, which explains why the 256-point GPU result was slower and why the measurements are not perfectly monotonic. The reliable conclusion is therefore based on the larger input: algorithmic efficiency makes FFT fastest, GPU parallelism accelerates the direct DFT, and CPU direct computation is slowest."

The stderr traceback came from an unnecessary `LD_LIBRARY_PATH` helper expression because `nvidia.cudnn.__file__` was `None`. It did not invalidate the run: TensorFlow subsequently reported `Created device ... GPU:0 ... NVIDIA A100-PCIE-40GB`, and the output contains real GPU timings. Remove that export line from future Slurm scripts because GPU discovery now works without it.

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

**Important honesty note:** the VAE result is completed and supported by the Slurm log and saved artifacts. The OASIS per-label DSC evidence exists, but the scores are below the required threshold. GAN realism evidence is still missing, so do not claim the Medium or Hard levels yet.

### How To Explain The Current U-Net Result

Say:

"The OASIS multi-class U-Net ran successfully on an NVIDIA A100. It uses four softmax output channels, one-hot encoded masks, categorical cross-entropy plus a foreground Dice loss, and `argmax` during inference. The test pipeline saved the model, loss history, and an input/ground-truth/prediction visualisation. The first measured discrete DSC values were 0.7861, 0.8050, and 0.8350 for labels 1, 2, and 3. Those confirm the implementation and evaluation pipeline work, but they are below the task requirement of greater than 0.9, so I am running a longer foreground-aware training configuration before claiming full Task 2 credit."

Explain the limitation:

"Pixel accuracy was high, but accuracy is dominated by background pixels and therefore does not prove good segmentation of the smaller foreground classes. The per-label DSC is the more relevant metric here. I addressed the first limitation by adding foreground Dice loss alongside categorical cross-entropy and increasing the planned training duration; augmentation and validation-based checkpoint selection are further improvements, while the test set remains untouched."

### What This U-Net Run Proves

This completed run proves that:

- The OASIS image and mask folders were found and loaded correctly.
- The model is genuinely multi-class, with four softmax output channels.
- The masks were converted to one-hot labels for categorical cross-entropy.
- Training completed on an NVIDIA A100 rather than the login CPU.
- Inference ran on the held-out test set using `argmax` class predictions.
- Per-label discrete DSC was calculated and saved for labels 1, 2, and 3.
- The trained model, loss history, and prediction visualisation were saved for demonstration.

This run does **not** prove that Task 2 reached full marks, because every foreground DSC must be greater than `0.9`. It is valid evidence of a working implementation and supports partial credit, but the reported values must be stated honestly as below threshold.

### Marking Criteria Self-Check (Section 4.5)

- Code functions and completes tasks, hosted on GitHub, results explained: DFT, LFW/CNN, binary U-Net, manual eigenfaces, the TF-tensor DFT A100 path, the OASIS VAE, and a completed but below-threshold OASIS U-Net run are evidenced. DAWNBench still needs its required run; the U-Net needs higher DSC and the GAN still needs completed realism evidence.
- GitHub practices, README, meaningful commits: verify this repo actually has a README and sensible commit history before claiming this mark; check with `git log --oneline` if unsure.
- Code commented, structured, follows engineering practice: the file stays organised as one function per experiment with local imports, which is a genuine strength to point out.

### Honest One-Line Summary For The Demonstrator

"DFT, the tensor DFT on an A100, LFW PCA/CNN, manual eigenfaces, the binary U-Net warm-up, the OASIS VAE, and a completed OASIS U-Net run have real results. The OASIS U-Net DSC values are below the required threshold, so I currently claim the VAE task and partial U-Net work only. I still need DSC above 0.9 for all foreground labels and GAN realism evidence before claiming the full 7/7."

## Last Minute Practice Checklist

- Run at least one quick command before the demo to make sure the environment works.
- Know where each major function is in `lab2_solution.py`.
- Have one honest limitation ready to discuss.
- Do not claim metrics you have not actually generated.
- When asked about AI use, explain what you understand and what you verified.
- Keep the first summary under 3 minutes, then let the demonstrator ask deeper questions.


QUESTION WILL BE ASK
THEORICTICAL CODE
- does it run
- how the bfs
- how the u turn
- how the gan
- how the dft 
- how the pca
- how the unet

## Additional Likely Theory Questions

### Does the code run?

Say: "The DFT part can be run locally with `python lab2_solution.py --part dft`. The LFW part requires its dataset and the required Python packages. The U-Net part also requires the PNG dataset path, supplied with `--data-root`. I should only claim results from commands I have actually executed."

### How does BFS work?

"Breadth-first search explores a graph level by level. It starts with a source node, marks it visited, and places it in a FIFO queue. It repeatedly removes the oldest node, visits each unvisited neighbour, marks that neighbour, and adds it to the queue. In an unweighted graph, BFS finds shortest paths by number of edges. Its complexity is $O(V+E)$ with an adjacency-list representation."

### What is a U-turn in the algorithm?

"A U-turn means reversing direction or backtracking when the current choice cannot continue toward the goal. In search or navigation code, the algorithm returns to a previous state and tries another available option. It is different from BFS itself: BFS normally avoids revisiting marked nodes, while backtracking explicitly returns to an earlier decision point."

### How does a GAN work?

"A generative adversarial network has two neural networks. The generator creates synthetic samples from random noise, and the discriminator tries to distinguish real samples from generated ones. They train against each other: the discriminator improves its detection, while the generator improves its ability to fool the discriminator. After training, the generator can produce samples resembling the training distribution."

### How does the DFT work?

"The discrete Fourier transform converts a sampled signal from the time domain into frequency components. For each frequency bin, it multiplies every sample by a complex sinusoid and sums the results. Large magnitude means that frequency is strongly present. The direct implementation costs $O(N^2)$, while FFT computes the same transform in about $O(N\log N)$."

### How does PCA work?

"Principal component analysis centres the data and finds orthogonal directions of greatest variance. The data is projected onto the selected principal components, reducing the number of features while retaining as much variation as possible. In the face experiment, flattened images become lower-dimensional eigenface-style features before classification. PCA must be fitted on training data only to avoid test-data leakage."

### How does U-Net work?

"U-Net performs pixel-level segmentation using an encoder-decoder structure. The encoder downsamples the image and learns increasingly abstract features. The decoder upsamples those features to the original resolution. Skip connections concatenate high-resolution encoder features with decoder features, preserving boundaries and spatial detail. A final sigmoid gives a foreground probability for each pixel, which can be thresholded into a binary mask."

## GAN: What You Need To Explain

For the MRI task, explain a **2D GAN**. The training samples are individual 2D MRI slices, so a 3D GAN would be an unnecessary mismatch unless the task explicitly gave 3D volumes.

### GAN Architecture

Say:

"A GAN has a generator and a discriminator. The generator receives a random latent vector $z$ and turns it into a synthetic 2D MRI slice. It normally uses dense layers followed by reshape and transpose convolutions, or upsampling plus convolution, to increase spatial resolution. The discriminator receives either a real slice or a generated slice and outputs a probability that it is real. It normally uses convolutional layers with downsampling, then a final classification layer."

The training objective is a minimax game:

$$
\min_G \max_D\; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]
$$

In practical alternating training:

- Train the discriminator on a batch of real images with target $1$, and generated images with target $0$.
- Freeze the discriminator while updating the generator.
- Train the generator so generated images receive the discriminator target $1$; it is rewarded when the discriminator calls its output real.

### GAN Collapse And Checkpoints

Say:

"GAN optimisation is unstable because the generator and discriminator are learning against each other. The discriminator can become too strong, leaving the generator with poor gradients. The generator can also collapse to producing only a few similar outputs, called mode collapse. I inspect generated samples after each epoch and save a checkpoint once the images begin to show brain-like structure. That preserves a useful model if later epochs degrade."

Reasonable stabilisation steps to mention:

- Normalise real images consistently, commonly to $[-1, 1]$ if the generator uses `tanh` output.
- Use a sensible balance of generator and discriminator learning rates; do not assume more epochs always improve a GAN.
- Save the generator, discriminator, optimiser states, epoch number, losses, and a fixed latent vector for comparable sample grids.
- Diagnose both losses alongside images. Loss values alone do not reliably prove image quality.

### GAN Results To Show

At minimum, show a grid of generated 2D slices that look broadly like brains, not random blocks. Include training curves if available, but explain that visual samples are essential for GAN evaluation. Save the best checkpoint and show the epoch it came from. Do not claim that a low GAN loss alone means the model is good.

## Validation, Testing, And Metrics

### Why Validation Is Not The Test Set

Say:

"I split the provided training data into training and validation subsets. During training, I use validation metrics to monitor convergence and choose hyperparameters or the best checkpoint. I keep the test set untouched until the model and settings are final. Testing happens once at the end, so it is an unbiased estimate of generalisation."

For timing tasks, the reported training time includes validation done during epochs because validation is part of the training workflow. Final test evaluation should be a separate step after training.

### Discrete Multi-Class Dice For OASIS

For each foreground label $c \in \{1, 2, 3\}$, first convert model probabilities to a hard predicted class map using `argmax` across classes. Then calculate Dice for that class:

$$
DSC_c = \frac{2\sum_i [\hat{y}_i=c][y_i=c]}{\sum_i [\hat{y}_i=c] + \sum_i [y_i=c]}
$$

Say:

"The official evaluation is discrete because both the ground truth and final prediction are class labels. I use `argmax` to choose one label per pixel, then compute a separate Dice score for labels 1, 2, and 3 over the test set. Background label 0 is excluded from the stated target. The requirement is an average test DSC above 0.9 for each foreground label, not only one combined score."

Know the architecture consequence:

- Binary practice U-Net: one output channel, sigmoid activation, binary cross-entropy, threshold at $0.5$.
- OASIS four-class U-Net: four output channels, softmax activation, sparse categorical cross-entropy when masks store integer labels, and `argmax` at inference.

Continuous or soft Dice can be a differentiable training loss, but it is not a replacement for the required discrete evaluation metric.

## Rangpur Rehearsal

Be logged in before the demo. You should be able to describe this sequence:

```bash
ssh <your-username>@rangpur.example.edu
cd <your-project-directory>
source $HOME/miniconda3/bin/activate
conda activate torch
sbatch train_job.sh
squeue -u $USER
```

Use the actual Rangpur hostname, project path, and environment name supplied by your module. Do not invent a completed job if you have not run one.

Example Slurm script structure:

```bash
#!/bin/bash
#SBATCH --job-name=oasis-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch
python train.py --data-root /home/groups/comp3710/... --epochs 50
```

Explain each directive: job name identifies the job, partition and account choose the course resources, `--gres=gpu:1` requests one GPU, `--time` sets a maximum runtime, `%j` inserts the job ID into log names, and `--output`/`--error` preserve training evidence. Use `cat logs/train_<jobid>.out` or `tail -f` to inspect logs, and `scancel <jobid>` only when you intend to stop the job.

Before a long job, run a short GPU smoke test that prints the framework version and confirms CUDA/GPU availability. Then use a small epoch run to check data loading, tensor shapes, one forward pass, loss calculation, checkpoint saving, and logging. Submit the full training job only after that succeeds.

## AI Code Ownership Questions

### How did you verify AI-assisted code?

Say:

"I treated generated code as a draft, not as evidence that it was correct. I checked the dataset shapes and labels, verified train/validation/test separation, confirmed the output activation matched the task type, ran a small smoke test, inspected the training curves and predictions, and compared evaluation logic against the task requirements."

### What simple change could you make?

Choose a real, explainable change in your own code, for example:

- Change epoch count, batch size, learning rate, or number of filters and explain the speed/memory/overfitting trade-off.
- Add checkpointing based on the best validation metric and explain why it avoids losing the best model.
- For multi-class segmentation, change the final layer from one sigmoid channel to four softmax channels, then update the loss and inference rule consistently.
- Change the validation split while leaving the test set untouched, then explain why the test set must remain unseen until final evaluation.

Avoid changing an activation, loss, or output shape in isolation. They form one contract with the label representation.

## Tomorrow's Evidence Checklist

- Have the task sheet open and tick every compulsory result.
- Be logged into Rangpur and have a working `train_job.sh` ready to submit.
- Keep one completed `.out` log, one `.err` log if relevant, training curves, a metric summary, and one inference image or image grid accessible.
- Know the saved model/checkpoint path and how to load it for inference.
- Prepare one true result, one limitation, and one improvement for every model you present.
- Do not present BFS or a U-turn explanation unless the task or your submitted code actually includes it; those are generic algorithms, not part of the displayed `lab2_solution.py` workflow.

## Step-By-Step Demo Plan For Tomorrow

Use this as the order of actions during the oral demo. Do not try to run every long training job live. Show saved, genuine evidence for long runs and perform a small smoke test only when it is safe.

### Before You Enter The Demo

1. Open the task sheet, `lab2_solution.py`, and this practice guide in separate tabs.
2. Open a terminal in the project folder and run a quick check:

```powershell
python lab2_solution.py --part dft
```

3. Keep your real screenshots, training curves, model checkpoint paths, Slurm scripts, `.out` logs, and prediction images in easy-to-find folders.
4. Log into Rangpur before your slot. In a separate terminal, ensure you can show your project folder and run `squeue -u $USER`.
5. Write down actual metrics and epoch numbers. Never guess a result under pressure.

### Step 1: Opening, 0:00 To 0:25

**Do:** Show the task sheet briefly, then open `lab2_solution.py` at `main()`.

**Say:**

"I will show the required results first, then explain the data flow, architecture, loss, evaluation, and how I ran the longer jobs on Rangpur. I use separate commands so the DFT, classification, and segmentation experiments are reproducible and do not all run accidentally."

**If asked, "What did AI do?" say:**

"AI assisted with drafting parts of the implementation, but I verified the code by checking shapes and labels, train/validation/test separation, activation and loss compatibility, small test runs, curves, and predictions. I can explain and modify the main components."

### Step 2: Demonstrate One Fast Command, 0:25 To 1:00

**Do:** Run:

```powershell
python lab2_solution.py --part dft
```

Show the reconstruction plot, terminal timings, `Results agree: True`, and spectrum plot. Avoid running the CNN or U-Net from scratch unless you know the environment and dataset are ready.

**Say:**

"This is a complete local smoke test. The square wave is reconstructed by odd sine harmonics. The direct DFT and NumPy FFT agree numerically, but the direct version takes longer because it calculates every sample-frequency combination, which is $O(N^2)$, while FFT is about $O(N\log N)$."

**If asked, "Why odd harmonics?" say:**

"The Fourier series of an ideal symmetric square wave contains only odd sine harmonics. Their amplitudes decrease as $1/k$. The overshoot around the jumps is Gibbs phenomenon."

### Step 3: Explain A Code Path, 1:00 To 1:45

**Do:** Point to `main()`, then use the relevant function in the file. Let the tutor choose a model if they ask; otherwise show either `run_lfw()` or `run_unet()`.

**Say this code-explanation pattern:**

"The input is ..., this preprocessing does ..., the model receives shape ..., it outputs ..., the loss compares that output with ..., and the metric tells us ...."

**For LFW/CNN, say:**

"The LFW images are split into training and test sets using stratification. PCA is fitted only on training images, then applied to test images, avoiding leakage. The CNN receives grayscale images with shape height by width by one. Convolution layers learn local features, pooling reduces resolution, and softmax outputs one probability per identity. Sparse categorical cross-entropy is appropriate because labels are integer class IDs."

**For U-Net, say:**

"Each image is paired with its mask and normalised to floating-point values. The encoder downsamples to learn context, the decoder upsamples, and skip connections retain fine spatial detail. In this local binary example, sigmoid produces one foreground probability per pixel and binary cross-entropy trains it."

### Step 4: Show Your Genuine Results, 1:45 To 2:20

**Do:** Open saved curves, prediction images, terminal logs, or a completed notebook result. State the run configuration and exact result.

**Say:**

"This result comes from [state your actual dataset, device, epoch count, and run]. The training curve shows [describe the actual trend]. The validation curve is used to monitor training and select settings or a checkpoint. This final test metric was calculated only after training, using data not used for tuning."

**For OASIS, add:**

"For OASIS I convert the four-class softmax output to hard labels with `argmax`, then calculate a separate discrete Dice score for labels 1, 2, and 3. I report each score, because the target applies to every foreground class individually."

**For GAN, add:**

"These are generated 2D MRI slices from the saved generator checkpoint at epoch [actual epoch]. I inspect image grids as well as losses because GAN losses by themselves do not guarantee quality. I save checkpoints through training because a GAN can improve and later collapse."

### Step 5: Demonstrate Rangpur Knowledge, 2:20 To 2:45

**Do:** Show a logged-in Rangpur terminal, your Slurm script, a submitted/completed job, or be ready to submit the script if asked.

**Say:**

"I debug locally first with a small run. For the longer GPU run, I activate the correct environment on Rangpur, submit the Slurm script with `sbatch`, monitor it with `squeue`, and inspect the output and error logs. The script requests the `comp3710` partition, account, one GPU, a time limit, and writes logs using the job ID."

**If asked to submit a job:**

```bash
mkdir -p logs
sbatch train_job.sh
squeue -u $USER
```

Then read back the submitted job ID and show that the matching output paths use that ID. Do not launch a costly full run just for a demonstration unless your tutor specifically instructs you to.

### Step 6: Finish, 2:45 To 3:00

**Say:**

"The main distinction is classification predicts one label per image, whereas segmentation predicts a label for every pixel. My workflow separates training, validation, and final testing; checks the model output and loss match the labels; and saves evidence and checkpoints so the result is reproducible. The next improvement would be [choose one true improvement: augmentation, checkpointing, class-aware loss, more training, or hyperparameter tuning]."

Then stop and let the tutor ask questions. A short, accurate answer is better than filling silence.

## Fast Follow-Up Answers

**Q: What is the difference between validation and testing?**  
"Validation is used during development to monitor learning and select a model. The test set stays unseen until the final evaluation, otherwise its metric becomes biased."

**Q: Why does softmax use four channels for OASIS?**  
"There are four mutually exclusive labels, 0 through 3. Softmax normalises the per-pixel class scores into probabilities that sum to one. `argmax` selects the highest-probability label."

**Q: Why is the local U-Net not the final OASIS implementation?**  
"It is a binary practice pipeline using a different supplied PNG dataset. The OASIS task requires the specified data, four-class output, and discrete per-label Dice evaluation, so the final implementation must change those parts together."

**Q: Why save a checkpoint?**  
"It saves the learned parameters at a known good epoch. I can reload it for inference or resume training, and for GANs it protects a good generator if training later becomes unstable."

**Q: The loss went down. Does that prove the model is good?**  
"No. I also compare validation and final test metrics, then inspect predictions. A falling training loss can still indicate overfitting or a metric mismatch."

**Q: The code throws an error live. What do you do?**  
"I read the traceback, identify whether it is an environment, path, data-shape, or model-contract issue, and show the last verified output rather than pretending it ran. For example, I would verify the dataset path, expected image-mask pairs, tensor shapes, output channels, and loss label format before changing code."


what is transform

frequency
  