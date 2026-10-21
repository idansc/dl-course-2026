# Lecture animations: plan and conventions

Short silent clips (15–45 s, 1920×1080, 30 fps) embedded in the lecture slides, 3Blue1Brown-style: one idea per clip,
built step by step, a bottom caption naming the current step, real numbers where possible.

## Conventions (all clips)
- `from style import *` (white background to match the slides, palette: BLUE_ queries/inputs, ORANGE_ keys,
  GREEN_ values/outputs, PURPLE_ weights/attention, RED_ errors/masks; Arial; `title()`, `caption()`,
  `swap_caption()`, `token_row()`, `heatmap()`, `bars()`, `softmax()`).
- One file per week: `wNN_<topic>.py`, one `Scene` class per clip, docstring = what the clip shows + target slide.
- Every step has a caption (`swap_caption`) stating the idea in one line. Captions are statements, not questions.
- Prefer real computations (numpy/torch inside the scene, or precomputed data in `data/`) over made-up numbers.
- No overlaps: check title, labels and caption do not collide (look at frames).
- Render: `manim -qh --disable_caching -o <Scene> wNN_x.py <Scene>`; outputs land in `media/videos/wNN_x/1080p30/`.
  Copy finished clips to `renders/wNN/<Scene>.mp4` and a poster frame `renders/wNN/<Scene>.png`.
- Verify: extract 3–4 frames per clip (PyAV: `av.open(...).decode(...)`) and look at them; fix anything unreadable.

## Clips per week (target slide in the current deck)

| Wk | Scene | Idea | Slides |
|---|---|---|---|
| 1 | PerceptronLearning | Δw = η(y−ŷ)x: each mistake rotates/shifts the boundary; converges on AND, cycles on XOR | p1 17–18, 29 |
| 1 | SpaceWarping | ReLU(Wx) folds the plane so XOR-like points become linearly separable | p3 22–25 |
| 1 | BackpropFlow | forward values flow up, gradients flow down: downstream = local × upstream, on σ(w0x0+w1x1+w2) | p3 30–38 |
| 1 | OptimizerPaths | SGD vs momentum vs Adam on an ill-conditioned valley (contours, live paths) | p2 45–57 |
| 1 | InitSignal | activation histograms per layer for std too small / too large / Kaiming | T01 §5 |
| 1 | TrainingLoop | the lecture-2 loop running on CIFAR-10 (nn.Linear(3072, 10), SGD lr 1e-3): real minibatch → 10 scores → loss → ∂L/∂W → W ← W − α∂L/∂W → repeat, at iterations 1, 11, 21 (loss 2.309, 2.263, 2.164); then 20 epochs: loss curve to 1.69, W turns into 10 class templates, test acc 40.7% (`w01_training_loop.py`; numbers from `code/training_loop.py`) | p2_training_loop 2, 127 |
| 1 | AdamBiasCorrection | constant g = 1, β1 = 0.9, β2 = 0.999: both moments start at 0; uncorrected step lr·m1/√m2 = 3.16·lr at t = 1, 6.57·lr at t = 12, 1.26·lr at t = 1000; dividing by 1 − βᵗ gives exactly lr from step 1 (`w01_training_loop.py`; numbers from `code/adam_bias_correction.py`) | p2_training_loop 115 |
| 1 | MatrixBackprop | shapes make the formula: S = XWᵀ + b; ∂L/∂W must be 10×3072, so (∂L/∂S)ᵀX (∂L/∂S·X rejected: 10 ≠ 128); ∂L/∂X = (∂L/∂S)W; autograd check 1e−17; the 39.3 M-entry Jacobian is never formed (`w01_training_loop.py`; numbers from `code/matrix_backprop.py`) | p2_training_loop 93 |
| 1 | LearningRateRegimes | s = w·bmi on the diabetes data (442 patients): L(w) is a parabola with curvature a = mean(bmi²) = 19.48; GD from w = 0 with lr = 0.1/a, 1/a, 1.9/a, 2.02/a (0.0051, 0.0513, 0.0976, 0.1037): slow, one step, oscillating, diverging; the distance to w* is multiplied by 1 − lr·a per step (×0.90, ×0.00, ×−0.90, ×−1.02), so it converges only if lr < 2/a = 0.1027 (`w01_optimization.py`; numbers from `code/optim_real.py`) | p2_training_loop 97 |
| 1 | IllConditioning | raw-centred (bmi, s5) bowl: curvatures 19.53 and 0.22, κ = 89.8; GD with its best fixed lr 2/(λmin + λmax) = 0.101 zigzags across and crawls along: 104 steps to 1% of the loss gap; standardized features: curvatures 1.45 and 0.55, κ = 2.6, lr = 1.0: 3 steps (`w01_optimization.py`; numbers from `code/optim_real.py`) | p2_training_loop 99 |
| 1 | NoisyGradients | standardized bowl, minibatch SGD (lr 0.05, 150 steps, same start and seed) with B = 442, 16, 1: smooth, wobbling, random walk; mean L − L* over the last 50 steps 0.001 / 2.0 / 78.2 (`w01_optimization.py`; numbers from `code/optim_real.py`) | p2_training_loop 101 |
| 1 | SaddlePoint | one hidden unit s = w2·tanh(w1·x), standardized bmi → progression: real loss surface, saddle at 0 (gradient 0, Hessian eigenvalues ±0.586 = ±mean(xy)), minima ±(0.17, 3.60) at L = 0.328; from (0.05, 0.02), lr 0.1: GD halfway down the loss at step 52, momentum (ρ 0.9) at step 16 (`w01_optimization.py`; numbers from `code/optim_real.py`) | p2_training_loop 100 |
| 1 | MomentumNesterov | raw diabetes bowl (κ 89.8) from w = 0, lr 0.05 for all, ρ 0.9: SGD lands on the valley floor and crawls, momentum speeds along but keeps swinging across, Nesterov damps the swing; 1% of the loss gap after 147 / 22 / 18 steps (`w01_optimization.py`; numbers from `code/optim_real.py`) | p2_training_loop 104–105, 107–108 |
| 1 | AdaptiveMethods | raw diabetes bowl, 100 steps, lr of each from a sweep (lowest loss, path inside the plot): SGD+Momentum 0.03, AdaGrad 10, RMSProp 1, Adam 3; L − L* after 100 steps 2.6e−2 / 5.5e−4 / 1.3e−5 / 1.6e−2; per-coordinate scaling undoes the bad feature scales; with MomentumNesterov it supersedes the synthetic OptimizerPaths (`w01_optimization.py`; numbers from `code/optim_real.py`) | p2_training_loop 112, 116 |
| 1 | MuonOrthogonalize | the Muon slides' matrix: W1's SGD-momentum buffer M (100×3072) after one CIFAR-10 epoch; 100 singular values, top-1 33% and top-5 73% of the energy; 5 Newton–Schulz steps (σ ← 3.4445σ − 4.7750σ³ + 2.0315σ⁵, Keller Jordan's code, MIT) leave all 100 in [0.68, 1.13]: update ≈ UVᵀ (`w01_optimization.py`; numbers from `code/muon_demo.py`, a wrapper around `code/muon_cifar.py`) | p2_training_loop 122 |
| 2 | ConvSliding | a 3×3 edge kernel slides over a real image, the feature map fills in | p3 53–60 |
| 2 | ReceptiveField | stacked 3×3 convs: the region one output sees grows by 2 per layer | p3 72–74 |
| 2 | ResidualHighway | plain deep net vs ResNet: gradient magnitude per layer, the skip path as a highway | p4 |
| 2 | GradCAMReveal | Grad-CAM heat appears over the object; a shortcut model looks at the background | exam B |
| 3 | RNNUnroll | RNN unrolled through time; gradient shrinks going back (∏ tanh'·w) | p6 |
| 3 | Seq2SeqBottleneck | one context vector squeezes the sentence; attention lets each output word look back | p7 2–17 |
| 3 | SoftLookup | attention as a soft dictionary lookup (rotate the query) | p7 58–63 ✅ |
| 3 | ScaledScores | why divide by √d | p7 63 ✅ |
| 3 | BPEMerges | byte-pair merges build tokens from characters | T04 |
| 4 | SelfAttentionHeads | real GPT-2 heads: coreference, previous token, broad | p7 64–70, 80 ✅ |
| 4 | CausalMask | masked self-attention and next-token generation | p8 33–34 ✅ |
| 4 | PermutationEquivariance | shuffle tokens → outputs shuffle the same way; add positions → no longer | p7 71–77 |
| 4 | TransformerBlock | residual stream: attention mixes across tokens, MLP acts per token | p7 84–92 |
| 4 | RoPERotation | queries/keys rotated by position; the score depends only on the offset | L8b |
| 4 | ViTPatches | image → 16×16 patches → tokens + positions → Transformer | p8 61 |
| 4 | SinusoidalClock | sinusoidal PE as 8 clock hands at θ_j = 10000^(−2j/d); d×T heatmap fills in; a shift k turns every hand by kθ_j (PE(m+k) = R_k PE(m)); PE·PE banded, constant along diagonals (`w04_positions.py`) | p7 71–77 |
| 4 | WherePositionEnters | the q·k pipeline and where position enters: (a) x+p before W_Q/W_K, (b) bias b(i−j) (T5/ALiBi), (c) rotary, (d) NoPE (mask only); then ALiBi on 4 heads, slopes 1/2…1/128 (`w04_positions.py`) | p7 71–77, L8b |
| 4 | LengthExtrapolation | RoPE past the training length (d=128, L=4096 → 16384): slow pairs never finish a turn, so angles past L are unseen; position interpolation squeezes them back; wavelengths for PI / NTK base 40,890 / YaRN (`w04_positions.py`) | L8b |
| 5 | Roofline | arithmetic intensity: prefill is compute-bound, decoding is memory-bound | L11b |
| 5 | MemoryBudget | params, grads, Adam states, activations, KV cache as stacked bars while the model grows | new |
| 5 | Quantization | float weights snap to an int8 grid; error vs bits | L11b |
| 5 | Parallelism | data / tensor / pipeline / FSDP: how a model and a batch split across GPUs | new |
| 5 | PowerLawFrontier | learning curves per model size; their lower envelope is a power law in C (Chinchilla fit) | L11 scaling |
| 5 | IsoFLOPProfiles | fixed budget → loss valley over N; minima give N_opt ∝ C^0.45, derived on screen | L11 scaling |
| 5 | TrainVsInferenceOptimal | same target loss: lifetime FLOPs 6ND + 2N·T; heavy serving favours small over-trained models (Llama 3 8B) | L11 scaling |
| 6 | ContrastiveSphere | two views pulled together, negatives pushed apart on the unit sphere | p5 |
| 6 | MAEMasking | 75% of patches hidden, the encoder sees the rest, the decoder reconstructs | p5 |
| 6 | CollapseVsEMA | two identical nets collapse to a constant; an EMA teacher prevents it | p5 |
| 7 | CLIPMatrix | image/text embeddings, the similarity matrix, the diagonal; zero-shot by prompting | mm4–5 |
| 7 | VLMRecipe | image patches → vision encoder → projector → tokens into the LLM | L9b |
| 7 | JanusTwoEncoders | SigLIP path for understanding, VQ tokens for generation, one Transformer | L9b |
| 8 | AEvsVAELatent | autoencoder latent has holes; the VAE KL term fills them; interpolation | p9, mm9 |
| 8 | Reparameterization | z = μ + σ·ε: the randomness moves outside the gradient path | mm9 |
| 8 | GANDynamics | 1-D: generator density chases the data, discriminator curve flattens; mode collapse | new |
| 8 | VQSnap | encoder vectors snap to the nearest codebook entry; straight-through gradient | p9 |
| 9 | ForwardReverseDiffusion | 2-D data dissolves into a Gaussian and is denoised back | p10 |
| 9 | FlowMatchingPaths | straight paths noise→data, the learned velocity field, Euler steps | L10b |
| 9 | GuidanceScale | CFG: increasing guidance pulls samples toward the class, less diversity | p10 59–62 |
| 9 | FewStepSampling | curved vs straightened (reflow) paths: 1, 2, 4 steps | L10b |
| 10 | GaussianSplatting | an image fitted by Gaussians that move, stretch and recolor | new |
| 10 | WorldModelRollout | action → predicted frame → next action; error grows with the horizon | new |
| 10 | MPCImagination | sample many imagined trajectories, execute the best first action | new |
| 11 | PostTrainingPipeline | pretrain → SFT → preference / RLVR; what each stage changes | p11 |
| 11 | LoRALowRank | frozen W plus low-rank B·A; trainable parameters shrink by orders of magnitude | L11b |
| 11 | PolicyGradientBars | sample completions, score them, push probability toward good ones (GRPO group advantages) | p12 |
| 11 | DPOMargin | chosen vs rejected log-prob ratios move apart; β keeps them near the reference | p11 |
| 12 | AgentLoop | think → call tool → observe → repeat, with a step budget | L12b |
| 12 | DenseRetrieval | query and passages in embedding space; nearest neighbors; BM25 vs dense | new |
| 12 | MaxSim | ColBERT: every query token finds its best-matching passage token | new |
| 12 | PassAtK | pass@k grows with k; majority vote vs verifier | L12b |
