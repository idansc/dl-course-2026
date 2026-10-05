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
