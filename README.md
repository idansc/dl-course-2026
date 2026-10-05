# Deep Learning 89-6877, Bar-Ilan University, Fall 2026

Lecturer: Idan Schwartz · TA: Tal Fikus

Slides, tutorial notebooks and the weekly plan. Exam 75%, homework 25% (HW1–HW4).

## Weekly plan

| Week | Lecture | Slides | Tutorial: implement | Notebook |
|---|---|---|---|---|
| 1 | Intro to DL: perceptron, MLP, backprop, batch norm | [p1](slides/lectures/p1.pdf), [p3](slides/lectures/p3.pdf) | From a neuron to backprop: autograd engine, manual MLP backward, initial weights, BatchNorm | [T01_neurons_to_backprop](tutorials/T01_neurons_to_backprop.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T01_neurons_to_backprop.ipynb) |
| 2 | Optimization: GD, momentum, Newton, Adam/AdamW, Muon, schedules | [p2](slides/lectures/p2.pdf) | Optimizers from scratch; Muon's Newton–Schulz step; LR schedules | — |
| 3 | CNNs: ResNet, ResNeXt, ConvNeXt; translation equivariance | [p3](slides/lectures/p3.pdf), [p4](slides/lectures/p4.pdf) | conv via im2col, residual blocks, ResNet-18 on CIFAR-10 | — |
| 4 | Sequence learning and language models: tokenization, RNN/LSTM, SSMs | [p6](slides/lectures/p6.pdf), [L8b](slides/2026-updates/L8b_architectures_looped_2026.pdf) | BPE, LSTM cell, SSM recurrent vs parallel form | — |
| 5 | Attention and Transformers: modern block, MoE, ViT; permutation symmetry | [p7](slides/lectures/p7.pdf), [p8](slides/lectures/p8.pdf), [p9](slides/lectures/p9.pdf), [L8b](slides/2026-updates/L8b_architectures_looped_2026.pdf) | Attention, RoPE/RMSNorm/SwiGLU mini-GPT, GQA, MoE, ViT | — |
| 6 | Efficient deep models: pruning, quantization, distillation, KV cache, FlashAttention | [L11b](slides/2026-updates/L11b_efficient_models_2026.pdf) | KV cache, int8/4-bit quantization, pruning, distillation | — |
| 7 | Self-supervised learning: SimCLR, MAE, DINOv2/v3, JEPA; dense probes, SAM 3 | [p5](slides/lectures/p5.pdf) | SimCLR, MAE, linear seg/depth probes on DINOv2 | — |
| 8 | Multimodal learning: CLIP/SigLIP, VLMs, unified token models, omni | [L9b](slides/2026-updates/L9b_unified_multimodal_2026.pdf), new deck | CLIP loss + zero-shot, tiny VLM (ViT + projector + GPT) | — |
| 9 | Generative models I: AR, GANs, VAEs, VQ tokenizers, MaskGIT/VAR | [p9](slides/lectures/p9.pdf) | VAE, DCGAN, VQ layer, AR over VQ tokens | — |
| 10 | Generative models II: flow matching, rectified flow, one-step, video | [p10](slides/lectures/p10.pdf), [L10b](slides/2026-updates/L10b_flow_video_2026.pdf) | DDPM, flow matching, reflow, CFG | — |
| 11 | World models and embodied AI: 3DGS, VGGT, Genie 3, V-JEPA 2, VLA | new deck, [L11a](slides/2026-updates/L11a_rl_foundations_2026.pdf) | 2D Gaussian splatting, action-conditioned predictor, MPC | — |
| 12 | Post-training: scaling laws, SFT, LoRA, DPO, GRPO/RLVR, reasoning | [L11](slides/2026-updates/L11_p11_with_scaling_laws_insert_2026.pdf), [L11a](slides/2026-updates/L11a_rl_foundations_2026.pdf), [p12](slides/lectures/p12.pdf), [L12b](slides/2026-updates/L12b_agents_harness_arcagi3_2026.pdf) | LoRA, SFT masking, DPO, GRPO | — |
| 13 | Interpretability: probes, logit lens, patching, SAEs, circuit tracing | new deck | Grad-CAM/IG, logit lens, activation patching, SAE on mini-GPT | — |
| 14 | Summary and exam rehearsal | [p13](slides/lectures/p13.pdf) | Exam-style problems | — |

`p1`–`p13` are the course lectures; `L8b`–`L12b` are the 2026 update decks (borrowed pages are credited on each slide).

## Tutorials

Each tutorial recaps the lecture and then implements its parts from scratch in PyTorch, checking every piece against
the library version. Later tutorials reuse code from earlier ones.

- `tutorials/TNN_*.ipynb`: the version with blanks (`# TODO` + `...`). Fill these in **before** the tutorial.
- `tutorials/solutions/`: the full version presented in the tutorial (published after the session).
- `tutorials/build/`: the generator. Edit `make_tNN.py`, not the notebooks; one source produces both versions:

```bash
cd tutorials && python build/make_t01.py
```

All notebooks run on CPU or Colab. Local setup: `pip install -r requirements.txt`.
