# Deep Learning 89-6877, Bar-Ilan University, Fall 2026

Lecturer: Idan Schwartz · TA: Tal Fiskus

Slides, tutorial notebooks and the weekly plan. Course 89-6877-01, semester A 2026-27 (תשפ"ז). Exam 75%, homework 25% (HW1–HW4).

## Weekly plan (13 weeks)

| Week | Lecture | Slides | Tutorial |
|---|---|---|---|
| 1 | Foundations: MLP, backprop, initialization, normalization, optimizers | [p1](slides/lectures/p1.pdf), [p3](slides/lectures/p3.pdf), [p2](slides/lectures/p2.pdf) | [T01_neurons_to_backprop](tutorials/T01_neurons_to_backprop.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T01_neurons_to_backprop.ipynb) · [recap](tutorials/recap/T01_neurons_to_backprop_recap.pdf) <br> [T01b_optimizers](tutorials/T01b_optimizers.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T01b_optimizers.ipynb) · [recap](tutorials/recap/T01b_optimizers_recap.pdf) |
| 2 | CNNs: convolution, ResNet, equivariance; Grad-CAM | [p3](slides/lectures/p3.pdf), [p4](slides/lectures/p4.pdf) | [T02_cnns](tutorials/T02_cnns.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T02_cnns.ipynb) · [recap](tutorials/recap/T02_cnns_recap.pdf) |
| 3 | Sequences and attention: tokenization, RNN/LSTM, SSMs, attention | [p6](slides/lectures/p6.pdf), [p7](slides/lectures/p7.pdf), [L8b](slides/2026-updates/L8b_architectures_looped_2026.pdf) | [T03_sequences](tutorials/T03_sequences.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T03_sequences.ipynb) · [recap](tutorials/recap/T03_sequences_recap.pdf) |
| 4 | Transformers: modern block, GQA/MoE, ViT; logit lens | [p8](slides/lectures/p8.pdf), [p9](slides/lectures/p9.pdf), [L8b](slides/2026-updates/L8b_architectures_looped_2026.pdf) | [T04_transformers](tutorials/T04_transformers.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T04_transformers.ipynb) · [recap](tutorials/recap/T04_transformers_recap.pdf) |
| 5 | Compute and efficiency: FLOPs, memory, roofline, quantization, pruning, distillation | [L11b](slides/2026-updates/L11b_efficient_models_2026.pdf) | [T05_compute_efficiency](tutorials/T05_compute_efficiency.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T05_compute_efficiency.ipynb) · [recap](tutorials/recap/T05_compute_efficiency_recap.pdf) |
| 6 | Self-supervised learning: SimCLR, MAE, DINOv2, JEPA; SAEs | [p5](slides/lectures/p5.pdf) | [T06_self_supervised](tutorials/T06_self_supervised.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T06_self_supervised.ipynb) · [recap](tutorials/recap/T06_self_supervised_recap.pdf) |
| 7 | Multimodal: CLIP, VLMs, Janus, omni | [mm3](slides/multimodal/mm3.pdf), [mm4](slides/multimodal/mm4.pdf), [mm5](slides/multimodal/mm5.pdf), [mm6](slides/multimodal/mm6.pdf), [mm7](slides/multimodal/mm7.pdf), [L9b](slides/2026-updates/L9b_unified_multimodal_2026.pdf) | [T07_multimodal](tutorials/T07_multimodal.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T07_multimodal.ipynb) · [recap](tutorials/recap/T07_multimodal_recap.pdf) |
| 8 | Generative models I: VAE, GAN, VQ, autoregressive | [p9](slides/lectures/p9.pdf), [mm9](slides/multimodal/mm9.pdf) | [T08_generative_models](tutorials/T08_generative_models.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T08_generative_models.ipynb) · [recap](tutorials/recap/T08_generative_models_recap.pdf) |
| 9 | Diffusion and flow matching, guidance, few-step | [p10](slides/lectures/p10.pdf), [L10b](slides/2026-updates/L10b_flow_video_2026.pdf) | [T09_diffusion_flow](tutorials/T09_diffusion_flow.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T09_diffusion_flow.ipynb) · [recap](tutorials/recap/T09_diffusion_flow_recap.pdf) |
| 10 | World models and embodied AI | [L10b](slides/2026-updates/L10b_flow_video_2026.pdf), [L11a](slides/2026-updates/L11a_rl_foundations_2026.pdf) | [T10_world_models](tutorials/T10_world_models.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T10_world_models.ipynb) · [recap](tutorials/recap/T10_world_models_recap.pdf) |
| 11 | Post-training: SFT, LoRA, DPO, GRPO, reasoning | [L11](slides/2026-updates/L11_p11_with_scaling_laws_insert_2026.pdf), [L11a](slides/2026-updates/L11a_rl_foundations_2026.pdf), [p12](slides/lectures/p12.pdf) | [T11_post_training](tutorials/T11_post_training.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T11_post_training.ipynb) · [recap](tutorials/recap/T11_post_training_recap.pdf) |
| 12 | Agents, harnesses and retrieval | [L12b](slides/2026-updates/L12b_agents_harness_arcagi3_2026.pdf) | [T12_agents_retrieval](tutorials/T12_agents_retrieval.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T12_agents_retrieval.ipynb) · [recap](tutorials/recap/T12_agents_retrieval_recap.pdf) |
| 13 | Summary and exam rehearsal | [p13](slides/lectures/p13.pdf) | [T13_exam_practice](tutorials/T13_exam_practice.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idansc/dl-course-2026/blob/main/tutorials/T13_exam_practice.ipynb) · [recap](tutorials/recap/T13_exam_practice_recap.pdf) |

`p1`–`p13` are the course lectures, `L8b`–`L12b` the 2026 update decks (borrowed pages are credited on each slide), `mm*` the Multimodal DL lectures.

## Tutorials

Each tutorial starts with ~15 minutes of recap slides (selected lecture pages, `tutorials/recap/`) and then implements its parts from scratch in PyTorch, checking every piece against
the library version. Later tutorials reuse code from earlier ones.

- `tutorials/TNN_*.ipynb`: the version with blanks (`# TODO` + `...`). Fill these in **before** the tutorial.
- `tutorials/solutions/`: the full version presented in the tutorial (published after the session).
- `tutorials/build/`: the generator. Edit `make_tNN.py`, not the notebooks; one source produces both versions:

```bash
cd tutorials && python build/make_t01.py   # one script per tutorial
```

All notebooks run on CPU or Colab. Local setup: `pip install -r requirements.txt`.
