# Code behind the week-1 animations (lecture 2)

Every clip in `../renders/w01/` has a runnable file here. The Manim scenes import these files (BackpropFlow and
SpaceWarping use the same constants inline), so each script prints
the numbers its clip shows and saves a static version of the clip to `figures/`. Run from this folder with the
course venv, e.g.

    ../../.venv/bin/python optim_real.py

Clean versions for the slides (no title, captions or numbers; to embed in place of the GIFs) live in `../renders/w01/clean/`: the same code, rendered with `DLC_CLEAN=1`.

Data: CIFAR-10 from `tutorials/data/cifar-10-batches-py` (downloaded on first use by `cifar_experiment.py`), and the
diabetes data from scikit-learn (442 patients, Efron et al. 2004, "Least Angle Regression"; bundled, no download):
bmi (kg/m²) and s5 (log serum triglycerides) predicting disease progression after one year.

| Clip | Code | Function | What to change |
|---|---|---|---|
| SpaceWarping | `space_warping.py` | `hidden`, `best_linear_accuracy` | remove the ReLU (`h = z`): no line separates the classes any more |
| BackpropFlow | `backprop_flow.py` | `forward_backward`, `numeric_gradient`, `check_with_autograd` | other inputs; each gate only needs its local derivative |
| TrainingLoop | `training_loop.py` | `train`, `load_cifar10` | `train(data, lr=1e-1)` or `lr=1e-4`; the 2-layer MLP instead of `nn.Linear(3072, 10)`: the loop stays the same |
| MatrixBackprop | `matrix_backprop.py` | `backward`, `check`, `rejected_product` | `dS @ X` raises a shape error, `dS.T @ X` fits |
| AdamBiasCorrection | `adam_bias_correction.py` | `adam_constant_gradient` | `beta2 = 0.99` or `0.9999`; a noisy gradient `dw` |
| LearningRateRegimes | `optim_real.py` | `learning_rate_regimes(factors, num_steps, w0)` | `factors=(2.0,)`: exactly on the threshold 2/a, w bounces between two points forever |
| IllConditioning | `optim_real.py` | `ill_conditioning`, `bowl`, `sgd` | a smaller lr than 2/(λmin + λmax): the zigzag goes, the crawl gets slower |
| NoisyGradients | `optim_real.py` | `noisy_gradients(batch_sizes, learning_rate, num_steps, seed)`, `minibatches` | `learning_rate=0.025`: the B = 1 loss floor drops from 78 to 46; another `seed` |
| SaddlePoint | `optim_real.py` | `saddle_point(start, learning_rate, rho, num_steps)`, `saddle_gradient` | `start=(0, 0)`: zero gradient, neither method moves; `start=(0.05, -0.05)`: GD needs 174 steps instead of 52 |
| MomentumNesterov | `optim_real.py` | `momentum_nesterov(learning_rate, rho, num_steps)`, `sgd_momentum`, `nesterov` | `learning_rate=0.07`: Nesterov diverges (its limit for ρ = 0.9 is 0.069); `rho=0.5` or `0.99` |
| MomentumOvershoot, NesterovLookAhead | `nesterov_damping.py` (the MomentumNesterov runs from `optim_real.py`, projected on the eigenvectors of H) | `steep_cut(learning_rate, rho, num_steps)`, `recurrence`, `roots`, `look_ahead`, `run_1d` | `learning_rate=0.03`: Nesterov's swing shrinks ×0.61 per step, momentum's still ×0.949 (√ρ, whatever the lr, as long as it swings); `rho=0.5`: ×0.71 vs ×0.11; `learning_rate=0.07`: Nesterov's steep root is −1.02, it diverges across the valley |
| AdaptiveMethods | `optim_real.py` | `adaptive_methods(num_steps, lr_grid, window)`, `adagrad`, `rmsprop`, `adam` | `decay_rate=0.9` in `rmsprop`; delete Adam's two bias-correction lines; `window=None` lets the sweep pick lr = 100 for AdaGrad |
| MuonOrthogonalize | `muon_demo.py` (wrapper around `muon_cifar.py`) | `muon_iterations(steps)`, `ns_map`; `momentum_buffer`, `zeropower_via_newtonschulz5` | `steps=10`: the values keep moving inside a band around 1; coefficients (1.5, −0.5, 0) converge to exactly 1 but need many more steps |
| ConvRunningExample | `conv_running_example.py` | `conv2d(image, kernel, stride, padding)`, `check_with_torch` | `stride=2`: 2×2 output [[4, 4], [2, 4]]; `padding=1`: 5×5 output, the same size as the input; the flipped kernel `kernel[::-1, ::-1]` (textbook convolution) gives the same output here, because this filter is symmetric |
| Muon slides (static, no clip) | `muon_explained.py` (uses `muon_cifar.py`, `muon_demo.py`) | `update_spectrum`, `one_minibatch`, `sign_example`, `same_output_budget`, `newton_schulz_curves` | `same_output_budget(torch.randn(100, 3072))`: a flat spectrum, U Vᵀ gains 1.2× instead of 3.1×; `one_minibatch(N=16)`: 16 outer products, rank at most 16 |

`optim_real.py` writes the optimizers as on the slides (`sgd`, `sgd_momentum`, `nesterov`, `adagrad`, `rmsprop`, `adam`);
each takes `compute_gradient(w)` and returns the whole path. `cifar_experiment.py` and `muon_cifar.py` are the code of the
lecture-2 slides (CIFAR-10 loss curves; Muon vs SGD-momentum vs AdamW). `zeropower_via_newtonschulz5` is Keller Jordan's
(github.com/KellerJordan/Muon, MIT license).
