# Code behind the week-1 animations (lecture 2)

Every clip in `../renders/w01/` has a runnable file here. The Manim scenes import these files, so each script prints
the numbers its clip shows and saves a static version of the clip to `figures/`. Run from this folder with the
course venv, e.g.

    ../../.venv/bin/python optim_real.py

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
| AdaptiveMethods | `optim_real.py` | `adaptive_methods(num_steps, lr_grid, window)`, `adagrad`, `rmsprop`, `adam` | `decay_rate=0.9` in `rmsprop`; delete Adam's two bias-correction lines; `window=None` lets the sweep pick lr = 100 for AdaGrad |
| MuonOrthogonalize | `muon_demo.py` (wrapper around `muon_cifar.py`) | `muon_iterations(steps)`, `ns_map`; `momentum_buffer`, `zeropower_via_newtonschulz5` | `steps=10`: the values keep moving inside a band around 1; coefficients (1.5, −0.5, 0) converge to exactly 1 but need many more steps |

`optim_real.py` writes the optimizers as on the slides (`sgd`, `sgd_momentum`, `nesterov`, `adagrad`, `rmsprop`, `adam`);
each takes `compute_gradient(w)` and returns the whole path. `cifar_experiment.py` and `muon_cifar.py` are the code of the
lecture-2 slides (CIFAR-10 loss curves; Muon vs SGD-momentum vs AdamW). `zeropower_via_newtonschulz5` is Keller Jordan's
(github.com/KellerJordan/Muon, MIT license).
