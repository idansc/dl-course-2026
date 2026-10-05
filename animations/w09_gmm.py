"""Exact math for the week-9 clips: 2-D isotropic Gaussian mixtures have closed-form noised densities, so the
diffusion score / denoiser and the flow-matching marginal velocity are exact (no network needed).

Mixture: means M (K,2), stds S (K,), weights W (K,).
Diffusion (VP / DDPM):  x_t = sqrt(abar) x_0 + sqrt(1-abar) eps   ->  component k: N(sqrt(abar) m_k, (abar s_k^2 + 1 - abar) I)
Flow matching (rectified flow, x_0 = noise, x_1 = data):  x_t = (1-t) x_0 + t x_1
    component k: x_t ~ N(t m_k, ((1-t)^2 + t^2 s_k^2) I),  E[x_1 - x_0 | x_t, k] = m_k + (t s_k^2 - (1-t)) / var_k * (x_t - t m_k)
"""
import numpy as np


class GMM:
    def __init__(self, M, S, W=None):
        self.M = np.asarray(M, float)
        self.S = np.broadcast_to(np.asarray(S, float), (len(self.M),)).copy()
        self.W = np.full(len(self.M), 1 / len(self.M)) if W is None else np.asarray(W, float) / np.sum(W)

    def sample(self, n, rng):
        k = rng.choice(len(self.M), n, p=self.W)
        return self.M[k] + self.S[k, None] * rng.standard_normal((n, 2)), k

    def subset(self, idx):
        return GMM(self.M[idx], self.S[idx], self.W[idx])

    @staticmethod
    def _resp(x, means, var, W):
        d2 = ((x[:, None] - means[None]) ** 2).sum(-1)                 # (n, K)
        logp = np.log(W)[None] - np.log(var)[None] - 0.5 * d2 / var[None]
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp)
        return r / r.sum(1, keepdims=True)

    # ---- diffusion -------------------------------------------------------------------------------------------
    def eps_hat(self, x, abar):
        """Exact E[eps | x_t] for the VP process (= -sqrt(1-abar) * score)."""
        var = abar * self.S ** 2 + 1 - abar
        mu = np.sqrt(abar) * self.M
        r = self._resp(x, mu, var, self.W)
        score = -(r[:, :, None] * (x[:, None] - mu[None]) / var[None, :, None]).sum(1)
        return -np.sqrt(1 - abar) * score

    def x0_hat(self, x, abar):
        return (x - np.sqrt(1 - abar) * self.eps_hat(x, abar)) / np.sqrt(abar)

    # ---- flow matching ---------------------------------------------------------------------------------------
    def velocity(self, x, t):
        var = (1 - t) ** 2 + t ** 2 * self.S ** 2
        mu = t * self.M
        r = self._resp(x, mu, var, self.W)
        coef = (t * self.S ** 2 - (1 - t)) / var                       # (K,)
        vk = self.M[None] + coef[None, :, None] * (x[:, None] - mu[None])
        return (r[:, :, None] * vk).sum(1)


def cosine_abar(t, s=0.008):
    """Continuous cosine schedule (Nichol & Dhariwal), t in [0, 1]."""
    f = lambda u: np.cos((u + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(f(np.asarray(t, float)) / f(0.0), 1e-5, 1.0)


def ddpm_sample(eps_fn, x, T=100, rng=None, record=False):
    """Ancestral DDPM sampling with T steps of the cosine schedule. eps_fn(x, abar) -> eps_hat.
    Returns final x (and the per-step states and x0-predictions if record)."""
    ts = np.linspace(0, 1, T + 1)
    ab = cosine_abar(ts)
    xs, x0s = [x.copy()], []
    for i in range(T, 0, -1):
        a_t, a_s = ab[i], ab[i - 1]
        alpha = a_t / a_s
        beta = 1 - alpha
        e = eps_fn(x, a_t)
        x0s.append((x - np.sqrt(1 - a_t) * e) / np.sqrt(a_t))
        mean = (x - beta / np.sqrt(1 - a_t) * e) / np.sqrt(alpha)
        if i > 1:
            var = beta * (1 - a_s) / (1 - a_t)
            x = mean + np.sqrt(var) * rng.standard_normal(x.shape)
        else:
            x = mean
        xs.append(x.copy())
    return (x, np.array(xs), np.array(x0s)) if record else x


def euler(v_fn, x, n, record=False):
    xs = [x.copy()]
    for i in range(n):
        x = x + (1 / n) * v_fn(x, i / n)
        xs.append(x.copy())
    return np.array(xs) if record else x


def two_moons(n_per=16, s=0.08, scale=1.6):
    th = np.linspace(0, np.pi, n_per)
    up = np.stack([np.cos(th), np.sin(th)], 1)
    lo = np.stack([1 - np.cos(th), 0.5 - np.sin(th)], 1)
    M = (np.concatenate([up, lo]) - np.array([0.5, 0.25])) * scale
    return GMM(M, s * scale)


def ring(k=8, radius=2.6, s=0.18):
    a = np.arange(k) * 2 * np.pi / k
    return GMM(radius * np.stack([np.cos(a), np.sin(a)], 1), s)
