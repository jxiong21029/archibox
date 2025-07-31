import matplotlib.pyplot as plt
import numpy as np

# ---------------- mixture + base parameters ------------------------------
sigma0_sq = 1.0  # variance of base Gaussian  x0
pis = np.array([0.5, 0.5])  # mixture weights  (π₁,π₂)
mus = np.array([1.1, -1.1])  # means            (μ₁,μ₂)
sigmas_sq = np.array([0.4, 0.4])  # component variances (σ₁²,σ₂²)


# ---------------- velocity function --------------------------------------
def velocity(xt, t):
    """
    Ground-truth rectified-flow velocity for transport
    from N(0, σ0_sq)  →  mixture Σ_i π_i N(μ_i, σ_i²).

    Parameters
    ----------
    xt : float or ndarray
        Value(s) of x_t at time t.
    t : float in (0,1)
        The interpolation time.

    Returns
    -------
    v : float or ndarray
        Velocity v_t(x_t) = E[x1 - x0 | x_t] .
    """
    xt = np.asarray(xt, dtype=float)
    t = np.asarray(t, dtype=float)
    one_minus_t = 1.0 - t

    # per-component mean/variance of x_t when the draw came from component i
    var_t = (
        one_minus_t[..., None] ** 2 * sigma0_sq + t[..., None] ** 2 * sigmas_sq
    )  # shape (m,)
    mean_t = t[..., None] * mus  # shape (m,)

    # mixture posterior weights   w_i(x_t) = π_i N(x_t; mean_t, var_t)
    log_w = np.log(pis) - 0.5 * (
        (xt[..., None] - mean_t) ** 2 / var_t + np.log(2 * np.pi * var_t)
    )
    log_w -= log_w.max(axis=-1, keepdims=True)  # for numerical stability
    w = np.exp(log_w)
    w /= w.sum(axis=-1, keepdims=True)  # shape (..., m)

    # conditional mean  E[x1 | x_t, i]
    mu_post = mus + (t[..., None] * sigmas_sq) / var_t * (xt[..., None] - mean_t)

    # E[x1 | x_t]  = Σ_i w_i μ_post_i
    e_x1 = np.sum(w * mu_post, axis=-1)

    # velocity v_t(x_t) = (E[x1] - x_t)/(1 - t)
    return (e_x1 - xt) / one_minus_t * 2.0


def integrate_single(x0, ts):
    xs = np.empty_like(ts)
    xs[0] = x0
    for k in range(len(ts) - 1):
        dt = ts[k + 1] - ts[k]
        xs[k + 1] = xs[k] + dt * velocity(xs[k], ts[k])
    return xs


def main():
    # Plot the velocity (rectified flow) field
    fig = plt.figure()
    fig, axes = plt.subplots(
        ncols=3, sharey=True, gridspec_kw={"width_ratios": [1, 3, 1]}
    )
    fig.set_size_inches(8, 6)

    x0_samples = np.random.normal(loc=0.0, scale=1.0, size=2000)
    axes[0].hist(x0_samples, orientation="horizontal", bins=30, density=True)
    axes[0].set_ylim(-4, 4)

    # Generate quiver grid
    t_vals = np.linspace(0.0, 1.0, 25)[1:-1]  # horizontal axis (time)
    x_vals = np.linspace(-4.0, 4.0, 21)  # vertical axis (data)
    T, X = np.meshgrid(t_vals, x_vals)
    V = velocity(X, T)  # vertical component
    U = np.ones_like(V)  # every arrow steps 1 in t
    speed = np.sqrt(U**2 + V**2)
    U_norm, V_norm = U / speed, V / speed

    axes[1].quiver(T, X, U_norm, V_norm, angles="xy", pivot="mid", scale=30, alpha=0.7)

    # Generate trajectories
    ts = np.linspace(0.0, 1.0, 501)
    trajectories = [integrate_single(x0, ts) for x0 in x0_samples]
    for xs in trajectories[:20]:
        axes[1].plot(ts, xs, c=plt.rcParams["axes.prop_cycle"].by_key()["color"][0])

    axes[2].hist(
        [traj[-1] for traj in trajectories],
        orientation="horizontal",
        bins=30,
        density=True,
    )

    fig.suptitle("Ground-truth Rectified Flow\n1-D Gaussian: σ²(0)=1 → σ²(1)=2")
    fig.savefig("tmp2.png")


if __name__ == "__main__":
    main()
