# Implemented mathematical model

## 1. Quantum states

For `n ∈ {1,2,3}` qubits, the Hilbert-space dimension is `d = 2^n`. Training states are sampled from a configurable mixture of:

1. Haar-random pure states;
2. Hilbert–Schmidt/Ginibre mixed states;
3. depolarized Haar-pure states.

Every target satisfies `rho >= 0`, `rho = rho†`, and `Tr(rho)=1`.

## 2. Pauli-cube measurements

The measurement settings are all tensor products in `{X,Y,Z}^n`. For setting `s` and outcome `o`, the projector is

`M_(s,o) = tensor_j [ (I + (-1)^(o_j) sigma_(s_j))/2 ]`.

Born probabilities are

`p_(s,o) = Tr(M_(s,o) rho)`.

For every setting independently,

`(n_(s,1),...,n_(s,d)) ~ Multinomial(N; p_(s,1),...,p_(s,d))`,

and `f_(s,o)=n_(s,o)/N`. Therefore each setting block sums to one, while the complete flattened vector sums to `3^n`.

Input dimensions are 6, 36, and 216 for one, two, and three qubits.

## 3. Physical copy-replacement attack

If `m = alpha N` copies are replaced by an attacker-selected state `sigma`, the effective ensemble is

`rho_eff = (1-alpha)rho + alpha sigma`.

The physical attack budget is

`D_tr(rho,rho_eff) <= epsilon_p`,

where `D_tr(rho,sigma)=0.5 ||rho-sigma||_1`. Since

`D_tr(rho,rho_eff)=alpha D_tr(rho,sigma) <= alpha`,

the implementation clips the requested fraction to

`alpha_eff = min(alpha_requested, epsilon_p / D_tr(rho,sigma), 1)`.

Implemented physical variants are random replacement, targeted replacement, fixed replacement, and a worst-eigenstate stress test.

## 4. Frequency-space PGD

Frequency PGD solves an untargeted inner maximization of reconstruction infidelity around a clean empirical frequency vector. Every step is projected onto the intersection of:

1. `||f_adv-f_clean||_infinity <= epsilon_f`;
2. `f_adv >= 0`;
3. each setting block sums exactly to one.

The bounded-simplex projection is performed independently for every Pauli setting.

## 5. Physical reconstruction head

The network outputs `d^2` real numbers. They parameterize a complex lower-triangular matrix `T`: positive diagonal elements are obtained with `softplus`, and each strict-lower-triangular entry uses one real and one imaginary parameter. The prediction is

`rho_hat = T T† / Tr(T T†)`.

This guarantees Hermiticity, positive semidefiniteness, and unit trace by construction.

## 6. Latest agreed loss

For clean and adversarial frequency vectors from the same true state,

`L_clean = 1 - F(rho,rho_hat_clean)`,

`L_adv = 1 - F(rho,rho_hat_adv)`,

`L_cons = 1 - F(stop_gradient(rho_hat_clean),rho_hat_adv)`.

The total objective is

`L_total = L_clean + L_adv + 0.1 L_cons`.

`F` is the squared Uhlmann fidelity. The clean prediction is detached only inside the consistency term, so the clean branch acts as the stable teacher for the adversarial branch.

There is no attack classifier, no detection head, and no BCE loss.
