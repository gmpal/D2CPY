# D2C Algorithm

The D2C algorithm is designed to predict the existence of a causal link between two variables in a multivariate setting. It works by:

1. Creating a set of features of the relationship between the members of the Markov Blankets of the two variables.
2. Using a classifier (e.g., a Random Forest) to learn a mapping between the features and the presence of a causal link.

Two sets of features are used to summarize the relationship between the two Markov blankets. The first set of features accounts for the presence (or the position if the Markov Blanket is obtained by ranking) of the terms of Mj in Mi and vice versa. The second set of features is based on the distributions of the asymmetric descriptors with a set of quantiles.

## D2C Algorithm Steps

For each pair of measured variables `z_i` and `z_j`,

1. Infers from data the two Markov Blankets (e.g., by using state-of-the-art approaches) `M_i` and `M_j` and the subsets `M_i \ z_j` and `M_j \ z_i`. Most existing algorithms associate a ranking with the Markov Blanket such that the most strongly relevant variables are ranked before.

2. Computes a set of (conditional) mutual information terms describing the dependency between `z_i` and `z_j`

$$
I=\left[I\left(\mathbf{z}_i ; \mathbf{z}_j\right), I\left(\mathbf{z}_i ; \mathbf{z}_j \mid \mathbf{M}_j \backslash \mathbf{z}_i\right), I\left(\mathbf{z}_i ; \mathbf{z}_j \mid \mathbf{M}_i \backslash \mathbf{z}_j\right)\right]
$$


3. Computes the positions `P_i^(k_i)` of the members `m^(k_i)` of `M_i \ z_j` in the ranking associated to `M_j \ z_i` and the positions `P_j^(k_j)` of the terms `m^(k_j)` in the ranking associated to `M_i \ z_j`. In case of the absence of a term of `M_i` in `M_j`, the position is set to `K_j+1` (respectively `K_i+1`).

4. Computes the populations based on the asymmetric descriptors:

(a) $D_1(i, j)=\left\{I\left(\mathbf{z}_i ; \mathbf{m}_j^{\left(k_j\right)} \mid \mathbf{z}_j\right), k_j=1, \ldots, K_j\right\}$ <br>
(b) $D_1(j, i)=\left\{I\left(\mathbf{z}_j ; \mathbf{m}_i^{\left(k_i\right)} \mid \mathbf{z}_i\right), k_i=1, \ldots, K_i\right\}$ <br>
(c) $D_2(i, j)=\left\{I\left(\mathrm{~m}_i^{\left(k_i\right)} ; \mathbf{m}_j^{\left(k_j\right)} \mid \mathbf{z}_j\right), k_i=1, \ldots, K_i, k_j=1, \ldots, K_j\right\}$ <br>
(d) $D_2(j, i)=\left\{I\left(\mathbf{m}_j^{\left(k_j\right)} ; \mathbf{m}_i^{\left(k_i\right)} \mid \mathbf{z}_i\right), k_i=1, \ldots, K_i, k_j=1, \ldots, K_j\right\}$ <br>
(e) $D_3(i, j)=\left\{I\left(\mathrm{z}_i ; \mathbf{m}_j^{\left(k_j\right)}\right), k_j=1, \ldots, K_j\right\}$ <br>
(f) $D_3(j, i)=\left\{I\left(\mathbf{z}_j, \mathbf{m}_i^{\left(k_i\right)}\right), k_i=1, \ldots, K_i\right\}$



5. Creates a vector of descriptors

$$
\begin{aligned}
& x=\left[I, \mathcal{Q}\left(\hat{P}_i\right), \mathcal{Q}\left(\hat{P}_j\right), \mathcal{Q}\left(\hat{D}_1(i, j)\right), \mathcal{Q}\left(\hat{D}_1(j, i)\right),\right. \\
& \left.\mathcal{Q}\left(\hat{D}_2(i, j)\right), \mathcal{Q}\left(\hat{D}_2(j, i)\right), \mathcal{Q}\left(\hat{D}_3(i, j)\right), \mathcal{Q}\left(\hat{D}_3(j, i)\right)\right]
\end{aligned}
$$





where `P_hat_i` and `P_hat_j` are the empirical distributions of the populations `{P_i^(k_i)}` and `{P_j^(k_j)}`, `D_hat_h(i, j)` denotes the empirical distribution of the corresponding population `D_h(i, j)`, and `Q` returns a set of sample quantiles of a distribution (in the experiments we set the quantiles to `0.1,0.25,0.5,0.75,0.9`).

The vector `x` can then be derived from observational data and used to create a vector of descriptors to be used in a supervised learning paradigm.

The rationale of the algorithm is that the asymmetries between `M_i` and `M_j` induce an asymmetry on the distributions `P^` and `D^`, and that the quantiles of those distributions provide information about the directionality of the causal link (zi ! zj or zj ! zi). These distributions would be more informative if we were able to rank the terms of the Markov Blankets by prioritizing the direct causes (i.e., the terms `c_i` and `c_j`).

In the experiments, we derive the information terms as differences between (conditional) entropy terms, which are estimated by a Lazy Learning regression algorithm under the assumption of Gaussian noise. The (conditional) mutual information terms are then obtained by using the relations (1) and (3).


