# Environment Setup
Just run `conda env create -f environment.yml` for an environment called `nearest-sequence`.
# Rough Algorithm
The general flow of this algorithm. See `nn_eval.py` for an example of implementation.
1. Deploy an agent in some environment
2. Every time the agent makes an observation, append it to an observation history buffer (`nn_agent.obs_history`)
3. For each observation made,
	1. Find all direct nearest neighbors to that query point using standard Euclidean distance (`candidates')
	2. Then, look back through the observation history and the previous states of all neighbors and calculate a cumulative distance calculation that will account for historical distance
	3. Run another pass, picking some (0.0, 100.0] percentile of the neighbors to proceed with
    4. There are a few different modes right now:
        - Perform locally-weighted linear linear regression to find an action (`NN_METHOD.LWR`)
        - At initialization-time, train a model on states and distances and now query it with the current neighbors and distances (`NN_METHOD.COND`)
        - Train a Gaussian Mixture Model with locally-weighted negative log-likelihood loss function and query it (`NN_METHOD.GMM`)
# Hyperparameter Explanation
- **Candidates**: The 'K' in KNN - how many candidate neighbors we want to do cumulative distance on.
- **Lookback**: How far back we want to look (in states) into each trajectory when doing the cumulative distance function.
- **Decay**: How exponentially we want to decrease the influence of older neighbors. For each index:
  - i.e. `i=1` is the most recent observation, and `i=10` is the 10th newest observation.
  - Each `i` will have its respective distance multiplied by `i^decay`.
  - Typically, we want decay to be negative (older observations have less influence).
- **Final Neighbors Ratio**: After calculating the cumulative distance, we take the
  (100 * `final_neighbors_ratio`)% best neighbors. This can be a cheap way to handle multi-modality.
  - If there are likely two modes evenly distributed in our neighbors, and `final_neighbors_ratio` is 0.5, we will take only the 50% closest neighbors post-cumulative distance function, ideally eliminating one of the two modes.
# Code to Look At
If you want to gain an understanding of this algorithm, take a look at `nn_eval.py` for the high-level implementation. For the actual nearest-sequence algorithm, take a look at the `nn_agent` and `nn_util` classes.

Run `python nn_eval.py config/env/hopper.yml config/policy/ns_lwr.yml` to see the program work. At the time of writing, you should eventually see
`Candidates 100, lookback 10, decay -2, ratio 0.5: mean 3574.30758240725, std 16.8464824159680`
# Results

## MuJoCo

| **Algorithm**  | **Hopper, 1 traj**       | **Ant, 10 traj**         | **Walker, 5 traj**       | **Halfcheetah, 50 traj**    |
|----------------|--------------------------|--------------------------|--------------------------|-----------------------------|
| BC             | 1365.59 ± 239.82          | 3395.87 ± 1990.10         | 657.01 ± 328.88           | 440.30 ± 309.62             |
| CCIL           | 1437.65 ± 731.84          | 3676.99 ± 1874.95         | 4161.88 ± 965.78          | 8624.58 ± 2694.34           |
| NN+LWR         | 1196.39 ± 1194.26         | **4411.94** ± 783.23      | 4296.24 ± 1468.08         | 10069.78 ± 2103.91          |
| NS+LWR         | **2041.35** ± 1366.38     | **4507.62** ± 811.26      | **4835.67** ± 627.87      | **10347.10** ± 1325.95      |
| NS+LWR+DTW     | 1848.73 ± 1362.48         | **4468.14** ± 1055.86     | **4905.67** ± 418.21      | **10326.08** ± 1370.11      |

## Metaworld

| **Algorithm**  | **Coffee Pull, 50 traj**  | **Coffee Push, 50 traj**  | **Button Press, 50 traj** | **Drawer Closer, 50 traj**   |
|----------------|---------------------------|---------------------------|---------------------------|------------------------------|
| BC             | **3980.27** ± 993.58      |                           |                           |                              |
| CCIL           | **3968.34** ± 1017.09     |                           |                           |                              |
| NN+LWR         | 3451.84 ± 1388.75         |                           |                           |                              |
| NS+LWR         | **4106.33** ± 454.94      |                           |                           |                              |
| NS+LWR+DTW     | **4106.19** ± 458.95      |                           |                           |                              |
