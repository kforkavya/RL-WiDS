# Module 1

In this module, we will be solving 2 (relatively) simple exercises to strengthen our grasp on basic RL techniques.

Your mission, should you choose to accept it is as follows:

## Assignment-1

For your first assignment you have to implement some policies for the n-Armed Bandit problem.

The file structure, inside the directory `n-Bandits`:

* `bandits.py` contains the class for the bandit. You do not need to edit this (unless you want to add input formats, for eg. batchwise). Currently implemented two types of distribution for bandit arms:
  * `"Bernoulli"` : Returns a reward of 1 with fixed probabilities.

  * `"Gaussian"` : Returns reward from a Gaussian Distribution with a fixed `mean` and some `variance` (dependent on the arm). The variance of a given arm may vary (boundedly) with time, whereas mean remains the same for every arm. For those more well-versed in probability theory, can you prove the convergence of your agent's estimate of the mean using the Law of Large Numbers?

    The class `Bandit` also keeps tract of $\text{regret} = k \cdot \text{optimal-reward} - \sum R_t$ which can be accessed at any timestamp using `get_regret()` function.

    Note that for Thompson sampling the Beta Distribution (```np.random.beta```) is the conjugate prior for a Bernoulli Distribution.

* `agents.py` contains the class for agents and sub classes for each policy type. You only need to implement the subclasses (again, unless you want to add something or fix bugs).

* `results.py` show us your results -> train the algorithms and plot the graphs. Be creative.


## Assignment-2

In your second assignment, we will be starting with a simple game and use Q-learning to "solve" the environment.

#### Gymnasium

Provides environments to represent RL problems with a simple interface to interact with. 

Documentation: https://gymnasium.farama.org/

A basic usage of gymnasium is provided in the code, but if you are interested in pursuing RL further, familiarising yourself with it would be highly useful.

### Mountain Car

The environment we will be solving in the assignment. The task and documentation can be found here:

https://gymnasium.farama.org/environments/classic_control/mountain_car/

### Basic Strategy: 

Since the observation space here is continous, but classic RL algorithms require discrete observation space, we will first be discretizing the observation space by grouping nearby values together (a technique mentioned in Sutton and Barto in the chapter of function approximations) and considering them as a single state.

Then, the method goes as usual, act $\epsilon$ greedily, and keep updating the state-action pair values based on Q-Learning update rule:

$Q(s,a) \leftarrow Q(s,a) + \alpha (r_t + \gamma \text{max}_{a'} Q(s', a') - Q(s,a))$

You are free to play along with the hyperparameters, but one set of them have been provided to you. It is highly sugggested to try different values though, as the ones provided may not be the optimal ones and later, you will have to decide these hyperparameters yourselves.

Once done with the basic requirements of assignment, you may try to plot various quantities to check the learning performance of agent. Creativity is encouraged. You may also save the q-table every few training episodes.

Don't forget to render the game at regular intervals. Nothing makes RL more satisfying than seeing your agent learn new behaviours in real time!
