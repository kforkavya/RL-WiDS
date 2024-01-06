from bandits import Bandit
# Import libraries if you need them
import numpy as np

class Agent:
    def __init__(self, bandit: Bandit) -> None:
        self.bandit = bandit
        self.banditN = bandit.getN()

        self.rewards = 0
        self.numiters = 0
    

    def action(self) -> int:
        '''This function returns which action is to be taken. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    def update(self, choice : int, reward : int) -> None:
        '''This function updates all member variables you may require. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    # dont edit this function
    def act(self) -> int:
        choice = self.action()
        reward = self.bandit.choose(choice)

        self.rewards += reward
        self.numiters += 1

        self.update(choice,reward)
        return reward

class GreedyAgent(Agent):
    def __init__(self, bandits: Bandit, initialQ : float) -> None:
        super().__init__(bandits)
        # add any member variables you may require
        self.Q = np.full(self.banditN, initialQ)
        self.N = np.zeros(self.banditN)
        
    # implement
    def action(self) -> int:
        return np.argmax(self.Q)

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.N[choice]+=1
        self.Q[choice] = self.Q[choice] + (reward - self.Q[choice])/self.N[choice]
        pass

class epsGreedyAgent(Agent):
    def __init__(self, bandits: Bandit, epsilon : float) -> None:
        super().__init__(bandits)
        self.epsilon = epsilon
        # add any member variables you may require
        self.Q = np.zeros(self.banditN)
        self.N = np.zeros(self.banditN)
    
    # implement
    def action(self) -> int:
        action = np.random.randint(self.banditN)
        if np.random.random() > self.epsilon:
            action = np.argmax(self.Q)
        return action

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.N[choice]+=1
        self.Q[choice] = self.Q[choice] + (reward - self.Q[choice])/self.N[choice]
        pass

class UCBAAgent(Agent):
    def __init__(self, bandits: Bandit, c: float) -> None:
        super().__init__(bandits)
        self.c = c
        # add any member variables you may require
        self.e = 0
        self.Q = np.zeros(self.banditN)
        self.N = np.zeros(self.banditN)

    # implement
    def action(self) -> int:
        action = self.e
        if self.e >= self.banditN:
            U = np.sqrt(self.c*np.log(self.e)/self.N)
            action = np.argmax(self.Q + U)
        return action

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.N[choice]+=1
        self.Q[choice] = self.Q[choice] + (reward - self.Q[choice])/self.N[choice]
        self.e+=1
        pass

class GradientBanditAgent(Agent):
    def __init__(self, bandits: Bandit, alpha : float) -> None:
        super().__init__(bandits)
        self.alpha = alpha
        # add any member variables you may require
        self.H_values = np.zeros(self.banditN)
        self.Q = np.zeros(self.banditN)

    # implement
    def action(self) -> int:
        self.action_probs = list(map(self.softmax, self.H_values))
        action = np.random.choices(list(range(self.banditN)), self.action_probs, k=1)[0]
        return action

    def softmax(self, H):
        return np.exp(H) / np.sum(np.exp(self.H_values))

    # implement
    def update(self, choice: int, reward: int) -> None:
        self.Q[choice] = self.Q[choice] + (reward - self.Q[choice])*self.alpha
        r_mean = self.Q[choice]
        error = reward - r_mean
        for a in range(self.banditN):
            if a != choice:
                self._updateActionPreference(a, error)
            else:
                self._updateActionPreference(a, error, is_curr_action=True)
        pass

    def _updateActionPreference(self, action, error, is_curr_action=False):
        action_pref = self.H_values[action]
        action_prob = self.action_probs[action]
        if is_curr_action:
            self.H_values[action] = action_pref + self.alpha * error * (1 - action_prob) 
        else:
            self.H_values[action] = action_pref - self.alpha * error * action_prob

class ThompsonSamplerAgent(Agent):
    def __init__(self, bandits: Bandit, alpha: float = 1, beta: float = 1) -> None:
        super().__init__(bandits)
        # add any member variables you may require
        self.alpha = alpha
        self.beta = beta
        self.successes = np.zeros(self.banditN)
        self.failures = np.zeros(self.banditN)

    # implement
    def action(self) -> int:
        sampled_values = np.random.beta(self.alpha + self.successes, self.beta + self.failures)
        return np.argmax(sampled_values)

    # implement
    def update(self, choice: int, reward: int) -> None:
        if reward == 1:
            self.successes[choice] += 1
        else:
            self.failures[choice] += 1
        pass

# Implement other subclasses if you want to try other strategies
