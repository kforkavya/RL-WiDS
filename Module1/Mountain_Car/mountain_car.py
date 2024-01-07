import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class QAgent:
    def __init__(self, env: str) -> None:
        self.env_name = env
        self.env = gym.make(env)
        self.state = self.env.reset()[0]

        self.observation_space_size = len(self.state)
        self.actions = self.env.action_space.n
        self.observation_space_low = self.env.observation_space.low
        self.observation_space_high = self.env.observation_space.high

        self.discrete_sizes = [25, 25]
        self.alpha = 0.1
        self.gamma = 0.99  # Adjusted gamma for potentially longer-term rewards

        self.num_train_episodes = 25000
        self.epsilon = 1
        self.num_episodes_decay = 15000
        self.epsilon_decay = self.epsilon / self.num_episodes_decay

        self.q_table = np.random.uniform(low=-1, high=1, size=(*self.discrete_sizes, self.actions))

    def get_state_index(self, state):
        indices = []
        low = self.observation_space_low
        high = self.observation_space_high
        for i in range(len(state)):
            r = self.discrete_sizes[i]
            r = float(high[i] - low[i]) / r
            s = state[i]
            indices.append(int((s - low[i]) / r))
        return tuple(indices)

    def update(self, state, action, reward, next_state, is_terminal):
        state_indices = self.get_state_index(state)
        next_state_indices = self.get_state_index(next_state)
        target = reward + self.gamma * self.q_table[next_state_indices].max()
        error = target - self.q_table[state_indices][action]
        self.q_table[state_indices, action] += self.alpha * error

        if is_terminal:
            self.state = self.env.reset()[0]

    def get_action(self):
        state_indices = self.get_state_index(self.state)

        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            q_values = self.q_table[state_indices]
            action = np.argmax(q_values)
            return action

    def env_step(self):
        action = self.get_action()
        next_state, reward, terminated, truncated, info = self.env.step(action)

        self.update(self.state, action, reward, next_state, terminated and not truncated)
        self.state = next_state

        return terminated or truncated

    def agent_eval(self):
        eval_env = gym.make(self.env_name, render_mode="human")
        done = False
        eval_state = eval_env.reset()[0]
        while not done:
            action = self.get_greedy_action()
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            eval_env.render()
            done = terminated or truncated
            eval_state = next_state

    def train(self, eval_intervals):
        for episode in range(1, self.num_train_episodes + 1):
            done = False
            if episode % 100 == 0:
                print(f"Episode {episode}")
            while not done:
                done = self.env_step()
            self.state = self.env.reset()[0]
            self.epsilon = max(0, self.epsilon - self.epsilon_decay)

            if episode % eval_intervals == 0:
                self.agent_eval()

    def get_greedy_action(self):
        state_indices = self.get_state_index(self.state)
        q_values = self.q_table[state_indices]
        action = np.argmax(q_values)
        return action

    def test_agent(self, num_episodes=10):
        eval_env = gym.make(self.env_name, render_mode="human")
        for episode in range(1, num_episodes + 1):
            done = False
            state = eval_env.reset()[0]
            while not done:
                action = self.get_greedy_action()
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                eval_env.render()
                done = terminated or truncated
                state = next_state

        self.env.close()

    def plot_rewards(self, rewards_per_episode):
        plt.figure(figsize=(10, 6))
        for i, episode_rewards in enumerate(rewards_per_episode):
            plt.plot(episode_rewards, label=f'Episode {i + 1}')

        plt.title('Cumulative Rewards per Step')
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    agent = QAgent('MountainCar-v0')
    agent.train(eval_intervals=1000)
    print("Training done!")
    agent.test_agent()
