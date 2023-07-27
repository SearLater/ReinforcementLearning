import gym
import torch
import numpy as np
from libraries.actor_critic.actor import Actor
from libraries.actor_critic.critic import Critic
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
num_episodes = 1000
discount_factor = 0.99
learning_rate = 0.001

actor = Actor(env.observation_space.shape[0], env.action_space.n)
critic = Critic(env.observation_space.shape[0], env.action_space.n)

actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

# Define the training loop
for episode in range(num_episodes):
    # Initialize the environment
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select an action using the agent's policy
        probs = actor(torch.tensor(state, dtype=torch.float32))
        val = critic(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(np.arange(len(probs)), p=probs.detach().numpy())

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Calculate the TD error and loss
        next_val = critic(torch.tensor(next_state, dtype=torch.float32))
        err = reward + discount_factor * (next_val * (1 - done)) - val
        actor_loss = -torch.log(probs[action]) * err
        critic_loss = torch.square(err)
        loss = actor_loss + critic_loss

        # Update the network
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

        # Set the state to the next state
        state = next_state

    # Print the total reward for the episode
    print(f'Episode {episode}: Total reward = {total_reward}')