#/usr/bin/python3
#Goal:
#Train a Taxi agent using Q-learning to improve over time.

#Instructions:
#1. Complete all the TODO sections.
#2. Run your code to train and test the agent.
#3. Observe how the taxi improves its performance!

# Step 1: Import libraries and initialize environment
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random

debug_mode = True

def debug_print_state(env, step=0, action="N/A", reward=0, total_reward=0):
    if not debug_mode:
        return
    
    print("=" * 50)
    print(f"step {step}, Action: {action}, Reward:{reward}")   
    print(f"Total reward: {total_reward}")
    print(env.render())

# TODO: Create the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode="ansi")

# TODO: Get number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Step 2: Create the Q-table (all zeros)
scores = np.zeros((n_states, n_actions))

# Q-learning parameters (you may adjust these)
alpha = 0.8                 # Learning rate     (0.1)
gamma = 0.5                 # Discount factor   (0.9)
epsilon = 1.0               # Exploration rate  (1.0)
epsilon_decay = 1.0         #                   (0.9995)
epsilon_min = 0.0           #                   (0.1)

episodes = 3000
max_steps = 100

# To store total rewards per episode
rewards = []

# Step 3: Training loop
for episode in range(episodes):
    # TODO: Reset environment and initialize variables
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # TODO: Choose an action (explore or exploit)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # explore
        else:
            action = np.argmax(scores[state]) # exploit

        # TODO: Take the action
        next_state, reward, done, truncated, info = env.step(action)

        # TODO: Update Q-table
        best_next = np.argmax(scores[next_state])
        scores[state, action] += alpha * (reward + gamma * best_next - scores[state, action])

        # TODO: Update state and reward tracker
        state = next_state
        total_reward += reward

        if done or truncated:
            break

    # TODO: Decay epsilon
    epsilon = max(epsilon_min, epsilon*epsilon_decay)
    
    # Save total reward for this episode
    rewards.append(total_reward)

    if (episode + 1) % 200 == 0:
        print(f"Episode {episode+1}/{episodes} complete; total_reward: {total_reward}; epsilon: {epsilon}")

print("\nTraining complete!")
print("Final scores:")
print(scores)

# Plot learning curve
plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'))
plt.title("Q-Learning on Taxi")
plt.xlabel("Episodes")
plt.ylabel("Average Reward (50-episode rolling)")
plt.savefig("images/pa2-output.pdf", format="pdf") 

# Step 4: Test the trained agent

state, info = env.reset()
total_test_reward = 0
done = False

debug_print_state(env)

print("\n--- TESTING TRAINED AGENT ---")
for step in range(max_steps):
    action = np.argmax(scores[state]) # exploit

    next_state, reward, done, truncated, info = env.step(action)
    total_test_reward += reward
    env.render()
    
    debug_print_state(env, step+1, action, reward, total_test_reward)

    if done or truncated:
        print(divider)
        print("Episodes concluded")
        break
        
    state = next_state

print("Total reward after training:", total_test_reward)
env.close()

