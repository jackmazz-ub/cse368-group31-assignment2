#/usr/bin/python3

# Goal:
# Train a Taxi agent using Q-learning to improve over time.

# Instructions:
# 1. Complete all the TODO sections.
# 2. Run your code to train and test the agent.
# 3. Observe how the taxi improves its performance!

# Step 1: Import libraries and initialize environment
import gymnasium as gym
import numpy as np
import random

# TODO: Create the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode="ansi")

# TODO: Get number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Step 2: Create the Q-table (all zeros)
scores = np.zeros((n_states, n_actions))

alpha = 0.8 # Learning rate
gamma = 0.9 # Discount factor

# Exploration rate
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.1

episodes = 2000
max_steps = 100

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
        
        best_action = np.argmax(scores[next_state]) # Estimated next best action
        best_next = scores[next_state,best_action]  # Q-value of estimated best action
        
        # TODO: Update Q-table
        scores[state, action] += alpha*(reward + gamma*(best_next) - scores[state,action])
        
        # TODO: Update state and reward tracker
        state = next_state
        total_reward += reward
        
        # If the episode ended, stop the loop
        if done or truncated:
            break
            
    # TODO: Decay epsilon
    epsilon = max(epsilon_min, epsilon*epsilon_decay)
    
    # Print episode information on every 200th episode
    if (episode + 1) % 200 == 0:
        print(f"Episode {episode+1}/{episodes} complete")

print("\nTraining complete!")
print("Final scores:")
print(scores)

# Step 4: Test the trained agent
state, info = env.reset()
total_test_reward = 0
done = False

print("\n--- TESTING TRAINED AGENT ---")
for step in range(max_steps):
    # TODO: Always pick the best action
    action = np.argmax(scores[state]) # exploit
    
    # Take the next action and accumulate the reward
    next_state, reward, done, truncated, info = env.step(action)
    total_test_reward += reward
    
    # Display the environment
    print(env.render())
    
    # If the episode ended, stop the loop
    if done or truncated:
        break
    
    # Update the state
    state = next_state

print("Total reward after training:", total_test_reward)
env.close()

