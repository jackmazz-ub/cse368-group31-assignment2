import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random

# Toggle if you want additional information to be printed during the testing phase
DEBUG_MODE = True

# Print state information if debug mode is enabled
def debug_print_state(env, step=0, action="N/A", reward=0, total_reward=0):
    if not DEBUG_MODE:
        return
    
    print("=" * 50)
    print(f"step {step}, Action: {action}, Reward:{reward}")   
    print(f"Total reward: {total_reward}")
    print(env.render())

# Initialize the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode="ansi")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize the Q-table
scores = np.zeros((n_states, n_actions))

alpha = 0.8 # Learning rate
gamma = 0.9 # Discount factor

# Exploration rate
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.1

episodes = 2000
max_steps = 100

# Episodic reward tracker
rewards = []

# Training loop
for episode in range(episodes):

    # Reset the state and reward tracker
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
    
        # Choose to explore or exploit based on P(epsilon)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # explore
        else:
            action = np.argmax(scores[state]) # exploit
            
        # Take the chosen action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        best_action = np.argmax(scores[next_state]) # Estimated next best action
        best_next = scores[next_state,best_action]  # Q-value of estimated best action
        
        # Update the Q-table
        scores[state, action] += alpha*(reward + gamma*(best_next) - scores[state,action])
        
        # Update the state and reward tracker
        state = next_state
        total_reward += reward
        
        # If the episode ended, stop the loop
        if terminated or truncated:
            break
            
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon*epsilon_decay)
    
    # Save total reward for this episode
    rewards.append(total_reward)
    
    # Print episode information on every 200th episode
    if (episode + 1) % 200 == 0:
        print(f"Episode {episode+1}/{episodes} complete, total_reward: {total_reward}")

print("\nTraining complete!")
print("Final scores:")
print(scores)

# Plot the learning curve
plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'))
plt.title("Q-Learning on Taxi")
plt.xlabel("Episodes")
plt.ylabel("Average Reward (50-episode rolling)")
plt.savefig("plots/learning-curve-plot.pdf", format="pdf") 

# Reset the environment
state, info = env.reset()
total_test_reward = 0

# Test the trained agent
print("\n--- TESTING TRAINED AGENT ---")
for step in range(max_steps):
    action = np.argmax(scores[state]) # exploit
    
    # Take the next action and accumulate the reward
    next_state, reward, terminated, truncated, info = env.step(action)
    total_test_reward += reward
    
    # Print state information if debug mode is enabled
    debug_print_state(env, step+1, action, reward, total_test_reward)
    
    # Update the state
    state = next_state
    
    # If the episode ended, stop the loop
    if terminated or truncated:
        break

print("Total reward after training:", total_test_reward)
env.close()

