# Proximal-Policy-Optimaization-PPO-Implementation

This project trains an AI to control a self-driving bike in a simulated environment using the Proximal Policy Optimization (PPO) algorithm. The goal is for the bike to learn how to navigate a road and avoid obstacles.
## Files

1. PPO_Environment.py: A custom environment for the bike simulation, built on the gymnasium library. It defines the bike's physics, a reward system, and the state and action spaces for the AI.

2. PPO_Network.py: Contains the neural networks (actor and critic) that make up the AI's "brain." The actor decides on actions (steering and acceleration), and the critic evaluates how good those actions were.

3. PPO_Training.ipynb: This is where the training happens. It runs the PPO algorithm to teach the AI to drive the bike better over time. It also includes code to track and visualize the learning process.

4. PPO_Testing.py: A script to see a trained AI in action. It loads a saved model and lets you watch the bike navigate the environment.

## How It Works

The AI learns through trial and error. It gets a reward for doing good things (like moving forward quickly) and a penalty for bad things (like crashing or going off-road). The PPO algorithm helps the AI figure out which actions lead to the best rewards over time.
