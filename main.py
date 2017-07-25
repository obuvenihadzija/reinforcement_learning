import gym
env = gym.make('BipedalWalker-v2')
print(env.reset())
for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
        break