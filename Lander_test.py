import random
import gym
import numpy as np
from gym import wrappers
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

Tsr = 0
N = 20
simTime = 1000
env = gym.make('LunarLander-v2')
#env = wrappers.Monitor(env, './film2/lander')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
done = False
model = Sequential()
model.add(Dense(128, input_dim = state_size, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(action_size, activation ='linear'))
model.load_weights('lander_iter8002.h5')

for e in range(N):
   state = env.reset()
   state = np.reshape(state, [1, state_size])
   rewSum = 0
   for time in range(simTime):
       env.render()
       if np.random.rand()>5:
           action = env.action_space.sample()
           state, reward, done, _ = env.step(np.argmax(action))
       else:
           action = model.predict(state)
           state, reward, done, _ = env.step(np.argmax(action[0]))
       state = np.reshape(state, [1, state_size])
       rewSum += reward
       if done:
           Tsr += time
           print("episode: {}, time: {} score:{:.2f}"
                   .format(e, time, rewSum))
           break