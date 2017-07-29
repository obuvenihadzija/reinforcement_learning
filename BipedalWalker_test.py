
import random
import gym
import numpy as np
from collections import deque
from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
env = gym.make('BipedalWalker-v2')
values = [-1, 0, 1]
actions = [[x1,x2,x3,x4] for x1 in values for x2 in values for x3 in values for x4 in values]
state_size = env.observation_space.shape[0]
action_size = len(actions)
done = False
model = Sequential()
model.add(Dense(128, input_dim=state_size, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(action_size, activation='softmax'))
#model.compile(loss = 'mse',
#                optimizer=Adam(lr=0.001))
model.load_weights('trenirano_iter21999.h5')
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]
Tsr = 0

for e in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()
        actionNum = np.argmax(model.predict(state))
        next_state, reward, done, _ = env.step(actions[actionNum])
        print(actions[actionNum])
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            Tsr += time
            print("episode: {}, score: {}"
                    .format(e, time))
            break
    if 0xFF == ord('q'):
        Tsr /= e
        print("Prosecno vreme simulacije: {}",Tsr)
        break