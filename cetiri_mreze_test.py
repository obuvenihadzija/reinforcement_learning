import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.optimizers import SGD, RMSprop, Adam, Adamax

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
        plt.plot(running_avg)
        plt.title("Running Average")
        plt.show()

env = gym.make('LunarLander-v2')

# build a set of samples so we can get a scaler fitted.
observation_samples = []

# play a bunch of games randomly and collect observations
for n in range(1000):
    observation = env.reset()
    observation_samples.append(observation)
    done = False
    while not done:
        action = np.random.randint(0, env.action_space.n)
        observation, reward, done, _ = env.step(action)
        observation_samples.append(observation)
observation_samples = np.array(observation_samples)


# Create scaler and fit
scaler = StandardScaler()
scaler.fit(observation_samples)

# Using the excellent Keras to build standard feedforward neural network.
# single output node, linear activation on the output
#  To keep things simple,  one NN is created per action.  So
# in this problem, 4 independant neural networks are create
# Admax optimizer is the most efficient one, using default parameters.

def create_nn():
    model = Sequential()
    model.add(Dense(128, init='lecun_uniform', input_shape=(8,)))
    model.add(Activation('relu'))
    #     model.add(Dropout(0.3)) #I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(256, init='lecun_uniform'))
    model.add(Activation('tanh'))
    #     model.add(Dropout(0.5))

    model.add(Dense(1, init='lecun_uniform'))
    model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

    #     rms = RMSprop(lr=0.005)
    #     sgd = SGD(lr=0.1, decay=0.0, momentum=0.0, nesterov=False)
    # try "adam"
    #     adam = Adam(lr=0.0005)
    adamax = Adamax()  # Adamax(lr=0.001)
    model.compile(loss='mse', optimizer=adamax)
    #     model.summary()
    return model

  # Holds one nn for each action
class Model:
    def __init__(self, env, scaler):
        self.env = env
        self.scaler = scaler
        self.models = []
        for i in range(env.action_space.n):
            model = create_nn()  # one nn per action
            self.models.append(model)

    def predict(self, s):
        X = self.scaler.transform(np.atleast_2d(s))
        return np.array([m.predict(np.array(X), verbose=0)[0] for m in self.models])

    def update(self, s, a, G):
        X = self.scaler.transform(np.atleast_2d(s))
        self.models[a].fit(np.array(X), np.array([G]), nb_epoch=1, verbose=0)

    def sample_action(self, s):
            return np.argmax(self.predict(s))
def play_one(env, model):
    observation = env.reset()
    done = False
    full_reward_received = False
    totalreward = 0
    iters = 0
    while not done:
        env.render()
        action = model.sample_action(observation)
        observation, reward, done, _ = env.step(action)

        # update the model
        totalreward += reward
        iters += 1
    return totalreward, iters

model = Model(env, scaler)
gamma = 0.99
env = wrappers.Monitor(env, './folder2/')
N = 8010
totalrewards = np.empty(N)
costs = np.empty(N)
model.models[0].load_weights("weights04300.h5")
model.models[1].load_weights("weights14300.h5")
model.models[2].load_weights("weights24300.h5")
model.models[3].load_weights("weights34300.h5")
for n in range(N):
    totalreward, iters = play_one(env, model)
    totalrewards[n] = totalreward
    #if n % 100 == 0:
     #   print("episode:", n, "iters", iters, "total reward:", totalreward, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

env.close()