import random
import matplotlib.pyplot as plt
import time
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers

EPISODES = 10000
action = 0  # 0 - nista, 1 - levo, 2 - gore, 3 - desno

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1 # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.0045#(self.epsilon - self.epsilon_min) / EPISODES
        self.learning_rate = 0.01
        self.regularization = 1e-6
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
 #       print(act_values)
        return np.argmax(act_values)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #agent.load("lander_iter20001.h5")
    done = False
    batch_size = 256
    progress = 0
    s = time.time()
    cena = []
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        ukupan_rew = 0
        for t in range(2000):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            ukupan_rew += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                if e>=1:
                    progress = (progress*(e-1)+t)/e
                    print("Epizoda: {}, nagrada: {:.2f}, srednja nagrada: {:.2f}, vreme: {}, epsilon: {:.2f}".format(e,ukupan_rew,np.mean(cena[e-100:e]),t,agent.epsilon))
                #print("episode: {}/{}, frames passed: {}, e: {:.2}, progress: {0:.3f}"
                #     .format(e, EPISODES, time, agent.epsilon, progress))
                break
        cena.append(ukupan_rew)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 100 == 2:
            agent.save("lander_iter"+str(e)+".h5")
 #   agent.save("lander_iter20001.h5")
    print("Vreme izvrsavanja [s]")
    print(time.time() - s)
    plt.plot(cena)
    plt.show()