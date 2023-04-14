
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

class ReplayMemory:
    def __init__(self,memlen):
        self.memory = deque(maxlen=memlen)
    
    def append(self,data):
        self.memory.append(data)
        
    def __len__(self,):
        return len(self.memory)
    
    def get_batch(self,batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch = np.array([sample[0] for sample in minibatch])
        action_batch = np.array([sample[1] for sample in minibatch])
        reward_batch = np.array([sample[2] for sample in minibatch])
        next_state_batch = np.array([sample[3] for sample in minibatch])
        done_batch = np.array([sample[4] for sample in minibatch])
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        

class QNetwork(tf.keras.Model):
    def __init__(self, arch, activations):
        super(QNetwork, self).__init__()
        
        layers = []
        for i in range(1,len(arch)):
            if i == 1: layers.append(
                Dense(arch[i], activation=activations[i-1], input_shape=(arch[i-1],)))
            else: layers.append(
                Dense(arch[i], activation=activations[i-1])
            )
                
        self.model = tf.keras.Sequential(layers)

    def call(self, inputs):
        return self.model(inputs)



class DQN:
    def __init__(self,arch,af,eta=0.001,epsilon=0.1,epsilon_decay=0.995,epsilon_min=0.01,gamma=0.95,maxlen=10000):
        self.Q = QNetwork(arch,af)
        self.Q_target = tf.keras.models.clone_model(self.Q)
        
        self.D = ReplayMemory(maxlen)
        self.action_size = arch[-1]

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=eta)
        

        
        self.arch = arch
        self.af = af
    
    def update_target_network(self):
        self.Q_target.set_weights(self.Q.get_weights())
        
    def decay_epsilon(self):
        self.epsilon = min(self.epsilon_min,self.epsilon*self.epsilon_decay)
    
    def e_greedy(self,state,env):
        if np.random.rand() < self.epsilon: return env.action_space.sample()
        
        q_values = self.Q(np.array([state]))
        return np.argmax(q_values)
    
    
    def learn(self,data,batch_size):
        self.D.append(data)
        if len(self.D) < batch_size: return 0
        
        state,action,reward,nxt_state,done = self.D.get_batch(batch_size)
        with tf.GradientTape() as tape:
            # Q-values for the s
            q_vals = self.Q(state)
            action_mask = tf.one_hot(action, self.action_size)
            q_vals = tf.reduce_sum(tf.multiply(q_vals, action_mask), axis=1)

            # Q-values for the s'
            nxt_q_vals = self.Q_target(nxt_state)
            max_nxt_q_vals = tf.reduce_max(nxt_q_vals, axis=1)
            target_q_vals = reward + self.gamma * (1 - done) * max_nxt_q_vals

            loss = tf.keras.losses.mean_squared_error(target_q_vals,q_vals)
        
        gradients = tape.gradient(loss, self.Q.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.Q.trainable_variables))
        return loss.numpy()
        
             
    

if __name__ == "__main__":
    import gym
    import pygame

    def get_surface(rgb_array):
        surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
        return surface

    def play(net,env):
        pygame.init()
        screen = pygame.display.set_mode((600,400))
        pygame.display.set_caption('CartPole')   
    
        state,_ = env.reset()
        done = False
        rewards = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: done = True

            
            action = np.argmax(net(np.array([state])))
            state, r, done, _,_ = env.step(action)
            rewards += r
            surface = get_surface(env.render())
            
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        print(rewards)
        pygame.quit()

    def train(agent,env,num_episodes=100,batch_size=32,C=100):
        steps=0
        for i in range(1,num_episodes+1):
            try:
                episode_reward = 0
                episode_loss = 0
                t = 0

                # Sample Phase
                agent.decay_epsilon()
                nxt_state = env.reset()[0]
                done = False
                while not done:
                    state = nxt_state
                    action = agent.e_greedy(state,env)
                    nxt_state,reward,done,_,_ = env.step(action)
                    episode_reward += reward
                
                    # Learning Phase
                    episode_loss += agent.learn((state,action,reward,nxt_state,done),batch_size)
                    steps +=1
                    t+=1

                    if steps % C == 0: agent.update_target_network()

                print(f"Episode: {i} Reward: {episode_reward} Loss: {episode_loss/t}")
            except KeyboardInterrupt:
                print(f"Training Terminated at Episode {i}")
                return 

    env = gym.make('CartPole-v1',render_mode= "rgb_array")
    arch = [4,4,3,2] # 4->4(sig)->3(relu)->2(lin)
    af = ["sigmoid","relu","linear"]
    agent = DQN(arch,af,eta=5e-3)
    train(agent,env,300,42)
    play(agent.Q,env)
