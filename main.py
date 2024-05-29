from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning

def play_multiple_times(env, agent, num_episodes, save_freq=25, random_seed=42, type='train'):
    np.random.seed(random_seed)
    score_history = []
    
    if type.lower() != 'train':
        agent.load_models()
        
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info, _ = env.step(act)
            if type.lower() == 'train':
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
            score += reward
            obs = new_state
            env.render()
        score_history.append(score)
        
        if type.lower() == 'train' and i % save_freq == 0:
            agent.save_models()
        
        if type.lower() == 'train':
            print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        else:
            print('episode ', i, 'score %.2f' % score)
            
    return score_history

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2', render_mode='human')
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                  batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, 
                  game="LunarLanderContinuous-v2")

    score_history = play_multiple_times(env, agent, 10, type='test')
    filename = 'LunarLander-alpha000025-beta00025-400-300.png'
    plotLearning(score_history, filename, window=100)