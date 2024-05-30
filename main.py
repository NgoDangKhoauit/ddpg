from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning
import os

def play_multiple_times(agent, num_episodes, game, save_freq=25, random_seed=42, type='train'):
    np.random.seed(random_seed)
    score_history = []
    
    if type.lower() == 'train':
        env = gym.make(game, render_mode=None)
        game_dir = os.path.join('models', game)
        filename = game + '-alpha000025-beta00025-400-300.png'
    else:
        env = gym.make(game, render_mode='human')
    
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

            with open(os.path.join(game_dir, 'scores.txt'), 'w') as f:
                for score in score_history:
                    f.write(str(score) + '\n')
        
        if type.lower() == 'train':
            print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        else:
            print('episode ', i, 'score %.2f' % score)
            
    return score_history

if __name__ == '__main__':
    game = 'LunarLanderContinuous-v2'
    type = 'train'
    env = gym.make(game)
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                  batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2, 
                  game=game)

    score_history = play_multiple_times(agent, 2000, game, type=type)
    if type.lower() == 'train':
        game_dir = os.path.join('models', game)
        filename = game + '-alpha000025-beta00025-400-300.png'
        plotLearning(score_history, os.path.join(game_dir, filename), window=100)