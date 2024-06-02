from ddpg_torch import Agent
import gym
import numpy as np
import os

def play_multiple_times(agent, env, num_episodes):
    np.random.seed(42)
    agent.load_models()
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, terminated, truncated, _ = env.step(act)
            score += reward
            obs = new_state
            env.render()
            done = terminated or truncated
        print('episode ', i, 'score %.2f' % score)
        
if __name__ == '__main__':
    dir = 'models/Humanoid-v4'
    max_ep = [
        '1k ep',
        '5k ep',
        '10k ep',
        '50k ep',
        '100k ep',
        '113k ep'
    ]
    for ep in max_ep:
        model_path = os.path.join(dir, ep)
        print('max ep:', ep)
        env = gym.make('Humanoid-v4', render_mode='human')
        agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]], tau=0.001, env=env,
                      batch_size=64,  layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0], 
                      model_path=model_path)
        play_multiple_times(agent, env, 10)
        env.close()