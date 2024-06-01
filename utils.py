import matplotlib.pyplot as plt 
import numpy as np
    
def plotLearning(scores, filename, window=100, figsize=(15, 7)):
    window = 100
    average_y_q = []
    for i in range(len(scores) - window + 1):
        average_y_q.append(np.mean(scores[i:i+window]))

        
    for ind in range(window -1):
        average_y_q.insert(0, np.nan)
        
    plt.figure(figsize=figsize)
    plt.plot(np.arange(len(scores)), average_y_q, color='blue', label='ddpg')
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel(f'Average over {window} episodes')
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close()