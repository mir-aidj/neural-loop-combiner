from matplotlib import pyplot as plt
import numpy as np  
    
def plot_layout(layout, save_path=''):
    x = np.array([ i for i in range(0, layout.shape[0] + 1)])
    y = np.array([ i for i in range(0, layout.shape[1] + 1)])
    plt.pcolormesh(x, y, layout.T)
    plt.ylabel('loop')
    plt.xlabel('bar')
    
    if save_path != '':
        if '.png' not in save_path:
            save_path = f'{save_path}.png'
        plt.savefig(save_path)
    
    plt.show()