
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

def simple_plot():

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])

    plt.plot(x, y)
    plt.show()


def catergorical_plotting():

    data = {'apples': 10, 'oranges': 15, 'lemons': 5, 'limes': 20}
    names = list(data.keys())
    values = list(data.values())

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    axs[0].bar(names, values)
    axs[1].scatter(names, values)
    axs[2].plot(names, values)
    fig.suptitle('Categorical Plotting')
    plt.show()

def artist_animation():

    fig = plt.figure()

    ims = []

    for i in range(10):
            rand = np.random.randn(100)
            im = plt.plot(rand)
            ims.append(im)

    ani = animation.ArtistAnimation(fig, ims, interval=100)
    plt.show()

def func_animation():
    fig = plt.figure()

    def plot(data):
        plt.cla()
        rand = np.random.randn(100)
        im = plt.plot(rand)

    ani = animation.FuncAnimation(fig, plot, interval=100)
    plt.show()

class Simulater:

    def __init__(self):
        self._cnt = 0
        self._x = np.array([], dtype=float)
        self._y = np.array([], dtype=float)

    def run(self):

        self.visualize()

    def simulate(self):

        self._x = np.append(self._x, self._cnt)
        self._y = np.append(self._y, self._cnt)
        self._cnt = self._cnt + 1

    def plot(self, data):
        self.simulate()
        plt.plot(self._x, self._y)

    def visualize(self):
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, partial(self.plot), interval=1000)
        plt.show()



if __name__ == "__main__":

    Simulater().run()