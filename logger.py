from collections.abc import Iterable
from matplotlib import pyplot as plt
import scipy.signal


class Plotter:
    def __init__(self, *labels, figsize=(16, 10)):
        plt.rcParams['figure.figsize'] = figsize

        self.timesteps = []
        self.data = [[] for _ in labels]
        self.mins = [float('inf')] * len(labels)
        self.maxes = [-float('inf')] * len(labels)
        self.figure, self.axes = plt.subplots(len(labels), 1)

        for ax, label in zip(self.axes, labels):
            ax.set_ylabel(label)

    def update(self, t, *args):
        self.timesteps.append(t)
        for i, arg in enumerate(args):
            if not isinstance(arg, Iterable):
                arg = [arg]
            data = self.data[i]
            if not data:
                data += [[] for _ in arg]

            moving_averages = []

            for a, d in zip(arg, data):
                d.append(a)

                WINDOW_LENGTH = 7
                window = max(WINDOW_LENGTH, len(d) // 16)
                window = window - window % 2 + 1
                if len(d) >= window:
                    moving_averages.append(scipy.signal.savgol_filter(d, window, 2))
                else:
                    moving_averages.append(d)

                self.mins[i] = min(self.mins[i], *moving_averages[-1])
                self.maxes[i] = max(self.maxes[i], *moving_averages[-1])

            ax = self.axes[i]
            ax.set_xlim(0, self.timesteps[-1])
            ax.set_ylim(self.mins[i], self.maxes[i])
            if ax.lines:
                for line, d in zip(ax.lines, data + moving_averages):
                    line.set_data(self.timesteps[:len(d)], d)
            else:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for d, color in zip(data, colors):
                    ax.plot(self.timesteps, d, alpha=1/4, color=color)
                for ma, color in zip(moving_averages, colors):
                    ax.plot(self.timesteps, ma, color=color)

        self.figure.canvas.draw()


class PlotterMA:
    def __init__(self, *labels, figsize=(16, 10)):
        plt.rcParams['figure.figsize'] = figsize

        self.timesteps = []
        self.data = [[] for _ in labels]
        self.moving_averages = [[] for _ in labels]
        self.figure, self.axes = plt.subplots(len(labels), 1)

        for ax, label in zip(self.axes, labels):
            ax.set_ylabel(label)

    def update(self, t, *args):
        def mean(iter):
            return sum(iter) / len(iter)

        self.timesteps.append(t)
        for i, arg in enumerate(args):
            if not isinstance(arg, Iterable):
                arg = [arg]
            data = self.data[i]
            moving_averages = self.moving_averages[i]
            if not data:
                data += [[] for _ in arg]
                moving_averages += [[] for _ in arg]

            mins = [float('inf')] * len(args)
            maxes = [-float('inf')] * len(args)
            for a, d, ma in zip(arg, data, moving_averages):
                d.append(a)

                WINDOW = 8
                del ma[-WINDOW:]
                ma.extend(mean(d[2*x-1:]) for x in range(-min(WINDOW, len(d)-1), 1))

                mins[i] = min(mins[i], *ma)
                maxes[i] = max(maxes[i], *ma)

            ax = self.axes[i]
            ax.set_xlim(0, self.timesteps[-1])
            ax.set_ylim(mins[i], self.maxes[i])
            if ax.lines:
                for line, d in zip(ax.lines, data + moving_averages):
                    line.set_data(self.timesteps[:len(d)], d)
            else:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for d, color in zip(data, colors):
                    ax.plot(self.timesteps, d, alpha=1/4, color=color)
                for ma, color in zip(moving_averages, colors):
                    ax.plot(self.timesteps, ma, color=color)

        self.figure.canvas.draw()
