import numpy as np
import matplotlib.pyplot as plt


class NewtonianPosition(object):
    def __init__(self, nodes, edges, top=None, bottom=None):
        self.nodes = nodes
        self.edges = edges
        self.top = np.zeros(len(self.nodes))
        self.bottom = np.zeros(len(self.nodes))
        self.weights = np.zeros((len(nodes), len(nodes)))
        for i, n in enumerate(nodes):
            for j, m in enumerate(nodes):
                for e in edges:
                    if (e[0] == n and e[1] == m) or (e[1] == n and e[0] == m):
                        self.weights[i, j] = 1.0
        if top is not None:
            for i, n in enumerate(nodes):
                if n in top:
                    self.top[i] = 1.0
        if bottom is not None:
            for i, n in enumerate(nodes):
                if n in bottom:
                    self.bottom[i] = 1.0


    def iterate(self, p, v):
        repulse = 0.1
        attract = 1.0
        top = 1.5
        bottom = 1.5
        dt = 0.1
        center = 0.1
        min_d = 0.01
        for i in range(p.shape[0]):
            v[i, :] -= p[i, :] * center
            if self.top[i] > 0:
                v[i, 0] += (2 - p[i, 0]) * top
            if self.bottom[i] > 0:
                v[i, 0] += (-2 - p[i, 0]) * bottom
            for j in range(p.shape[0]):
                if i == j:
                    continue
                dist_vec = p[i, :] - p[j, :]
                dist = np.sqrt(np.sum(dist_vec * dist_vec))
                if dist < min_d:
                    dist = min_d
                c = dist_vec * repulse / (dist * dist)
                v[i, :] += c
                if self.weights[i, j] > 0:
                    a = attract * dist
                    v[i, :] -= a * dist_vec
        p += v * dt
        v *= 0.5

    def plot(self, p, i):
        plt.clf()
        plt.scatter(p[:, 0], p[:, 1])
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.savefig("../../temp/%d.png" % i)

    def fit(self, plot=False):
        dim = 2
        p = np.random.random(size=(len(self.nodes), dim)) - 0.5
        v = np.zeros(p.shape)

        for i in range(101):
            self.iterate(p, v)
            if plot and i % 10 == 0:
                self.plot(p, i)
                print(i)

        x = p[:, 1]
        y = p[:, 0]
        if x.max() - x.min() > 0.2:
            x = (x - x.min()) / (x.max() - x.min())
        else:
            x = x - x.min() + 0.5
        y = (y - y.min()) / (y.max() - y.min())
        x = np.round(x, decimals=1)
        y = np.round(y, decimals=1)
        return x, y
