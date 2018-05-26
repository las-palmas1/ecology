import numpy as np
import matplotlib.pyplot as plt
import typing
from .sources import Octaves


def get_sum_sound_power_level(L_p_arr: typing.List[np.ndarray]):
    a = np.zeros([L_p_arr[0].shape[0]])
    for L_p in L_p_arr:
        a += 10**(0.1 * L_p)
    return 10 * np.log10(a)


class Room:
    def __init__(self, volume, barrier_square, L_allow, L_out, room_coef=1 / 10, n_barrier=1, ):
        self.volume = volume
        self.barrier_square = barrier_square
        self.L_allow = L_allow
        self.L_out = L_out
        self.octaves = Octaves()
        self.room_coef = room_coef
        self.n_barrier = n_barrier
        self.B = None
        self.R = None

    @classmethod
    def _get_mu(cls, volume):
        if volume <= 200:
            return np.array([0.8, 0.75, 0.7, 0.8, 1., 1.4, 1.8, 2.5])
        elif 200 < volume <= 1000:
            return np.array([0.65, 0.62, 0.64, 0.75, 1., 1.5, 2.4, 4.2])
        elif 1000 < volume:
            return np.array([0.5, 0.5, 0.55, 0.7, 1, 1.6, 3, 6])

    def compute(self):
        self.B = self.volume / self.room_coef * self._get_mu(self.volume)
        self.R = (self.L_out + 10 * np.log10(self.barrier_square) - 10 * np.log10(self.B) +
                  6 - self.L_allow + 10 * np.log10(self.n_barrier))

    def plot(self, figsize=(8, 6), fname=None):
        plt.figure(figsize=figsize)
        plt.plot(self.octaves.octave_centers, self.L_out, lw=1.5, c='orange', label=r'$L_{p\ нар}$')
        plt.plot(self.octaves.octave_centers, self.L_allow, lw=1.5, c='black', label=r'$L_{p\ доп}$')
        plt.grid()
        plt.xscale('log')
        plt.xlabel(r'$f,\ Гц$', fontsize=12)
        plt.ylabel(r'$L_p,\ Дб$', fontsize=12)
        plt.legend(fontsize=14)
        if fname:
            plt.savefig(fname)
        plt.show()

