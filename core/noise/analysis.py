import numpy as np
import matplotlib.pyplot as plt
import typing
from .sources import Octaves, OutletNoiseSource, InletNoiseSource
from .barriers import Channel, OpenSpace
import pandas as pd


def get_sum_sound_power_level(L_p_arr: typing.List[np.ndarray]):
    a = np.zeros([L_p_arr[0].shape[0]])
    for L_p in L_p_arr:
        a += 10**(0.1 * L_p)
    return 10 * np.log10(a)


class Room:
    def __init__(self, volume, barrier_square, L_allow, L_out, room_coef=1 / 10, n_barrier=1):
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


class TurbineNoise:
    def __init__(
            self,
            inlet_source: InletNoiseSource,
            outlet_source: OutletNoiseSource,
            inlet_channel: Channel,
            outlet_channel: Channel,
            open_space: OpenSpace,
            switchboard: Room,
    ):
        self.inlet_source = inlet_source
        self.outlet_source = outlet_source
        self.inlet_channel = inlet_channel
        self.outlet_channel = outlet_channel
        self.open_space = open_space
        self.switchboard = switchboard
        self.L_p_near_board_sum = None
        self.L_p_near_board_list = None

    def compute(self):
        self.inlet_source.compute()
        self.outlet_source.compute()
        self.inlet_channel.compute()
        self.outlet_channel.compute()
        self.open_space.compute()
        self.L_p_near_board_list = []

        for element in self.inlet_channel.elements:
            self.L_p_near_board_list.append(self.inlet_source.L_p - element.delta_L_p)
        self.L_p_near_board_list.append(self.inlet_source.L_p - self.inlet_channel.delta_L_p)
        for element in self.outlet_channel.elements:
            self.L_p_near_board_list.append(self.outlet_source.L_p - element.delta_L_p)
        self.L_p_near_board_list.append(self.outlet_source.L_p - self.outlet_channel.delta_L_p)

        self.L_p_near_board_sum = get_sum_sound_power_level(self.L_p_near_board_list)

        self.switchboard.L_out = self.L_p_near_board_sum
        self.switchboard.compute()

    def plot_channel_noise_drop(self, figsize=(8, 6), fname=None):
        plt.figure(figsize=figsize)
        plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p, lw=1.5,
                 color='red', label=r'$Вход\ в\ ГТУ$')
        plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p - self.inlet_channel.delta_L_p,
                 lw=1.5, c='red', ls='--', label=r'$Вход\ в\ канал\ КОВУ$')
        for n, element in enumerate(self.inlet_channel.elements):
            if n == 0:
                plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p - element.delta_L_p,
                         lw=1.5, c='red', ls=':', label=r'$Стенки\ участков\ КОВУ$')
            else:
                plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p - element.delta_L_p,
                         lw=1.5, c='red', ls=':')

        plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p, lw=1.5,
                 color='blue', label=r'$Выход\ из\ ГТУ$')
        plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p - self.outlet_channel.delta_L_p,
                 lw=1.5, c='blue', ls='--', label=r'$Выход\ из\ канала\ вых.\ у-ва$')
        for n, element in enumerate(self.outlet_channel.elements):
            if n == 0:
                plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p - element.delta_L_p,
                         lw=1.5, c='blue', ls=':', label=r'$Стенки\ участков\ канала\ вых.\ у-ва$')
            else:
                plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p - element.delta_L_p,
                         lw=1.5, c='blue', ls=':')

        plt.grid()
        plt.xscale('log')
        plt.xlabel(r'$f,\ Гц$', fontsize=14)
        plt.ylabel(r'$L_p,\ Дб$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10)
        if fname:
            plt.savefig(fname)
        plt.show()

    def plot_inlet_open_space_noise_drop(self, figsize=(8, 6), fname=None):
        plt.figure(figsize=figsize)
        plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p - self.inlet_channel.delta_L_p,
                 lw=1.5, c='red', ls='--', label=r'$Вход\ в\ канал\ КОВУ$')
        for n, element in enumerate(self.inlet_channel.elements):
            if n == 0:
                plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p - element.delta_L_p,
                         lw=1.5, c='red', ls=':', label=r'$Стенки\ участков\ КОВУ$')
            else:
                plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p - element.delta_L_p,
                         lw=1.5, c='red', ls=':')

        plt.plot(self.inlet_source.octaves.octave_centers,
                 self.inlet_source.L_p - self.inlet_channel.delta_L_p - self.open_space.delta_L_p,
                 lw=1.5, c='blue', ls='--', label=r'$Вход\ в\ канал\ ГТУ\ через\ %.1f\ м$' % self.open_space.r)
        for n, element in enumerate(self.inlet_channel.elements):
            if n == 0:
                plt.plot(self.inlet_source.octaves.octave_centers,
                         self.inlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p,
                         lw=1.5, c='blue', ls=':', label=r'$Стенки\ участков\ КОВУ\ через\ %.1f\ м$' % self.open_space.r)
            else:
                plt.plot(self.inlet_source.octaves.octave_centers,
                         self.inlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p,
                         lw=1.5, c='blue', ls=':')

        plt.grid()
        plt.xscale('log')
        plt.xlabel(r'$f,\ Гц$', fontsize=14)
        plt.ylabel(r'$L_p,\ Дб$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10)
        if fname:
            plt.savefig(fname)
        plt.show()

    def plot_outlet_open_space_noise_drop(self, figsize=(8, 6), fname=None):
        plt.figure(figsize=figsize)
        plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p - self.outlet_channel.delta_L_p,
                 lw=1.5, c='red', ls='--', label=r'$Выход\ из\ канала\ вых.\ у-ва$')
        for n, element in enumerate(self.outlet_channel.elements):
            if n == 0:
                plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p - element.delta_L_p,
                         lw=1.5, c='red', ls=':', label=r'$Стенки\ участков\ канала\ вых.\ у-ва$')
            else:
                plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p - element.delta_L_p,
                         lw=1.5, c='red', ls=':')

        plt.plot(self.outlet_source.octaves.octave_centers,
                 self.outlet_source.L_p - self.outlet_channel.delta_L_p - self.open_space.delta_L_p,
                 lw=1.5, c='blue', ls='--', label=r'$Выход\ из\ канала\ вых.\ у-ва\ через\ %.1f\ м$' % self.open_space.r)
        for n, element in enumerate(self.outlet_channel.elements):
            if n == 0:
                plt.plot(self.outlet_source.octaves.octave_centers,
                         self.outlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p,
                         lw=1.5, c='blue', ls=':',
                         label=r'$Стенки\ участков\ канала\ вых.\ у-ва\ через\ %.1f\ м$' % self.open_space.r)
            else:
                plt.plot(self.outlet_source.octaves.octave_centers,
                         self.outlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p,
                         lw=1.5, c='blue', ls=':')

        plt.grid()
        plt.xscale('log')
        plt.xlabel(r'$f,\ Гц$', fontsize=14)
        plt.ylabel(r'$L_p,\ Дб$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10)
        if fname:
            plt.savefig(fname)
        plt.show()

    def plot_near_switchboard_noise(self, figsize=(8, 6), fname=None):
        plt.figure(figsize=figsize)

        plt.plot(self.inlet_source.octaves.octave_centers,
                 self.inlet_source.L_p - self.inlet_channel.delta_L_p - self.open_space.delta_L_p,
                 lw=1.5, c='red', ls='--', label=r'$Вход\ в\ канал\ ГТУ$')
        for n, element in enumerate(self.inlet_channel.elements):
            if n == 0:
                plt.plot(self.inlet_source.octaves.octave_centers,
                         self.inlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p,
                         lw=1.5, c='red', ls=':', label=r'$Стенки\ участков\ КОВУ$')
            else:
                plt.plot(self.inlet_source.octaves.octave_centers,
                         self.inlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p,
                         lw=1.5, c='red', ls=':')

        plt.plot(self.outlet_source.octaves.octave_centers,
                 self.outlet_source.L_p - self.outlet_channel.delta_L_p - self.open_space.delta_L_p,
                 lw=1.5, c='blue', ls='--', label=r'$Выход\ из\ канала\ вых.\ у-ва$')
        for n, element in enumerate(self.outlet_channel.elements):
            if n == 0:
                plt.plot(self.outlet_source.octaves.octave_centers,
                         self.outlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p,
                         lw=1.5, c='blue', ls=':',
                         label=r'$Стенки\ участков\ канала\ вых.\ у-ва$')
            else:
                plt.plot(self.outlet_source.octaves.octave_centers,
                         self.outlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p,
                         lw=1.5, c='blue', ls=':')

        plt.plot(self.inlet_source.octaves.octave_centers, self.L_p_near_board_sum, c='black', lw=2,
                 label=r'$Суммарный\ шум\ у\ щита\ управления$')
        plt.plot(self.inlet_source.octaves.octave_centers, self.switchboard.L_allow, c='black', lw=2, ls='--',
                 label=r'$Допустимый\ шум\ внутри\ щита\ управления$')

        plt.grid()
        plt.xscale('log')
        plt.xlabel(r'$f,\ Гц$', fontsize=14)
        plt.ylabel(r'$L_p,\ Дб$', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10)
        if fname:
            plt.savefig(fname)
        plt.show()

    def get_data(self) -> pd.DataFrame:
        columns = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        records = [
            list(self.inlet_source.L_p),
            list(self.outlet_source.L_p),
            list(self.inlet_channel.delta_L_p_out),
            list(self.inlet_channel.delta_L_p),
            list(self.outlet_channel.delta_L_p_out),
            list(self.outlet_channel.delta_L_p),
            list(self.open_space.delta_L_p),
            list(self.switchboard.L_out),
            list(self.switchboard.L_allow),
            list(self.switchboard.R),
        ]
        index = [
            [
                'Воздухозабор',
                'Выхлоп',
                'Канал вх-ого устр-ва',
                'Канал вх-ого устр-ва',
                'Канал вых-ого устр-ва',
                'Канал вых-ого устр-ва',
                'Открытое пр-во',
                'Щит управления',
                'Щит управления',
                'Щит управления',
            ],
            [
                '$L_p$',
                '$L_p$',
                '$\Delta L_{p\ вых}$',
                '$\Delta L_p$',
                '$\Delta L_{p\ вых}$',
                '$\Delta L_p$',
                '$\Delta L_p$',
                '$L_{нар}$',
                '$L_{доп}$',
                '$R$',
            ]
        ]
        return pd.DataFrame.from_records(records, index=index, columns=columns)






