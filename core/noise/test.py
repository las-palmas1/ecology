import unittest
from .sources import Outlet, OutletNoiseSource, InletNoiseSource
from .barriers import *
from .analysis import Room, get_sum_sound_power_level
from gas_turbine_cycle.gases import KeroseneCombustionProducts
import matplotlib.pyplot as plt


class TestTurbineNoise(unittest.TestCase):
    def setUp(self):
        self.outlet = Outlet(
            work_fluid=KeroseneCombustionProducts(),
            height=1.2,
            width=1.1,
            G=40,
            G_fuel=1,
            T_stag_in=550,
            p_stag_in=1.05e5,
            sigma=0.99
        )
        self.inlet_source = InletNoiseSource(
            n=11000,
            H_ad=6e4,
            D1_out=0.5,
            eta_ad_stag=0.86,
            G=40,
            a1=310,
            T1=275,
            p1_stag=1e5,
            k=1.4,
            R=287,
            lam1=0.1,
            blade_num=18
        )
        self.outlet.compute()
        self.outlet_source = OutletNoiseSource(
            c_out=self.outlet.c_out,
            rho_out=self.outlet.rho_out,
            width=self.outlet.width,
            height=self.outlet.height,
        )

        self.inlet_round_channel = RoundMetalChannelElement(D=0.5, length=1.5, thickness=4e-3)
        self.inlet_turn_channel = RectangularSmoothTurnElement(
            R=0.4, width=0.8, height=0.8, angle=np.pi/2, thickness=4e-3
        )
        self.inlet_rect_channel = RectangularChannelElement(
            width=self.inlet_turn_channel.width, height=self.inlet_turn_channel.height, length=2.5, thickness=4e-3
        )
        self.inlet_adapter_element = AdapterElement(
            square_in=self.inlet_round_channel.get_section_square(),
            square_out=self.inlet_turn_channel.get_section_square(),
            length=0.5, thickness=4e-3
        )
        self.inlet_channel = Channel(
            elements=[self.inlet_round_channel, self.inlet_adapter_element,
                      self.inlet_turn_channel, self.inlet_rect_channel]
        )

        self.outlet_rect_channel = RectangularChannelElement(
            width=self.outlet.width,
            height=self.outlet.height,
            length=2, thickness=4e-3
        )
        self.outlet_round_channel = RoundMetalChannelElement(
            D=0.5,
            length=4,
            thickness=4e-3
        )
        self.outlet_adapter = AdapterElement(
            square_in=self.outlet_rect_channel.get_section_square(),
            square_out=self.outlet_round_channel.get_section_square(),
            length=0.5,
            thickness=4e-3
        )
        self.outlet_channel = Channel(
            elements=[self.outlet_rect_channel, self.outlet_adapter, self.outlet_round_channel]
        )

        self.open_space = OpenSpace(
            r=100, fi=1, omega=4*np.pi
        )

    def test_sources(self):
        self.inlet_source.compute()
        self.outlet_source.compute()
        plt.figure(figsize=(8, 6))
        plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p, lw=1.5, color='red', label='inlet')
        plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p, lw=1.5, color='blue',
                 label='outlet')
        plt.xlabel(r'$f$', fontsize=14)
        plt.ylabel(r'$L_p$', fontsize=14)
        plt.legend(fontsize=14)
        plt.xscale('log')
        plt.grid()
        plt.show()

    def test_inlet_channel(self):
        self.inlet_source.compute()
        self.inlet_channel.compute()
        plt.figure(figsize=(8, 6))
        plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p, lw=1.5, color='red', label='inlet',
                 ls='--')
        plt.plot(self.inlet_source.octaves.octave_centers,
                 self.inlet_source.L_p - self.inlet_channel.delta_L_p, lw=1.5, color='red', ls='-',
                 label='inlet + channel drop')
        plt.xlabel(r'$f$', fontsize=14)
        plt.ylabel(r'$L_p$', fontsize=14)
        plt.legend(fontsize=14)
        plt.xscale('log')
        plt.grid()
        plt.show()

    def test_outlet_channel(self):
        self.outlet_source.compute()
        self.outlet_channel.compute()
        plt.figure(figsize=(8, 6))
        plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p, lw=1.5, color='red', label='outlet',
                 ls='-')
        plt.plot(self.outlet_source.octaves.octave_centers,
                 self.outlet_source.L_p - self.outlet_channel.delta_L_p, lw=1.5, color='red', ls='--',
                 label='outlet + channel drop')
        plt.xlabel(r'$f$', fontsize=14)
        plt.ylabel(r'$L_p$', fontsize=14)
        plt.legend(fontsize=14)
        plt.xscale('log')
        plt.grid()
        plt.show()

    def test_inlet_channel_element(self):
        self.inlet_source.compute()
        self.inlet_channel.compute()
        plt.figure(figsize=(8, 6))
        plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p, lw=1.5, color='red', label='inlet',
                 ls='-')
        for element in self.inlet_channel.elements:
            plt.plot(self.inlet_source.octaves.octave_centers,
                     self.inlet_source.L_p - element.delta_L_p, lw=1.5, ls='--')
        plt.xlabel(r'$f$', fontsize=14)
        plt.ylabel(r'$L_p$', fontsize=14)
        plt.xscale('log')
        plt.grid()
        plt.show()

    def test_outlet_channel_element(self):
        self.outlet_source.compute()
        self.outlet_channel.compute()
        plt.figure(figsize=(8, 6))
        plt.plot(self.outlet_source.octaves.octave_centers, self.outlet_source.L_p, lw=1.5, color='red', label='inlet',
                 ls='-')
        for element in self.outlet_channel.elements:
            plt.plot(self.outlet_source.octaves.octave_centers,
                     self.outlet_source.L_p - element.delta_L_p, lw=1.5, ls='--')
        plt.xlabel(r'$f$', fontsize=14)
        plt.ylabel(r'$L_p$', fontsize=14)
        plt.xscale('log')
        plt.grid()
        plt.show()

    def test_delta_L_p_prime_computing(self):
        self.outlet_channel.compute()
        ar1 = self.outlet_rect_channel.delta_L_p_prime == np.zeros([8])
        ar2 = self.outlet_adapter.delta_L_p_prime == self.outlet_rect_channel.delta_L_p_length
        ar3 = (
                self.outlet_round_channel.delta_L_p_prime ==
                self.outlet_rect_channel.delta_L_p_length + self.outlet_adapter.delta_L_p_length
        )
        self.assertEqual(ar1.all(), True)
        self.assertEqual(ar2.all(), True)
        self.assertEqual(ar3.all(), True)

    def test_inlet_open_space(self):
        self.inlet_source.compute()
        self.inlet_channel.compute()
        self.open_space.compute()

        plt.figure(figsize=(8, 6))
        plt.plot(self.inlet_source.octaves.octave_centers, self.inlet_source.L_p, lw=1.5, color='red', label='inlet',
                 ls='-')

        plt.plot(self.inlet_source.octaves.octave_centers,
                 self.inlet_source.L_p - self.inlet_channel.delta_L_p, lw=1.5, color='black', ls='--')
        for element in self.inlet_channel.elements:
            plt.plot(self.inlet_source.octaves.octave_centers,
                     self.inlet_source.L_p - element.delta_L_p, lw=1.5, ls='--')

        plt.plot(self.inlet_source.octaves.octave_centers,
                 self.inlet_source.L_p - self.inlet_channel.delta_L_p - self.open_space.delta_L_p,
                 lw=1.5, color='black', ls=':')
        for element in self.inlet_channel.elements:
            plt.plot(self.inlet_source.octaves.octave_centers,
                     self.inlet_source.L_p - element.delta_L_p - self.open_space.delta_L_p, lw=1.5, ls=':')

        plt.xlabel(r'$f$', fontsize=14)
        plt.ylabel(r'$L_p$', fontsize=14)
        plt.xscale('log')
        plt.grid()
        plt.show()



