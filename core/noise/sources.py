import numpy as np
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions as gd
from scipy.interpolate import interp1d
from abc import ABCMeta, abstractmethod, abstractproperty
from scipy.optimize import newton
from gas_turbine_cycle.gases import IdealGas


class Octaves:
    def __init__(self):
        self._octave_centers = np.array([
            63, 125, 250, 500, 1000, 2000, 4000, 8000
        ])
        self._octave_bounds = np.array([
            [45, 90], [90, 180], [180, 355], [355, 710], [710, 1400], [1400, 2800], [2800, 5600],
            [5600, 11200]
        ])
        self._third_octave_centers = np.array([
            50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000
        ])
        self._third_octave_bounds = np.array([
            [44.7, 56.1], [56.1, 71], [71, 89], [89, 112],
            [112, 141], [141, 179], [179, 224], [224, 281], [281, 355], [355, 447], [447, 561], [561, 710],
            [710, 895], [895, 1120], [1120, 1410], [1410, 1790], [1790, 2240], [2240, 2810], [2810, 3550],
            [3550, 4470], [4470, 5610], [5610, 7100], [7100, 8950], [8950, 11200]
        ])

    @property
    def octave_centers(self) -> np.ndarray:
        return self._octave_centers

    @property
    def octave_bounds(self) -> np.ndarray:
        return self._octave_bounds

    @property
    def third_octave_centers(self) -> np.ndarray:
        return self._third_octave_centers

    @property
    def third_octave_bounds(self) -> np.ndarray:
        return self._third_octave_bounds

    @classmethod
    def _get_center(cls, f, centers: np.ndarray, bounds: np.ndarray):
        for center, bound in zip(centers, bounds):
            if bound[0] < f <= bound[1]:
                return center

    def get_octave_center(self, frequency):
        return self._get_center(frequency, self.octave_centers, self.octave_bounds)

    def get_third_octave_center(self, frequency):
        return self._get_center(frequency, self.third_octave_centers, self.third_octave_bounds)


class NoiseSource(metaclass=ABCMeta):
    def __init__(self):
        self.octaves = Octaves()

    @abstractmethod
    def _L_p_sum_get(self):
        pass

    @abstractmethod
    def _L_p_get(self):
        pass

    L_p_sum = abstractproperty(fget=_L_p_sum_get)
    L_p = abstractproperty(fget=_L_p_get)

    @abstractmethod
    def get_L_p(self, frequency):
        pass

    @abstractmethod
    def compute(self):
        pass


class InletNoiseSource(NoiseSource):
    def __init__(self, n, H_ad, D1_out, eta_ad_stag, G, a1, T1, p1_stag, k, R, lam1, blade_num: int):
        NoiseSource.__init__(self)
        self.n = n
        self.H_ad = H_ad
        self.D1_out = D1_out
        self.eta_ad_stag = eta_ad_stag
        self.G = G
        self.a1 = a1
        self.T1 = T1
        self.p1_stag = p1_stag
        self.k = k
        self.R = R
        self.lam1 = lam1
        self.blade_num = blade_num
        self.f_rel_base = blade_num
        self.P0 = 10e-12
        self.delta_L_p_base_num = 6
        self.delta_L_p_init = np.array([26, 24, 22.5, 21, 16, 12, 3, 5, 6, 7, 9, 11, 17, 18.5])
        self.delta_L_p_interp = None
        self.frequency_interp = None
        self.pi1 = None
        self.p1 = None
        self.rho1 = None
        self.P = None
        self._L_p_sum = None
        self.delta_L_p = None
        self._L_p = None

    def _get_delta_L_p_and_frequency(self):
        f_base = self.f_rel_base * self.n / 60
        third_octave_center_base = self.octaves.get_third_octave_center(f_base)
        third_octave_center_base_index = list(self.octaves.third_octave_centers).index(third_octave_center_base)
        frequency = self.octaves.third_octave_centers[
                    third_octave_center_base_index - self.delta_L_p_base_num:
                    third_octave_center_base_index + len(self.delta_L_p_init) - self.delta_L_p_base_num]
        return self.delta_L_p_init[0: frequency.shape[0]], frequency

    def compute(self):
        self.delta_L_p_interp, self.frequency_interp = self._get_delta_L_p_and_frequency()
        self.pi1 = gd.pi_lam(self.lam1, self.k)
        self.p1 = self.p1_stag * self.pi1
        self.rho1 = self.p1 / (self.R * self.T1)
        self.P = (0.5 * ((1 - self.eta_ad_stag) / self.eta_ad_stag)**2 * self.G**2 * self.H_ad**2 /
                  (self.rho1 * self.a1**3 * self.D1_out**2))
        self._L_p_sum = 10 * np.log10(self.P / self.P0)
        self.delta_L_p = interp1d(np.log10(self.frequency_interp), self.delta_L_p_interp,
                                  bounds_error=False, fill_value='extrapolate')(np.log10(self.octaves.octave_centers))
        self._L_p = self.L_p_sum - self.delta_L_p

    def get_L_p(self, frequency):
        return interp1d(np.log10(self.frequency_interp), self.delta_L_p_interp,
                        bounds_error=False, fill_value='extrapolate')(np.log10(frequency)).__float__()

    def _L_p_sum_get(self):
        return self._L_p_sum

    def _L_p_get(self):
        return self._L_p

    L_p = property(fget=_L_p_get)
    L_p_sum = property(fget=_L_p_sum_get)


class Outlet:
    def __init__(self, work_fluid: IdealGas, height, width, G, G_fuel, T_stag_in, p_stag_in, sigma):
        self.work_fluid = work_fluid
        self.height = height
        self.width = width
        self.G = G
        self.G_fuel = G_fuel
        self.T_stag_in = T_stag_in
        self.p_stag_in = p_stag_in
        self.sigma = sigma

        self.T_stag_out = None
        self.p_stag_out = None
        self.fuel_content = None
        self.F_out = None
        self.alpha = None
        self.c_p = None
        self.k = None
        self.c_out = None
        self.static_out = None
        self.T_out = None
        self.p_out = None
        self.rho_out = None
        self.lam_out = None

    @classmethod
    def get_static(cls, c, T_stag, p_stag, k, R):
        a_cr = gd.a_cr(T_stag, k, R)
        lam = c / a_cr
        tau = gd.tau_lam(lam, k)
        pi = gd.pi_lam(lam, k)
        T = T_stag * tau
        p = p_stag * pi
        rho = p / (R * T)
        return T, p, rho, lam

    def compute(self):
        self.T_stag_out = self.T_stag_in
        self.p_stag_out = self.p_stag_in * self.sigma
        self.fuel_content = self.G_fuel / (self.G - self.G_fuel)
        self.F_out = self.width * self.height
        self.alpha = 1 / (self.work_fluid.l0 * self.fuel_content)
        self.c_p = self.work_fluid.c_p_real_func(self.T_stag_in, alpha=self.alpha)
        self.k = self.work_fluid.k_func(self.c_p)
        c_init = 0.7 * self.G / (self.F_out * self.p_stag_out / (self.T_stag_in * self.work_fluid.R))
        self.c_out = newton(
            lambda c: self.F_out * c * self.get_static(c, self.T_stag_out, self.p_stag_out,
                                                       self.k, self.work_fluid.R)[2] - self.G,
            x0=c_init
        )
        self.static_out = self.get_static(self.c_out, self.T_stag_out, self.p_stag_out, self.k, self.work_fluid.R)
        self.T_out = self.static_out[0]
        self.p_out = self.static_out[1]
        self.rho_out = self.static_out[2]
        self.lam_out = self.static_out[3]


class OutletNoiseSource(NoiseSource):
    def __init__(self, c_out, rho_out, width, height):
        NoiseSource.__init__(self)
        self.c_out = c_out
        self.rho_out = rho_out
        self.width = width
        self.height = height
        self.F_out = width * height
        self.Sh = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
        self.delta_L_p_interp = np.array([20, 11.5, 6, 4, 7, 11, 13, 18.5, 21.5])
        self._L_p_sum = None
        self.d_h_out = None
        self.frequency_interp = None
        self.delta_L_p = None
        self._L_p = None

    def compute(self):
        self._L_p_sum = 80 * np.log10(self.c_out) + 20 * np.log10(self.rho_out) + 10 * np.log10(self.F_out) - 44
        self.d_h_out = 2 * self.F_out / (self.width + self.height)
        self.frequency_interp = self.Sh * self.c_out / self.d_h_out
        self.delta_L_p = interp1d(np.log10(self.frequency_interp), self.delta_L_p_interp,
                                  bounds_error=False, fill_value='extrapolate')(np.log10(self.octaves.octave_centers))
        self._L_p = self._L_p_sum - self.delta_L_p

    def get_L_p(self, frequency):
        return interp1d(np.log10(self.frequency_interp), self.delta_L_p_interp,
                        bounds_error=False, fill_value='extrapolate')(np.log10(frequency)).__float__()

    def _L_p_sum_get(self):
        return self._L_p_sum

    def _L_p_get(self):
        return self._L_p

    L_p = property(fget=_L_p_get)
    L_p_sum = property(fget=_L_p_sum_get)


if __name__ == '__main__':
    in_noise = InletNoiseSource(11000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 18)
    in_noise.compute()


