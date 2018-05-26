from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from .sources import Octaves
import typing
from scipy.interpolate import interp1d


class Insulation(metaclass=ABCMeta):
    def __init__(self):
        self.octaves = Octaves()

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def _R_get(self):
        pass

    R = abstractproperty(fget=_R_get)


class FlatSteelInsulation(Insulation):
    def __init__(self, thickness):
        Insulation.__init__(self)
        self.thickness = thickness
        self.f_b = 6 / thickness
        self.f_c = 12 / thickness
        self.R_b = 39
        self.R_c = 31
        self.k_ab = 4
        self.k_cd = 8
        self.R_a = None
        self.R_d = None
        self._R = None

    def compute(self):
        f_b_log = np.log10(self.f_b)
        f_c_log = np.log10(self.f_c)
        f_a_log = np.log10(self.octaves.octave_centers[0])
        f_d_log = np.log10(self.octaves.octave_centers[len(self.octaves.octave_centers) - 1])
        self.R_a = self.R_b - (f_b_log - f_a_log) * self.k_ab
        self.R_d = self.R_c + (f_d_log - f_c_log) * self.k_cd
        self._R = interp1d(
            [f_a_log, f_b_log, f_c_log, f_d_log], [self.R_a, self.R_b, self.R_d, self.R_c],
            bounds_error=False, fill_value='extrapolate')(np.log10(self.octaves.octave_centers))

    def _R_get(self):
        return self._R

    R = property(fget=_R_get)


class CylindricalSteelInsulation(Insulation):
    def __init__(self, D, thickness):
        Insulation.__init__(self)
        self.D = D
        self.thickness = thickness
        self.f_b = 1.6e3 / D
        self.R_b = 74 - 20 * np.log10(D / thickness)
        self.f_c = 12e1 / self.thickness
        self.R_c = 31
        self.k_ab = 6
        self.k_cd = 8
        self.R_a = None
        self.R_d = None
        self._R = None

    def compute(self):
        f_b_log = np.log10(self.f_b)
        f_c_log = np.log10(self.f_c)
        f_a_log = np.log10(self.octaves.octave_centers[0])
        f_d_log = np.log10(self.octaves.octave_centers[len(self.octaves.octave_centers) - 1])
        self.R_a = self.R_b + (f_b_log - f_a_log) * self.k_ab
        self.R_d = self.R_c + (f_d_log - f_c_log) * self.k_cd
        self._R = interp1d(
            [f_a_log, f_b_log, f_c_log, f_d_log], [self.R_a, self.R_b, self.R_d, self.R_c],
            bounds_error=False, fill_value='extrapolate')(np.log10(self.octaves.octave_centers))

    def _R_get(self):
        return self._R

    R = property(fget=_R_get)


class Barrier(metaclass=ABCMeta):

    @abstractmethod
    def _delta_L_p_get(self):
        pass

    delta_L_p = abstractproperty(fget=_delta_L_p_get)

    @abstractmethod
    def compute(self):
        pass


class ChannelElement(Barrier, metaclass=ABCMeta):
    def __init__(self):
        self.insulation: Insulation = None
        self._delta_L_p_prime = None
        self._delta_L_p = None

    @property
    def delta_L_p_prime(self):
        return self._delta_L_p_prime

    @delta_L_p_prime.setter
    def delta_L_p_prime(self, value):
        self._delta_L_p_prime = value

    @abstractmethod
    def _delta_L_p_length_get(self):
        pass

    @abstractmethod
    def _length_get(self):
        pass

    @abstractmethod
    def get_section_square(self) -> float:
        pass

    @abstractmethod
    def get_surface_square(self) -> float:
        pass

    length = abstractproperty(fget=_length_get)

    delta_L_p_length = abstractproperty(fget=_delta_L_p_length_get)

    def compute(self):
        self.insulation.compute()
        self._delta_L_p = (
            self.delta_L_p_prime - 10 * np.log10(self.get_surface_square() / self.get_section_square()) +
            self.insulation.R + 3 - 10 * np.log10(1 + 10**(-0.1 * self.delta_L_p_length))
        )

    def _delta_L_p_get(self):
        return self._delta_L_p

    delta_L_p = property(fget=_delta_L_p_get)


class RoundMetalChannelElement(ChannelElement):
    def __init__(self, D, length, thickness):
        ChannelElement.__init__(self)
        self.insulation = CylindricalSteelInsulation(D, thickness)
        self.D = D
        self._length = length

    def get_section_square(self):
        return np.pi * self.D ** 2 / 4

    def get_surface_square(self):
        return self._length * self.D * np.pi

    @classmethod
    def _get_delta_L_p_rel(self, D) -> np.ndarray:
        if D <= 75e-3:
            return np.array([0.1, 0.1, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3])
        elif 75e-3 < D <= 200e-3:
            return np.array([0.1, 0.1, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3])
        elif 200e-3 < D <= 400e-3:
            return np.array([0.06, 0.1, 0.1, 0.15, 0.2, 0.2, 0.2, 0.2])
        elif 400e-3 < D <= 800e-3:
            return np.array([0.03, 0.06, 0.06, 0.1, 0.15, 0.15, 0.15, 0.15])
        elif 800e-3 < D <= 1600e-3:
            return np.array([0.03, 0.03, 0.03, 0.06, 0.06, 0.06, 0.06, 0.06])
        elif 1600e-3 < D:
            return np.array([0.03, 0.03, 0.03, 0.06, 0.06, 0.06, 0.06, 0.06])

    def _delta_L_p_length_get(self):
        return self._length * self._get_delta_L_p_rel(self.D)

    def _length_get(self):
        return self._length

    length = property(fget=_length_get)
    delta_L_p_length = property(fget=_delta_L_p_length_get)


class RectangularChannelElement(ChannelElement):
    def __init__(self, width, height, length, thickness):
        ChannelElement.__init__(self)
        self.insulation = FlatSteelInsulation(thickness)
        self.width = width
        self.height = height
        self._length = length
        self.D_h = 2 * self.width * self.height / (self.width + self.height)

    def get_section_square(self):
        return self.width * self.height

    def get_surface_square(self):
        return 2 * (self.width + self.height) * self._length

    @classmethod
    def _get_delta_L_p_rel(self, D_h) -> np.ndarray:
        if D_h <= 75e-3:
            return np.array([0.6, 0.6, 0.45, 0.3, 0.3, 0.3, 0.3, 0.3])
        elif 75e-3 < D_h <= 200e-3:
            return np.array([0.6, 0.6, 0.45, 0.3, 0.3, 0.3, 0.3, 0.3])
        elif 200e-3 < D_h <= 400e-3:
            return np.array([0.6, 0.6, 0.45, 0.3, 0.2, 0.2, 0.2, 0.2])
        elif 400e-3 < D_h <= 800e-3:
            return np.array([0.6, 0.6, 0.3, 0.15, 0.15, 0.15, 0.15, 0.15])
        elif 800e-3 < D_h <= 1600e-3:
            return np.array([0.45, 0.3, 0.15, 0.1, 0.06, 0.06, 0.06, 0.06])
        elif 1600e-3 < D_h:
            return np.array([0.45, 0.3, 0.15, 0.1, 0.06, 0.06, 0.06, 0.06])

    def _delta_L_p_length_get(self):
        return self._length * self._get_delta_L_p_rel(self.D_h)

    def _length_get(self):
        return self._length

    length = property(fget=_length_get)
    delta_L_p_length = property(fget=_delta_L_p_length_get)


class RectangularSmoothTurnElement(ChannelElement):
    def __init__(self, R, width, height, angle, thickness):
        ChannelElement.__init__(self)
        self.insulation = FlatSteelInsulation(thickness)
        self.R = R
        self.width = width
        self.height = height
        self.angle = angle

    def get_section_square(self):
        return self.width * self.height

    def get_turn_width(self):
        return 2 * (self.R + 0.5 * self.width) * np.sin(self.angle / 2)

    def get_surface_square(self):
        sides_square = (self.angle * self.R + self.angle * (self.R + self.width)) * self.height
        ring_segments_square = np.pi * ((self.R + self.width)**2 - self.R**2) * self.angle / (2 * np.pi)
        return sides_square + ring_segments_square

    @classmethod
    def _get_delta_L_p_length(cls, turn_width):
        if turn_width <= 125e-3:
            return np.array([0, 0, 0, 0, 1, 2, 3, 3])
        elif 125e-3 < turn_width <= 250e-3:
            return np.array([0, 0, 0, 0, 1, 2, 3, 3])
        elif 250e-3 < turn_width <= 500e-3:
            return np.array([0, 0, 0, 1, 2, 3, 3, 3])
        elif 500e-3 < turn_width <= 1000e-3:
            return np.array([0, 0, 1, 2, 3, 3, 3, 3])
        elif 1000e-3 < turn_width <= 2000e-3:
            return np.array([0, 1, 2, 3, 3, 3, 3, 3])
        elif 2000e-3 < turn_width:
            return np.array([0, 1, 2, 3, 3, 3, 3, 3])

    def _delta_L_p_length_get(self):
        return self._get_delta_L_p_length(self.get_turn_width())

    def _length_get(self):
        return self.angle * (self.R + 0.5 * self.width)

    length = property(fget=_length_get)
    delta_L_p_length = property(fget=_delta_L_p_length_get)


class AdapterElement(ChannelElement):
    def __init__(self, square_in, square_out, length, thickness):
        ChannelElement.__init__(self)
        self.square_in = square_in
        self.square_out = square_out
        self.m = self.square_in / self.square_out
        self._length = length
        self.insulation = CylindricalSteelInsulation(
            np.sqrt(0.5 * (square_out + square_in) * 4 / np.pi),
            thickness
        )
        self._bound_section_sizes = np.array([5000, 2500, 1400, 700, 400, 200, 100, 50]) * 1e-3

    def get_surface_square(self):
        return self.get_max_section_size() * np.pi * self.length

    def get_section_square(self):
        return 0.5 * (self.square_in + self.square_out)

    def get_max_section_size(self):
        return max(np.sqrt(4 * self.square_in / np.pi), np.sqrt(4 * self.square_out / np.pi))

    def _delta_L_p_length_get(self):
        results = np.zeros([8])
        for n, bound_section_size in enumerate(self._bound_section_sizes):
            if self.get_max_section_size() < bound_section_size:
                results[n] = 10 * np.log10((self.m + 1)**2 / (4 * self.m))
            else:
                if self.m > 1:
                    results[n] = 10 * np.log10(self.m)
                else:
                    results[n] = 0
        return results

    def _length_get(self):
        return self._length

    length = property(fget=_length_get)
    delta_L_p_length = property(fget=_delta_L_p_length_get)


class Channel(Barrier):
    def __init__(self, elements: typing.List[ChannelElement]):
        self.elements = elements
        self.surface_square = None
        self.section_square = None
        self._delta_L_p = None
        self.delta_L_p_prime = None

    @classmethod
    def _get_L_p_outlet(cls, section_size):
        if section_size <= 25e-3:
            return np.array([24, 22, 19, 15, 10, 6, 2, 0])
        elif 25e-3 < section_size <= 50e-3:
            return np.array([24, 22, 19, 15, 10, 6, 2, 0])
        elif 50e-3 < section_size <= 80e-3:
            return np.array([22, 19, 15, 10, 5, 2, 0, 0])
        elif 80e-3 < section_size <= 100e-3:
            return np.array([20, 16, 11, 7, 3, 0, 0, 0])
        elif 100e-3 < section_size <= 125e-3:
            return np.array([19, 14, 10, 5, 2, 0, 0, 0])
        elif 125e-3 < section_size <= 140e-3:
            return np.array([18, 13, 8, 4, 1, 0, 0, 0])
        elif 140e-3 < section_size <= 160e-3:
            return np.array([16, 12, 8, 4, 1, 0, 0, 0])
        elif 160e-3 < section_size <= 180e-3:
            return np.array([16, 11, 7, 3, 0, 0, 0, 0])
        elif 180e-3 < section_size <= 200e-3:
            return np.array([15, 11, 6, 2, 0, 0, 0, 0])
        elif 200e-3 < section_size <= 225e-3:
            return np.array([14, 10, 6, 2, 0, 0, 0, 0])
        elif 225e-3 < section_size <= 250e-3:
            return np.array([14, 9, 5, 1, 0, 0, 0, 0])
        elif 250e-3 < section_size <= 280e-3:
            return np.array([13, 8, 4, 1, 0, 0, 0, 0])
        elif 280e-3 < section_size <= 315e-3:
            return np.array([12, 8, 3, 1, 0, 0, 0, 0])
        elif 315e-3 < section_size <= 355e-3:
            return np.array([11, 7, 3, 0, 0, 0, 0, 0])
        elif 355e-3 < section_size <= 400e-3:
            return np.array([11, 6, 2, 0, 0, 0, 0, 0])
        elif 400e-3 < section_size <= 450e-3:
            return np.array([10, 5, 2, 0, 0, 0, 0, 0])
        elif 450e-3 < section_size <= 500e-3:
            return np.array([8, 5, 1, 0, 0, 0, 0, 0])
        elif 500e-3 < section_size <= 560e-3:
            return np.array([8, 4, 1, 0, 0, 0, 0, 0])
        elif 560e-3 < section_size <= 600e-3:
            return np.array([8, 3, 1, 0, 0, 0, 0, 0])
        elif 600e-3 < section_size <= 710e-3:
            return np.array([7, 3, 1, 0, 0, 0, 0, 0])
        elif 710e-3 < section_size <= 800e-3:
            return np.array([6, 2, 0, 0, 0, 0, 0, 0])
        elif 800e-3 < section_size <= 900e-3:
            return np.array([5, 2, 0, 0, 0, 0, 0, 0])
        elif 900e-3 < section_size <= 1000e-3:
            return np.array([5, 2, 0, 0, 0, 0, 0, 0])
        elif 1000e-3 < section_size <= 1250e-3:
            return np.array([4, 1, 0, 0, 0, 0, 0, 0])
        elif 1250e-3 < section_size <= 1400e-3:
            return np.array([3, 0, 0, 0, 0, 0, 0, 0])
        elif 1400e-3 < section_size <= 1600e-3:
            return np.array([2, 0, 0, 0, 0, 0, 0, 0])
        elif 1600e-3 < section_size <= 2000e-3:
            return np.array([2, 0, 0, 0, 0, 0, 0, 0])
        elif 2000e-3 < section_size <= 2500e-3:
            return np.array([1, 0, 0, 0, 0, 0, 0, 0])
        elif 2500e-3 < section_size:
            return np.array([0, 0, 0, 0, 0, 0, 0, 0])

    def get_section_square(self):
        res = 0
        length = 0
        for element in self.elements:
            res += element.length * element.get_section_square()
            length += element.length
        return res / length

    def get_surface_square(self):
        res = 0
        for element in self.elements:
            res += element.get_surface_square()
        return res

    def get_delta_L_p_elements_sum(self):
        res = np.zeros([8])
        for element in self.elements:
            res += element.delta_L_p_length
        return res

    def compute(self):
        self.surface_square = self.get_surface_square()
        self.section_square = self.get_section_square()
        self._delta_L_p = self.get_delta_L_p_elements_sum() + self._get_L_p_outlet(
            np.sqrt(self.elements[len(self.elements) - 1].get_section_square())
        )
        self.delta_L_p_prime = np.zeros([8])
        for element in self.elements:
            element.delta_L_p_prime = self.delta_L_p_prime.copy()
            element.compute()
            self.delta_L_p_prime += element.delta_L_p_length

    def _delta_L_p_get(self):
        return self._delta_L_p

    delta_L_p = property(fget=_delta_L_p_get)


class OpenSpace(Barrier):
    def __init__(self, r, fi=1, omega=4 * np.pi):
        self.r = r
        self.fi = fi
        self.omega = omega
        self.beta_a = np.array([0, 0.7, 1.5, 3, 6, 12, 24, 48])
        self._delta_L_p = None

    def compute(self):
        if self.r <= 50:
            self._delta_L_p = 15 * np.log10(self.r) - 10 * np.log10(self.fi) + 10 * np.log10(self.omega)
        else:
            self._delta_L_p = 15 * np.log10(self.r) - 10 * np.log10(self.fi) + 10 * np.log10(self.omega) + \
                              self.beta_a * self.r / 1000

    def _delta_L_p_get(self):
        return self._delta_L_p

    delta_L_p = property(fget=_delta_L_p_get)






