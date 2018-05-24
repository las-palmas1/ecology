from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import typing


class ChannelElement(metaclass=ABCMeta):

    @abstractmethod
    def _delta_L_p_get(self):
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

    delta_L_p = abstractproperty(fget=_delta_L_p_get)


class RoundMetalChannelElement(ChannelElement):
    def __init__(self, D, length):
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

    def _delta_L_p_get(self):
        return self._length * self._get_delta_L_p_rel(self.D)

    def _length_get(self):
        return self._length

    length = property(fget=_length_get)
    delta_L_p = property(fget=_delta_L_p_get)


class RectangularChannelElement(ChannelElement):
    def __init__(self, width, height, length):
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

    def _delta_L_p_get(self):
        return self._length * self._get_delta_L_p_rel(self.D_h)

    def _length_get(self):
        return self._length

    length = property(fget=_length_get)
    delta_L_p = property(fget=_delta_L_p_get)


class RectangularSmoothTurnElement(ChannelElement):
    def __init__(self, R, width, height, angle):
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
    def _get_delta_L_p(cls, turn_width):
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

    def _delta_L_p_get(self):
        return self._get_delta_L_p(self.get_turn_width())

    def _length_get(self):
        return self.angle * (self.R + 0.5 * self.width)

    length = property(fget=_length_get)
    delta_L_p = property(fget=_delta_L_p_get)


class AdapterElement(ChannelElement):
    def __init__(self, square_in, square_out, length):
        self.square_in = square_in
        self.square_out = square_out
        self.m = self.square_in / self.square_out
        self._length = length
        self._bound_section_sizes = np.array([5000, 2500, 1400, 700, 400, 200, 100, 50]) * 1e-3

    def get_surface_square(self):
        return self.get_max_section_size() * np.pi * self.length

    def get_section_square(self):
        return 0.5 * (self.square_in + self.square_out)

    def get_max_section_size(self):
        return max(np.sqrt(4 * self.square_in / np.pi), np.sqrt(4 * self.square_out / np.pi))

    def _delta_L_p_get(self):
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
    delta_L_p = property(fget=_delta_L_p_get)


class Channel:
    def __init__(self, elements: typing.List[ChannelElement], R: np.ndarray=np.zeros([8])):
        self.elements = elements
        self.R = R

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

    def get_L_p_sum(self):
        res = np.zeros([8])
        for element in self.elements:
            res += element.delta_L_p

    def compute(self):
        self.surface_square = self.get_surface_square()
        self.section_square = self.get_section_square()
        self.delta_L_p_elements_sum = self.get_L_p_sum() + self._get_L_p_outlet(
            np.sqrt(self.elements[len(self.elements) - 1].get_section_square())
        )
        self.delta_L_p = (
                -10 * np.log10(self.surface_square / self.section_square) + self.R + 3 -
                10 * np.log10(1 + 10**(-0.1 * self.delta_L_p_elements_sum))
        )





