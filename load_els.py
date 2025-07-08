import numpy as np

class LoadElement:
    def __init__(self,
                 m_axles: np.ndarray,
                 m_carriage: float,
                 di: np.ndarray):

        if len(di) != len(m_axles) - 1:
            raise ValueError("Axle spacing vector not consistent with number of axles")

        self.n_axles = len(m_axles)
        self.m_carriage = m_carriage
        self.di = np.concatenate(([0], di))
        self.location = np.empty(len(m_axles))
        self.m_per_axle = np.ones(self.n_axles) * self.m_carriage/self.n_axles + m_axles

class LoadSystem:
    def __init__(self, elements: list, dij: np.ndarray):
        if len(dij) != len(elements) - 1:
            raise ValueError("Interspacing vector not consistent with number of vehicles")
        if not all(isinstance(e, LoadElement) for e in elements):
            raise TypeError("All elements must be instances of LoadElement")

        self.elements = elements
        self.interspacing = dij
        self.get_initial_location()

    def total_mass(self):
        """Returns the total mass of all load elements (axles + carriage)."""
        total = 0
        for elem in self.elements:
            total += np.sum(elem.m_per_axle)
        return total

    def get_initial_location(self):
        location = 0
        for i, elem in enumerate(self.elements):
            elem.location = location - elem.di
            location += elem.location[-1]
            if i < len(self.interspacing):
                location -= self.interspacing[i]
