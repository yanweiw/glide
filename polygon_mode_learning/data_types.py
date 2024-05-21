#! /usr/bin/env python3
# Author : Tsun-Hsuan Wang
#
# Distributed under terms of the MIT license.

import numpy as np
from typing import Optional, Sequence
from dataclasses import dataclass
from shapely import Polygon, Point

@dataclass
class Region:
    """Data to be consumed by `matplotlib.patches.Rectangle`."""
    xy: Sequence[int] = None
    width: int = None
    height: int = None

    polygon: Optional[np.ndarray] = None
    polygon_shapely: Optional[Polygon] = None

    def init_shapely(self):
        if self.polygon_shapely is None:
            self.polygon_shapely = Polygon(self.polygon)

    def in_region(self, x, y):
        # check if (x, y) is in the polygon
        self.init_shapely()
        return self.polygon_shapely.contains(Point(x, y))

    def sample(self):
        # sample a point in the polygon
        self.init_shapely()
        bounds = self.polygon_shapely.bounds
        while True:
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            if self.in_region(x, y):
                return x, y

@dataclass
class Mode:
    """Information of a mode in the task graph."""
    id: int
    name: str
    color: Sequence[float] = None
    region: Region = None
    is_initial: bool = True # by default, we can start with any mode (except for the goal mode)
    is_goal: bool = False


@dataclass
class Transition:
    """Transition across modes."""
    inp: Mode
    out: Mode
