import itertools
import random

import numpy as np

from gym_minigrid.minigrid import Grid, Wall, COLOR_NAMES, Door, Goal
from gym_minigrid.multiagent_minigrid import MultiAgentMiniGridEnv, Agent, Grass

from gym_minigrid.envs import MAMultiGoalEnv, FoxAndSheep
from gym_minigrid.register import register

from itertools import combinations


class FoxAndSheepBigHouse(FoxAndSheep):

    def __init__(self,
                 numGoals=25,
                 num_foxes=2,
                 num_sheep=10,
                 grid_size=40):

        self.num_foxes = num_foxes
        self.num_sheep = num_sheep
        self.foxes = []
        self.sheep = []
        super(FoxAndSheep, self).__init__(
            1, 1, 10, # ignored
            numGoals=numGoals,
            doorsOpen=True,
            init_num_agents=num_sheep + num_foxes,
            grid_size=grid_size,
        )

    def _gen_grid(self, width, height):
        self.goals_consumed = 0
        self.foxes = []
        self.sheep = []

        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.make_snake(0, self.grid.height // 2, length=50)
        self.make_snake(self.grid.width // 2, 0, length=50)
        self.make_snake(self.grid.width // 2, self.grid.height-1, length=50)
        self.make_snake(self.grid.width // 2, self.grid.height // 2, length=50)
        self.make_snake(self.grid.width-1, self.grid.height // 2, length=50)

        # Randomize the starting agent position and direction
        for i in range(self.num_foxes):
            a = Agent()
            a.color = 'red'
            self.agents.append(a)
            self.foxes.append(a)
            self.place_agent(a, (1,1), (width-1, height-1))

        for i in range(self.num_sheep):
            a = Agent()
            a.color = 'blue'
            self.agents.append(a)
            self.sheep.append(a)
            self.place_agent(a, (1,1), (width-1, height-1))

        for g in range(20):
            clumpsize = 5
            topxy = (1, 1)
            size = (self.grid.width, self.grid.height)
            for i in range(clumpsize):
                try:
                    pos = self.place_obj(Grass(), topxy, size, max_tries=1000)
                    size = (3,3)
                    print(f"Clump {g} item {i} placed at {pos}")
                    topxy = (max(1, pos[0]-1), max(1, pos[1]-1))
                    print(f"New box is {topxy}")
                except RecursionError:
                    pass

        self.mission = "Eat the grass, dont get eaten"

    def make_snake(self, start_x, start_y, length=20):
        directions = (
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
        )
        wall = Wall()
        coords = [(start_x, start_y)]
        if start_x < self.grid.width // 2:
            direction = directions[2]
        else:
            direction = directions[3]
        changeprob = 0.2

        max_iterations = 1000
        iterations = 0
        cur_x = start_x
        cur_y = start_y
        while len(coords) < length and iterations < max_iterations:
            iterations += 1
            if self._rand_float(0, 1.0) < changeprob:
                direction = self._rand_elem(directions)
            x = cur_x + direction[0]
            y = cur_y + direction[1]
            if x < 0 or x >= self.grid.width or y < 0 or y >= self.grid.height or self.wall_adjacent((x,y), (cur_x, cur_y)) or (x,y) in coords:
                continue
            # print(f"Setting {x},{y} to wall.. adjacent: {self.wall_adjacent((x,y), (cur_x, cur_y))} in coords: {(x,y) in coords}")
            self.grid.set(x, y, wall)
            cur_x = x
            cur_y = y
            coords.append((x,y))

    @staticmethod
    def modcoords(d):
        if d == 0:
            return [-1, 0, 1]
        if d > 0:
            return [0, 1]
        if d < 0:
            return [-1, 0]

    def wall_adjacent(self, pos, but_not=None):
        i, j = pos
        dir_x = pos[0] - but_not[0]
        dir_y = pos[1] - but_not[1]

        for dx, dy in itertools.product(self.modcoords(dir_x), self.modcoords(dir_y)):
            x = np.clip(i+dx, 0, self.grid.width-1)
            y = np.clip(j+dy, 0, self.grid.height-1)
            if (x,y) != but_not and type(self.grid.get(x,y)) == Wall:
                return True
        return False




register(
    id='MiniGrid-FoxAndSheepBigHouse-v0',
    entry_point='gym_minigrid.envs:FoxAndSheepBigHouse'
)


if __name__=="__main__":
    import time
    env = FoxAndSheepBigHouse()
    env.seed(np.random.randint(0, 1000))
    env.reset()
    env.render()
    time.sleep(1000)
