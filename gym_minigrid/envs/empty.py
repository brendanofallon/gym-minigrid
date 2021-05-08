from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        min_width=4,
        max_width=16,
        min_height=4,
        max_height=16,

    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        if size:
            min_width = size
            max_width = size
            min_height = size
            max_height = size

        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

        width = random.randint(self.min_width, self.max_width)
        height = random.randint(self.min_height, self.max_height)

        super().__init__(
            grid_size=max(self.max_width, self.max_height),
            max_steps=4*size*size if size else 4*self.max_width*self.max_height,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, gridwidth, gridheight):
        assert self.max_width <= gridwidth, "Uh oh, self.max_width > grid width!"
        assert self.max_height <= gridheight, "Uh oh, self.max_height > grid height!"

        width = random.randint(self.min_width, self.max_width)
        height = random.randint(self.min_height, self.max_height)

        # Create an empty grid
        self.grid = Grid(gridwidth, gridheight)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent((0,0), (width, height))

        self.mission = "get to the green goal square"

class EmptyEnv5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)

class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)

class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)


class EmptyEnv4x8(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=None,
                         agent_start_pos=None,  # Otherwise agent will always start at (1, 1)
                         agent_start_dir=None,
                         min_width=4,
                         max_width=8,
                         min_height=4,
                         max_height=8,
                         **kwargs)

class EmptyEnv6x12(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=None,
                         agent_start_pos=None,  # Otherwise agent will always start at (1, 1)
                         agent_start_dir=None,
                         min_width=6,
                         max_width=12,
                         min_height=6,
                         max_height=12,
                         **kwargs)


register(
    id='MiniGrid-Empty-4x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv4x8'
)

register(
    id='MiniGrid-Empty-6x12-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x12'
)

register(
    id='MiniGrid-Empty-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyEnv5x5'
)

register(
    id='MiniGrid-Empty-Random-5x5-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-Random-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)
