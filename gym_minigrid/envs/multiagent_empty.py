from gym_minigrid.minigrid import *
from gym_minigrid.multiagent_minigrid import MultiAgentMiniGridEnv, Agent
from gym_minigrid.register import register

class MAEmptyEnv(MultiAgentMiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        width,
        height,
        num_agents=2,
    ):
        super().__init__(
            width=width,
            height=height,
            max_steps=100*width*height,
            # Set this to True for maximum speed
            see_through_walls=True,
            init_num_agents=num_agents,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agents
        for i in range(self.init_num_agents):
            a = Agent()
            self.agents.append(a)
            self.place_agent(a)

        self.mission = "get to the green goal square first"


class MAEmptyEnv6x6(MAEmptyEnv):
    def __init__(self):
        super().__init__(width=6,
                         height=6,
                         num_agents=2)

class MAEmptyEnv12x12(MAEmptyEnv):
    def __init__(self):
        super().__init__(width=12,
                         height=12,
                         num_agents=2)

register(
    id='MiniGrid-MultiAgent-Empty-6x6-N2-v0',
    entry_point='gym_minigrid.envs:MAEmptyEnv6x6'
)

register(
    id='MiniGrid-MultiAgent-Empty-12x12-N2-v0',
    entry_point='gym_minigrid.envs:MAEmptyEnv12x12'
)