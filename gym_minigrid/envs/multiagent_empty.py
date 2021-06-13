from gym_minigrid.minigrid import *
from gym_minigrid.multiagent_minigrid import MultiAgentMiniGridEnv, Agent
from gym_minigrid.register import register

class MAEmptyEnv(MultiAgentMiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size,
        num_agents=2,
    ):
        self.num_agents = num_agents

        super().__init__(
            grid_size=size,
            max_steps=100*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agents
        for i in range(self.num_agents):
            a = Agent()
            self.agents.append(a)
            self.place_agent(a)

        self.mission = "get to the green goal square first"


class MAEmptyEnv6x6(MAEmptyEnv):
    def __init__(self):
        super().__init__(size=6,
                         num_agents=2)
