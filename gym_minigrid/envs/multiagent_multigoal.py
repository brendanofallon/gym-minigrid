import random

from gym_minigrid.multiagent_minigrid import Agent, MultiAgentMiniGridEnv

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym_minigrid.envs import MultiRoomEnv, MAMultiRoomEnv


class MAMultiGoalEnv(MAMultiRoomEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        numGoals=3,
        doorsOpen=True,
        init_num_agents=2,
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.numGoals = numGoals
        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize
        self.doorsOpen = doorsOpen
        self.goals_consumed = 0
        self.rooms = []

        super(MAMultiGoalEnv, self).__init__(
            minNumRooms,
            maxNumRooms,
            maxRoomSize,
            doorsOpen=doorsOpen,
            init_num_agents=init_num_agents,
        )

    def _gen_grid(self, width, height):
        self.goals_consumed = 0
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor, is_open=self.doorsOpen, is_locked=not self.doorsOpen)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        for i in range(self.init_num_agents):
            a = Agent()
            self.agents.append(a)
            self.place_agent(a, roomList[0].top, roomList[0].size)

        for g in range(self.numGoals):
            #pick a random room...
            room = random.choice(roomList)
            try:
                self.goal_pos = self.place_obj(Goal(), room.top, room.size, max_tries=1000)
            except RecursionError:
                pass

        self.mission = 'traverse the rooms to get to the goals'

    def step(self, agent, action):
        # Invalid action
        if action >= self.action_space.n:
            raise NotImplementedError("Action outside action space bounds")

        # Get the position in front of the agent
        fwd_pos = agent.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Update the agent's position/direction
        obs, reward, done, info = MultiAgentMiniGridEnv.step(self, agent, action)
        done = False
        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward:

            if fwd_cell == None or fwd_cell.can_overlap():
                self.grid.set(*agent.cur_pos, None)
                agent.cur_pos = fwd_pos
                self.grid.set(*agent.cur_pos, agent)

            if fwd_cell != None and fwd_cell.type == 'goal':
                reward = 1.0 / self.numGoals * self._reward()
                # Make the goal disappear
                self.grid.set(*fwd_pos, None)
                self.goals_consumed += 1

            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        done = done or (self.step_count == self.max_steps) or (self.goals_consumed == self.numGoals)
        return obs, reward, done, info


class MAMultiGoalOpenDoorN3S5(MAMultiGoalEnv):

    def __init__(self, minNumRooms=3, maxNumRooms=3):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=5,
            numGoals=3,
            doorsOpen=True,
        )

class MAMultiGoalOpenDoorN3S8(MAMultiGoalEnv):

    def __init__(self, minNumRooms=3, maxNumRooms=3):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=8,
            numGoals=3,
            doorsOpen=True,
        )

class MAMultiGoalOpenDoorN4S8(MAMultiGoalEnv):

    def __init__(self, minNumRooms=4, maxNumRooms=4):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=8,
            numGoals=4,
            doorsOpen=True,
        )


class MAMultiGoalOpenDoorG10N8S10(MAMultiGoalEnv):

    def __init__(self, minNumRooms=8, maxNumRooms=8):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=10,
            numGoals=10,
            doorsOpen=True,
        )

class MAMultiGoalOpenDoorG6N6S8(MAMultiGoalEnv):

    def __init__(self, minNumRooms=6, maxNumRooms=6):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=8,
            numGoals=6,
            doorsOpen=True,
        )


class MAMultiGoalOpenDoorG6N6S10(MAMultiGoalEnv):

    def __init__(self, minNumRooms=6, maxNumRooms=6):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=10,
            numGoals=6,
            doorsOpen=True,
        )

class MAMultiGoalOpenDoorG8N8S8(MAMultiGoalEnv):

    def __init__(self, minNumRooms=8, maxNumRooms=8):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=8,
            numGoals=8,
            doorsOpen=True,
        )

class MAMultiGoalOpenDoorN8S10(MAMultiGoalEnv):

    def __init__(self, minNumRooms=8, maxNumRooms=8):
        super().__init__(
            minNumRooms=minNumRooms,
            maxNumRooms=maxNumRooms,
            maxRoomSize=10,
            numGoals=6,
            doorsOpen=True,
        )

register(
    id='MiniGrid-MA-MultiGoal-G3-N3-S5-v0',
    entry_point='gym_minigrid.envs:MAMultiGoalOpenDoorN3S5'
)

register(
    id='MiniGrid-MA-MultiGoal-G3-N3-S8-v0',
    entry_point='gym_minigrid.envs:MAMultiGoalOpenDoorN3S8'
)

register(
    id='MiniGrid-MA-MultiGoal-G4-N4-S8-v0',
    entry_point='gym_minigrid.envs:MAMultiGoalOpenDoorN4S8'
)

register(
    id='MiniGrid-MA-MultiGoal-G6-N6-S8-v0',
    entry_point='gym_minigrid.envs:MAMultiGoalOpenDoorG6N6S8'
)

register(
    id='MiniGrid-MA-MultiGoal-G6-N6-S10-v0',
    entry_point='gym_minigrid.envs:MAMultiGoalOpenDoorG6N6S10'
)

register(
    id='MiniGrid-MA-MultiGoal-G8-N8-S8-v0',
    entry_point='gym_minigrid.envs:MAMultiGoalOpenDoorG8N8S8'
)

register(
    id='MiniGrid-MA-MultiGoal-G10-N8-S10-v0',
    entry_point='gym_minigrid.envs:MAMultiGoalOpenDoorG10N8S10'
)

register(
    id='MiniGrid-MA-MultiGoal-G6-N8-S10-v0',
    entry_point='gym_minigrid.envs:MAMultiGoalOpenDoorN8S10'
)
