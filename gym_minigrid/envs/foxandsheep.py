import random

from gym_minigrid.minigrid import Grid, Wall, COLOR_NAMES, Door, Goal
from gym_minigrid.multiagent_minigrid import MultiAgentMiniGridEnv, Agent, Grass

from gym_minigrid.envs import MAMultiGoalEnv
from gym_minigrid.register import register


class FoxAndSheep(MAMultiGoalEnv):


    def __init__(self,
                 minNumRooms,
                 maxNumRooms,
                 maxRoomSize=10,
                 numGoals=3,
                 num_foxes=1,
                 num_sheep=1,
                 grid_size=25,
                 grass_growth_rate=0.0):

        self.num_foxes = num_foxes
        self.num_sheep = num_sheep
        self.grass_growth_rate = grass_growth_rate
        self.foxes = []
        self.sheep = []
        super(FoxAndSheep, self).__init__(
            minNumRooms,
            maxNumRooms,
            maxRoomSize,
            numGoals=numGoals,
            doorsOpen=True,
            init_num_agents=num_sheep + num_foxes,
            grid_size=grid_size,
        )

    def _gen_grid(self, width, height):
        self.goals_consumed = 0
        self.foxes = []
        self.sheep = []
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
        for i in range(self.num_foxes):
            a = Agent()
            a.color = 'red'
            self.agents.append(a)
            self.foxes.append(a)
            self.place_agent(a, roomList[0].top, roomList[0].size)

        for i in range(self.num_sheep):
            a = Agent()
            a.color = 'blue'
            self.agents.append(a)
            self.sheep.append(a)
            self.place_agent(a, roomList[0].top, roomList[0].size)

        # for g in range(self.numGoals):
        #     #pick a random room...
        #     room = random.choice(roomList)
        #     try:
        #         self.place_obj(Goal(), room.top, room.size, max_tries=1000)
        #     except RecursionError:
        #         pass

        for g in range(2):
            clumpsize = 2
            self._place_grass(clumpsize)

        self.mission = 'traverse the rooms to get to the goals'

    def _place_grass(self, clumpsize=1):
        ri = self._rand_int(0, len(self.rooms))
        topxy = self.rooms[ri].top
        size = self.rooms[ri].size
        for i in range(clumpsize):
            try:
                pos = self.place_obj(Grass(init_nrg=self._rand_int(2,8)), topxy, size, max_tries=1000)
            except RecursionError:
                pass


    def step(self, agent, action):
        # Invalid action
        if action >= self.action_space.n:
            raise NotImplementedError("Action outside action space bounds")

        is_sheep = agent in self.sheep
        is_fox = agent in self.foxes
        assert is_sheep or is_fox, "Agent must be a fox or a sheep!"

        # Get the position in front of the agent
        fwd_pos = agent.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Update the agent's position/direction
        obs, reward, done, info = MultiAgentMiniGridEnv.step(self, agent, action)
        reward = 0.0
        done = False

        if is_sheep and action == self.actions.pickup:
            # Pickup means 'eat', as in eat the grass
            if type(fwd_cell) == Grass:
                reward = 0.1
                fwd_cell.nrg -= 1
                if fwd_cell.nrg <= 0:
                    self.grid.set(*fwd_pos, None)

        if action == self.actions.forward:
            if is_fox and fwd_cell in self.sheep:
                whichsheep = self.sheep[self.sheep.index(fwd_cell)]
                reward = 1.0
                done = True
                whichsheep.reward_mod = -1.0

            if fwd_cell is None or fwd_cell.can_overlap():
                self.grid.set(*agent.cur_pos, None)
                agent.cur_pos = fwd_pos
                self.grid.set(*agent.cur_pos, agent)

            if is_sheep and fwd_cell != None:
                if fwd_cell.type == 'goal':
                    reward = 1.0 / self.numGoals * self._reward()
                    # Make the goal disappear
                    self.grid.set(*fwd_pos, None)
                    self.goals_consumed += 1

                if fwd_cell in self.foxes:
                    reward = -1.0
                    done = True

            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        if self._rand_float(0, 1.0) < self.grass_growth_rate:
            self._place_grass(1)

        reward += agent.reward_mod
        agent.reward_mod = 0
        done = done or (self.step_count == self.max_steps) or (self.goals_consumed == self.numGoals)
        return obs, reward, done, info


class FoxAndSheepEmpty6x6(FoxAndSheep):

    def __init__(self):
        super().__init__(
            minNumRooms=1,
            maxNumRooms=1,
            maxRoomSize=8,
            numGoals=3,
            num_sheep=1,
            num_foxes=1,
            grass_growth_rate=0.01,
        )


register(
    id='MiniGrid-FoxAndSheep-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:FoxAndSheepEmpty6x6'
)

class FoxAndSheepJustSheep(FoxAndSheep):

    def __init__(self):
        super().__init__(
            minNumRooms=1,
            maxNumRooms=1,
            maxRoomSize=12,
            numGoals=4,
            num_sheep=1,
            num_foxes=0,
            grass_growth_rate=0.02,
        )

register(
    id='MiniGrid-FoxAndSheep-JustSheep-v0',
    entry_point='gym_minigrid.envs:FoxAndSheepJustSheep'
)

class FoxAndSheepEmpty12x12(FoxAndSheep):

    def __init__(self):
        super().__init__(
            minNumRooms=1,
            maxNumRooms=1,
            maxRoomSize=12,
            numGoals=4,
            num_sheep=1,
            num_foxes=1,
            grass_growth_rate=0.02,
        )

register(
    id='MiniGrid-FoxAndSheep-Empty-12x12-v0',
    entry_point='gym_minigrid.envs:FoxAndSheepEmpty12x12'
)


class FoxAndSheep3Room(FoxAndSheep):

    def __init__(self):
        super().__init__(
            minNumRooms=3,
            maxNumRooms=3,
            maxRoomSize=12,
            numGoals=6,
            num_sheep=1,
            num_foxes=1,
            grass_growth_rate=0.03,
        )

register(
    id='MiniGrid-FoxAndSheep-3room-v0',
    entry_point='gym_minigrid.envs:FoxAndSheep3Room'
)