import sys
import time

import MalmoPython
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from PIL import Image
from time import strftime
import json


def make(mission_file):
    return Minecraft(mission_file)

GOAL_REWARD = 10
DEATH_REWARD = -10
TIMEOUT_REWARD = -1

base_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <About>
    <Summary>A simple 10 second mission with a reward for reaching a location.</Summary>
  </About>
  <ModSettings>
    <MsPerTick>20</MsPerTick>
  </ModSettings>
  <ServerSection>
    <ServerInitialConditions>
            <Time>
                <StartTime>6000</StartTime>
                <AllowPassageOfTime>false</AllowPassageOfTime>
            </Time>
            <Weather>clear</Weather>
            <AllowSpawning>false</AllowSpawning>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
      <ServerQuitFromTimeUp description="" timeLimitMs="100000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Cristina</Name>
    <AgentStart>
    </AgentStart>
    <AgentHandlers>
      <ContinuousMovementCommands turnSpeedDegs="180"/>
      <ObservationFromFullStats/>
        <RewardForMissionEnd rewardForDeath="-10">
        <Reward description="out_of_time" reward="-1" />
      </RewardForMissionEnd>
    </AgentHandlers>
  </AgentSection>

</Mission>
'''


class Maze(object):
    def __init__(self):
        pass

    def create_maze_array(self):
        pass


class TMaze(Maze):
    def __init__(self, kwargs, x_bound=10, z_bound=9):
        self.x_bound = x_bound
        self.z_bound = z_bound
        # Coordinate of t junction
        self.t_coord = (self.z_bound - 1, int(self.x_bound / 2.0))
        self.start = (1, self.t_coord[1])
        self.end = (self.z_bound - 2, 1)
        self.trap = (self.z_bound - 2, self.x_bound - 2)

    def create_maze_array(self):
        ''' x is east west and z is north south
        [['1' '1' '1' '1' '1' '1' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' 's' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' 'e' '0' '0' '0' '0' '0' '0' 'l' '1']
         ['1' '1' '1' '1' '1' '1' '1' '1' '1' '1']]

        '''

        maze_array = np.chararray((self.z_bound, self.x_bound))
        maze_array[:] = '1'
        maze_array[1:self.t_coord[0], self.t_coord[1]] = '0'
        maze_array[self.z_bound - 2, 1:self.x_bound - 2] = '0'
        maze_array[self.start] = 's'
        maze_array[self.end] = 'e'
        maze_array[self.trap] = 'l'
        maze_array[self.trap[0], self.trap[1] + 1] = 'r'

        return maze_array

class TMazeLava(TMaze):
    def __init__(self, kwargs, x_bound=10, z_bound=9):
        super(self.__class__, self).__init__(kwargs, x_bound, z_bound)
        self.trap2 = (self.z_bound / 2, self.x_bound / 2)

    def create_maze_array(self):

        maze_array = super(TMazeLava, self).create_maze_array()
        maze_array[self.trap2] = 'l'
        return maze_array

class Platform(Maze):
    def __init__(self, kwargs, x_bound=11, z_bound=10):
        self.x_bound = x_bound
        self.z_bound = z_bound
        # Coordinate of t junction
        self.start = (4, 2)
        self.end = (-5, -5)

    def create_maze_array(self):
        ''' x is east west and z is north south
        [['1' '1' '1' '1' '1' '1' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' 's' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' '1' '1' '1' '1' '0' '1' '1' '1' '1']
         ['1' 'e' '0' '0' '0' '0' '0' '0' 'l' '1']
         ['1' '1' '1' '1' '1' '1' '1' '1' '1' '1']]

        '''

        maze_array = np.chararray((self.z_bound, self.x_bound))
        maze_array[:] = '0'
        maze_array[:2, :] = 'l'
        maze_array[-2:, :] = 'l'
        maze_array[:, :2] = 'l'
        maze_array[:, -2:] = 'l'

        maze_array[self.start] = 's'
        maze_array[self.end] = 'e'

        return maze_array


def create_maze(maze_def):
    maze_type = maze_def['type']
    if maze_type == 'TMaze':
        return TMaze(maze_def)
    if maze_type == 'TMazeLava':
        return TMazeLava(maze_def)
    if maze_type == 'Platform':
        return Platform(maze_def)
    else:
        raise NotImplementedError


class MissionGen(object):
    def __init__(self):
        self.floor = 227
        self.height = 3
        self.end_height = 12
        self.goal_reward = GOAL_REWARD
        self.goal_tolerance = 1.1
        self.goal_pos = None

    def base_config(self):
        my_mission = MalmoPython.MissionSpec(base_xml, False)
        my_mission.requestVideo(640, 480)
        my_mission.timeLimitInSeconds(20)
        # my_mission.allowAllChatCommands()
        # my_mission.allowAllInventoryCommands()
        my_mission.setTimeOfDay(6000, True)
        # my_mission.observeChat()
        # my_mission.observeGrid(-1, -1, -1, 1, 1, 1, 'grid')
        # my_mission.observeHotBar()
        # my_mission.o
        # my_mission.forceWorldReset()
        return my_mission

    def draw_wall(self, mission, x1, x2, z1, z2, height):
        for y in range(self.floor, self.floor + height):
            mission.drawLine(x1, y, z1, x2, y, z2, 'stone')

    def generate_mission(self, maze_array, reset=False):
        mission = self.base_config()
        if reset:
            mission.forceWorldReset()
        height, width = maze_array.shape
        self.z_bounds = (-2, height + 2)
        self.x_bounds = (-2, width + 2)
        for z in range(height):
            for x in range(width):
                elem = maze_array[z, x]
                if elem == '1':
                    self.draw_wall(mission, x, x, z, z, self.height)
                elif elem == 's':
                    mission.startAt(x, self.floor, z)
                elif elem == 'l':
                    mission.drawBlock(x, self.floor - 1, z, 'lava')
                elif elem == 'r':
                    mission.drawLine(x, self.floor, z, x, self.floor + self.end_height, z, 'redstone_block')
                elif elem == 'e':
                    self.goal_pos = (x + 0.5, self.floor, z + 0.5)
                    mission.drawLine(x, self.floor, z, x, self.floor + self.end_height, z, 'lapis_block')
                    mission.rewardForReachingPosition(x + 0.5, self.floor, z + 0.5, self.goal_reward,
                                                      self.goal_tolerance)
                    mission.endAt(x + 0.5, self.floor, z + 0.5, self.goal_tolerance)
                    mission.observeDistance(x + 0.5, self.floor, z + 0.5, 'Goal')

        return mission

    def test_maze(self):
        pass
        # mission = self.create_maze() #MalmoPython.MissionSpec(missionXML, True) #
        # agent = MalmoPython.AgentHost()
        # mission_record_spec =  MalmoPython.MissionRecordSpec()
        # agent.startMission(mission, mission_record_spec)


class Minecraft(object):
    def __init__(self, maze_def, reset, video_dim=(32, 32), num_parallel=1, time_limit=30,
                 discrete_actions=False, vision_observation=True, depth=False, num_frames=1, grayscale=True):
        self.video_width, self.video_height = video_dim
        self.image_width, self.image_height = video_dim
        self.discrete_actions = discrete_actions
        self.vision_observation = vision_observation
        self.depth = depth
        self.num_parallel = num_parallel

        maze = create_maze(maze_def)
        self.mission_gen = MissionGen()
        self.mission = self.mission_gen.generate_mission(maze.create_maze_array(), reset=reset)
        self.XGoalPos, self.YGoalPos = self.mission_gen.goal_pos[0], self.mission_gen.goal_pos[2]

        # with open(mission_file, 'r') as f:
        #     print("Loading mission from %s" % mission_file)
        #     mission_xml = f.read()
        #     self.mission = MalmoPython.MissionSpec(mission_xml, True)
        self.mission.requestVideo(self.video_height, self.video_width)
        self.mission.observeRecentCommands()
        self.mission.allowAllContinuousMovementCommands()
        # self.mission.timeLimitInSeconds(time_limit)

        if self.num_parallel > 1:
            self.client_pool = MalmoPython.ClientPool()
            for i in range(num_parallel):
                port = 10000 + i
                self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", port))

        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.KEEP_ALL_OBSERVATIONS)
        # self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.KEEP_ALL_FRAMES)
        # self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

        self.mission_record_spec = MalmoPython.MissionRecordSpec()

        if discrete_actions:
            self._action_set = {0: "move 1", 1: "turn 1", 2: "turn -1"}
            self.action_space = Discrete(n=len(self._action_set))
        else:
            # self._action_set = ["move", "turn", "pitch"]
            # self.action_space = Box(np.array([0, -.5, -.5]), np.array([1, .5, .5]))
            self._action_set = ["move", "turn", "jump"]
            self.action_space = Box(np.array([-1, -.5, -1]), np.array([1, 0.5, 1]))

        self.num_frames = num_frames
        self.grayscale = grayscale
        if self.grayscale:
            self.num_frame_channels = 1
            high = 1
        else:
            self.num_frame_channels = 3
            high = 255

        # Obs keys and bounds
        x_bounds = self.mission_gen.x_bounds
        z_bounds = self.mission_gen.z_bounds
        self.max_dist = np.linalg.norm((x_bounds[-1], z_bounds[-1]))
        self.minDistanceFromGoal = None
        if self.vision_observation:
            self.observation_space = Box(low=0,
                                         high=high,
                                         shape=(self.num_frames * self.num_frame_channels, self.image_height,
                                                self.image_width))
        else:

            self.obs_keys = [(u'XPos', x_bounds),
                             (u'ZPos', z_bounds),
                             (u'yaw', (0, 360)),
                             (u'XGoalPos', x_bounds),
                             (u'YGoalPos', z_bounds),
                             (u'DistanceTravelled', (0, 30)),
                             (u'distanceFromGoal', (0, self.max_dist))]
            l_bounds = [key[1][0] for key in self.obs_keys]
            u_bounds = [key[1][1] for key in self.obs_keys]
            self.observation_space = Box(np.array(l_bounds), np.array(u_bounds))
        # self._horizon = env.spec.timestep_limit
        self.last_obs = None
        self.cum_reward = 0
        self.distance_travelled = 0
        self.terminal = False
        self.jump = 0


    def _get_obs(self, world_state):
        total_reward = -.1
        for reward in world_state.rewards:
            total_reward += reward.getValue()

        if self.vision_observation:
            # if len(world_state.video_frames) < 1:
            #    print("No frames, setting obs and reward to 0")
            #    return np.zeros(self.observation_space.shape), 0

            frame_concat = []
            for i in range(self.num_frames, 0, -1):
                frame = world_state.video_frames[-i]
                image = Image.frombytes('RGB', (frame.width, frame.height), str(frame.pixels))
                if self.grayscale:
                    image = image.convert('L')
                    image_np = np.array(image).reshape(1, self.image_height, self.image_width)
                else:
                    # gray_scale = image.convert('LA')
                    image_np = np.array(image.resize((self.image_width, self.image_height))) \
                        .transpose(2, 0, 1).reshape(3, self.image_height, self.image_width)
                frame_concat.append(image_np)


            reward_distance_from_goal = True
            if reward_distance_from_goal:
                distanceFromGoal = json.loads(world_state.observations[-1].text).get('distanceFromGoal')
                if self.minDistanceFromGoal == None:
                    self.minDistanceFromGoal = distanceFromGoal
                if distanceFromGoal < self.minDistanceFromGoal:
                    total_reward += (self.minDistanceFromGoal - distanceFromGoal)
                    self.minDistanceFromGoal = distanceFromGoal

            return (np.concatenate(frame_concat, axis=0), total_reward)
        else:
            # if len(world_state.video_frames) < 1 or len(world_state.observations) < 1:
            #    print("No frames, setting obs to last state")
            #    return self.last_obs, total_reward
            frame = world_state.video_frames[-1]

            msg = json.loads(world_state.observations[-1].text)
            state = []
            for index, key in enumerate(self.obs_keys):
                if key[0] == 'yaw':
                    obs = getattr(frame, 'yaw') % 360  # Yaw is cumulative so need to take mod
                elif key[0] == u'DistanceTravelled':
                    obs = msg.get(key[0]) - self.lastDistanceTravelled
                    self.distance_travelled = obs
                    self.lastDistanceTravelled = msg.get(key[0])
                elif 'GoalPos' in key[0]:
                    obs = getattr(self, key[0])
                else:
                    obs = msg.get(key[0])
                if obs is None:
                    print("Obs was None", key[0])
                state.append(float(obs))
            # print state
            # self.last_obs = state
            return state, total_reward

    def _send_command(self, action_str):
        try:
            self.agent_host.sendCommand(action_str)
        except RuntimeError as e:
            print("Failed to send command: %s" % e)

    def _perform_action(self, action):
        if self.discrete_actions:
            action = self._action_set[action]
            self._send_command(action)
        else:
            # print(zip(self._action_set, action.tolist()))
            for action_name, val in zip(self._action_set, action.tolist()):
                if action_name == 'jump':
                    if val >= 0:
                        self.jump = 1
                        self._send_command('jump 1')
                else:
                    self._send_command('%s %f' % (action_name, val))

    # @property
    # def observation_space(self):
    #     return self._observation_space
    #
    #
    # #@property
    # def action_space(self):
    #     return self.action_space


    # @property
    def horizon(self):
        raise NotImplementedError
        # return self._horizon

    def render(self):
        raise NotImplementedError

    def wait_for_initial_state(self):
        # wait for a valid observation
        world_state = self.agent_host.peekWorldState()
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        # print("Waiting for initial state")
        while world_state.is_mission_running and (len(world_state.video_frames) < self.num_frames
                                                  or len(world_state.observations) < 1
                                                  or all(e.text == '{}' for e in
                                                         world_state.observations)):  # world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = self.agent_host.peekWorldState()
            time.sleep(0.1)
            sys.stdout.write('.')

        print("Waiting for initial state done")
        # world_state = self.agent_host.getWorldState()
        self.lastDistanceTravelled = json.loads(world_state.observations[-1].text).get(u'DistanceTravelled', 0)

        return world_state

    def reset(self):
        self.total_steps = 0
        self.cum_reward = 0
        self.terminal = False
        self.minDistanceFromGoal = None
        # Try starting mission
        max_retries = 3
        for retry in range(max_retries):
            try:
                if self.num_parallel > 1:
                    self.agent_host.startMission(self.mission, self.client_pool,
                                                 self.mission_record_spec, 0, strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    self.agent_host.startMission(self.mission, self.mission_record_spec)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2.5)
        time.sleep(0.5)
        print("Mission has started")
        # Wait till mission has begun
        world_state = self.agent_host.getWorldState()

        while not world_state.is_mission_running or not world_state.has_mission_begun:
            sys.stdout.write(".")
            time.sleep(0.1)
            world_state = self.agent_host.peekWorldState()
            for error in world_state.errors:
                print("Error:%s" % error.text)

        world_state = self.wait_for_initial_state()
        # assert world_state.has_mission_begun and world_state.is_mission_running \
        #       and len(world_state.video_frames) > 0
        return self._get_obs(world_state)[0]

    def step(self, action):
        self.total_steps += 1

        step_info = {}
        # world_state = self.agent_host.getWorldState()

        # Flag is set when mission is ended.
        # Deal with issue where sample that ends mission won't be sampled from replay memory
        # if self.terminal:
        #    return self.last_obs, 0, True, step_info
        if self.jump == 1:
            self._send_command(('jump 0'))
            self.jump = 0
        self._perform_action(action)

        time.sleep(.02)
        world_state = self.agent_host.getWorldState()

        while world_state.is_mission_running and \
                (world_state.number_of_video_frames_since_last_state < self.num_frames or
                         len(world_state.observations) < 1):  # and \
            # world_state.number_of_observations_since_last_state < 1:
            # print("Waiting for frames...")
            time.sleep(0.01)
            world_state = self.agent_host.peekWorldState()
        if not world_state.is_mission_running:
            ob = self.last_obs
            total_reward = 0
            for reward in world_state.rewards:
                total_reward += reward.getValue()
            if total_reward == 0 and self.cum_reward == 0:
                total_reward = -1
            self.cum_reward += total_reward
            print("Mission is over with total reward and total steps", self.cum_reward, self.total_steps)
        else:

            assert len(world_state.observations) > 0 and len(world_state.video_frames) > 0
            ob, total_reward = self._get_obs(world_state)
            self.last_obs = ob

        # if len(world_state.mission_control_messages) > 0:
        #     for x in world_state.mission_control_messages:
        #         print x.text

        # total_reward += 0.1 * self.distance_travelled
        # print self.distance_travelled


        self.cum_reward += total_reward

        return ob, total_reward, not world_state.is_mission_running, step_info


if __name__ == '__main__':
    maze = create_maze({'type':sys.argv[1]})
    mission = MissionGen().generate_mission(maze.create_maze_array(), reset=True)
    agent = MalmoPython.AgentHost()
    mission_record_spec = MalmoPython.MissionRecordSpec()
    agent.startMission(mission, mission_record_spec)
    print maze.create_maze_array()
