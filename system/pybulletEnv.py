from cmath import sqrt
from random import random
import pybullet as p
import os
import pybullet_data
import numpy as np
import math

# from system.helper import compute_angle

# from system.controller.navigationPhase import pick_intermediate_goal_vector, find_new_goal_vector


class PybulletEnvironment:
    """This class deals with everything pybullet or environment (obstacles) related"""
    def __init__(self, visualize, env_model, dt, cuePos, pod=None):
        self.visualize = visualize  # to open JAVA application
        self.env_model = env_model  # string specifying env_model
        self.pod = pod  # if Phase Offset detectors are used

        # for random braitenberg
        self.t = 0
        self.velocityL = 5.5
        self.velocityR = 5.5

        if self.visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        base_position = [4, 4, 0.02]  # [0, 0.05, 0.02] ensures that it actually starts at origin
        orientation = p.getQuaternionFromEuler([0,0,random() * np.pi * 2])#p.getQuaternionFromEuler([0, 0, np.pi/2])  # faces North
        # (w, x, y, z) = (cos(a/2), sin(a/2) * nx, sin(a/2)* ny, sin(a/2) * nz)
        # Where a is the angle of rotation and {nx,ny,nz} is the axis of rotation
        print(orientation)
        arena_size = 15  # circular arena size with radius r
        goal_location = None
        max_speed = 5.5  # determines speed at which agent travels: max_speed = 5.5 -> actual speed of ~0.5 m/s
        self.carID = None
        # load environment map

        if env_model == "T-shape":
            p.loadURDF("p3dx/plane/plane.urdf")
            self.cueId = p.loadURDF("p3dx/cue/visual_cue.urdf", basePosition=cuePos)
            self.carID = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=base_position, baseOrientation=orientation)
            # self.carID = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=[-1.5, -2, 0.02])#[0, -2, 0.02])
        elif env_model == "large_box":
            # TODO implement
            p.loadURDF("p3dx/plane/plane_large.urdf")
            self.cueId = p.loadURDF("p3dx/cue/visual_cue.urdf", basePosition=cuePos)
            self.carID = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=base_position, baseOrientation=orientation)
            
        elif env_model == "box":
            p.loadURDF("p3dx/plane/plane_box.urdf")
        elif env_model == "linear_sunburst":
            doors_option = "plane_doors"  # "plane" for default, "plane_doors", "plane_doors_individual"
            p.loadURDF("environment/linear_sunburst_map/" + doors_option + ".urdf")
            base_position = [5.5, 0.55, 0.02]
            arena_size = 15
            goal_location = np.array([1.5, 10])
            max_speed = 6
        else:
            urdfRootPath = pybullet_data.getDataPath()
            p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"))

            self.carID = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=base_position, baseOrientation=orientation)

        p.setGravity(0, 0, -9.81)

        self.dt = dt
        p.setTimeStep(self.dt)

        self.xy_coordinates = []  # keeps track of agent's coordinates at each time step
        self.orientation_angle = []  # keeps track of agent's orientation at each time step
        self.xy_speeds = []  # keeps track of agent's speed (vector) at each time step
        self.speeds = []  # keeps track of agent's speed (value) at each time step
        self.save_position_and_speed()  # save initial configuration

        # Set goal location to preset location or current position if none was specified
        self.goal_location = goal_location if goal_location is not None else self.xy_coordinates[0]

        self.max_speed = max_speed
        self.arena_size = arena_size
        self.goal = np.array([0, 0])  # used for navigation (eg. sub goals)

        # self.goal_vector_original = np.array([1, 1])  # egocentric goal vector after last recalculation
        # self.goal_vector = np.array([0, 0])  # egocentric goal vector after last update

        self.goal_idx = 0  # pc_idx of goal

        self.turning = False  # agent state, used for controller

        self.num_ray_dir = 16  # number of direction to check for obstacles for
        self.num_travel_dir = 4  # valid traveling directions, 4 -> [E, N, W, S]
        self.directions = np.empty(self.num_ray_dir, dtype=bool)  # array keeping track which directions are blocked
        self.topology_based = False  # agent state, used for controller

        self.euler_angle = p.getEulerFromQuaternion(p.getLinkState(self.carID, 0)[1])
        self.euler_angle_before = 0
        # addition for BC-Model helper that returns coordinates of encountered boundary segments wth getRays()
        self.raysThatHit = []
        self.cuePos = cuePos

        p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-80, cameraTargetPosition=[0, 0, 1.0])

        #TODO does it show rays now?
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1) 

        # # Make Robot go in a circle
        # p.setJointMotorControlArray(bodyUniqueId=self.carID,
        #  jointIndices=[4, 6],
        #  controlMode=p.VELOCITY_CONTROL,
        #  targetVelocities=[0.168*20, 0.2*20],
        #  forces=[10, 10])

    def compute_movement(self, gc_network, pc_network, cognitive_map=None, exploration_phase=True):
        """Compute and set motor gains of agents. Simulate the movement with py-bullet"""
        # gains = self.avoid_obstacles(gc_network, pc_network, cognitive_map, exploration_phase)

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1, p.COV_ENABLE_GUI, 0)
        self.euler_angle_before = self.euler_angle
        self.braitenberg()
        #  self.change_speed(gains)
        p.stepSimulation()

        self.save_position_and_speed()
        # if self.visualize:
        #     time.sleep(self.dt/5)

        # return change in orientation
        # before
        e_b = self.euler_angle_before[2]
        # after
        e_a = self.euler_angle[2]
        # fix transitions pi <=> -pi
        # in top left quadrant
        e_b_topleft = e_b < np.pi and e_b > np.pi / 2
        e_a_topleft = e_a < np.pi and e_a > np.pi / 2
        # in bottom left quadrant
        e_b_bottomleft = e_b < -np.pi / 2 and e_b > -np.pi
        e_a_bottomleft = e_a < -np.pi / 2 and e_a > -np.pi
        if e_a_topleft and e_b_bottomleft:
            # transition in negative direction
            return -(abs(e_a - np.pi) + abs(e_b + np.pi))
        elif e_a_bottomleft and e_b_topleft:
            # transition in positive direction
            return abs(e_a + np.pi) + abs(e_b - np.pi)
        else:
            # no transition, just the difference
            return e_a - e_b


    def change_speed(self, gains):
        p.setJointMotorControlArray(bodyUniqueId=self.carID,
                                    jointIndices=[4, 6],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=gains,
                                    forces=[10, 10])

    def save_position_and_speed(self):
        [position, angle] = p.getBasePositionAndOrientation(self.carID)
        angle = p.getEulerFromQuaternion(angle)
        self.xy_coordinates.append(np.array([position[0], position[1]]))
        self.orientation_angle.append(angle[2])

        [linear_v, _] = p.getBaseVelocity(self.carID)
        self.xy_speeds.append([linear_v[0], linear_v[1]])
        self.speeds.append(np.linalg.norm([linear_v[0], linear_v[1]]))

    def end_simulation(self):
        p.disconnect()

    def ray_detection(self, angles):
        """Check for obstacles in defined directions."""

        p.removeAllUserDebugItems()

        # NOTE ray_len = 1 ?
        ray_len = 2  # max_ray length to check for

        ray_from = []
        ray_to = []

        ray_from_point = np.array(p.getLinkState(self.carID, 0)[0])
        ray_from_point[2] = ray_from_point[2] + 0.02

        for angle in angles:
            ray_from.append(ray_from_point)
            ray_to.append(np.array([
                np.cos(angle) * ray_len + ray_from_point[0],
                np.sin(angle) * ray_len + ray_from_point[1],
                ray_from_point[2]
            ]))

        ray_dist = np.empty_like(angles)
        results = p.rayTestBatch(ray_from, ray_to, numThreads=0)
        for idx, result in enumerate(results):
            hit_object_uid = result[0]

            dist = ray_len
            if hit_object_uid != -1:
                hit_position = result[3]
                dist = np.linalg.norm(hit_position - ray_from_point)
                ray_dist[idx] = dist

            ray_dist[idx] = dist

            # if dist < 1:
            #     p.addUserDebugLine(ray_from[idx], ray_to[idx], (1, 1, 1))
            
        return ray_dist

    def braitenberg(self):
        detect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if True: #self.model == "maze" or "curved":
            braitenbergL = np.array(
                [-0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0,  0.0,   0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.6, -1.55, -1.5, -1.45, -1.4, -1.35, -1.3, -1.25, -1.2, -1.15, -1.1, -1.05, -1.0]
                )
            braitenbergR = np.array(
                [-1.0, -1.05, -1.1, -1.15, -1.2, -1.25, -1.3, -1.35, -1.4, -1.45, -1.5, -1.55, -1.6,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0,
                 -0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5, -0.55, -0.6, -0.65, -0.7, -0.75, -0.8])
            noDetectionDist = 1.75
            velocity_0 = 5.5
            maxDetectionDist = 0.25
        else:
            braitenbergL = np.array(
                [-0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0,-1.6, -1.55, -1.5, -1.45, -1.4, -1.35, -1.3, -1.25, -1.2, -1.15, -1.1, -1.05, -1.0]
                )
            braitenbergR = np.array(
                [-1.0, -1.05, -1.1, -1.15, -1.2, -1.25, -1.3, -1.35, -1.4, -1.45, -1.5, -1.55, -1.6,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0,
                 -0.2, -0.25, -0.3, -0.35, -0.4, -0.45, -0.5, -0.55, -0.6, -0.65, -0.7, -0.75, -0.8])
            noDetectionDist = 1.0
            velocity_0 = 2.0
            maxDetectionDist = 0.2

        rayDist = self.bc_ray_detection()
        for i in range(len(rayDist)):
            if 0 < rayDist[i] < noDetectionDist:
                # something is detected
                if rayDist[i] < maxDetectionDist:
                    rayDist[i] = maxDetectionDist
                # dangerous level, the higher, the closer
                detect[i] = 1.0 - 1.0 * ((rayDist[i] - maxDetectionDist) * 1.0 / (noDetectionDist - maxDetectionDist))
            else:
                # nothing is detected
                detect[i] = 0

        vLeft = velocity_0
        vRight = velocity_0

        if np.sum(np.multiply(braitenbergL[i], detect[i])) == 0 and np.sum(np.multiply(braitenbergR[i], detect[i])) == 0:
            # with a 5 % chance change velocity
            if random() > 0.5 and self.t <= 0:
                self.t = 30 
                self.velocityL, self.velocityR = velocity_0, velocity_0
                temp = random() + 1
                if random() < 0.5:
                    self.velocityL = velocity_0 + temp
                else:
                    self.velocityR = velocity_0 + temp
            vLeft = self.velocityL
            vRight = self.velocityR
        else:
            vLeft = velocity_0
            vRight = velocity_0

        self.t -= 1

        # print(detect)
        for i in range(len(rayDist)):
            vLeft = vLeft + braitenbergL[i] * detect[i] * 1
            vRight = vRight + braitenbergR[i] * detect[i] * 1

        if vLeft < 0.5 and vRight < 0.5:
            vRight = 5
            vLeft = 0

        '''
        minVelocity = 0.5
        if abs(vLeft) < minVelocity and abs(vRight) < minVelocity:
            vLeft = minVelocity
            vRight = minVelocity
        print("V Left:", vLeft, "V Right", vRight)
        '''
        p.setJointMotorControlArray(bodyUniqueId=self.carID,
                                    jointIndices=[4, 6],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[vLeft, vRight],
                                    forces=[10, 10])

    def bc_ray_detection(self):
# the index of the ray is from the front, counter-clock-wise direction #
        # detect range rayLen = 1 #
        p.removeAllUserDebugItems()
        rayReturn = []
        rayFrom = []
        rayTo = []
        rayIds = []
        numRays = 51
        rayLen = 5 # set ray length
        rayHitColor = [1, 0, 0]
        rayMissColor = [1, 1, 1]

        replaceLines = True

        for i in range(numRays):
            # rayFromPoint = p.getBasePositionAndOrientation(self.carId)[0]
            rayFromPoint = p.getLinkState(self.carID, 0)[0]
            rayReference = p.getLinkState(self.carID, 0)[1]
            euler_angle = p.getEulerFromQuaternion(rayReference)  # in degree
            # print("Euler Angle: ", rayFromPoint)
            rayFromPoint = list(rayFromPoint)
            rayFromPoint[2] = rayFromPoint[2] + 0.02
            rayFrom.append(rayFromPoint)
            rayTo.append([
                rayLen * math.cos(
                    2.0 * math.pi * float(i) / numRays + 360.0 / numRays / 2 * math.pi / 180 + euler_angle[2]) +
                rayFromPoint[0],
                rayLen * math.sin(
                    2.0 * math.pi * float(i) / numRays + 360.0 / numRays / 2 * math.pi / 180 + euler_angle[2]) +
                rayFromPoint[1],
                rayFromPoint[2]
            ])

        results = p.rayTestBatch(rayFrom, rayTo, numThreads=0)
        for i in range(numRays):
            hitObjectUid = results[i][0]

            if (hitObjectUid < 0):
                hitPosition = [0, 0, 0]
                # p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
                if(i ==0):
                    p.addUserDebugLine(rayFrom[i], rayTo[i], (0,0,0))
                p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor)
                rayReturn.append(-1)
            else:
                hitPosition = results[i][3]
                # p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])
                p.addUserDebugLine(rayFrom[i], rayTo[i], rayHitColor)
                rayReturn.append(
                    math.sqrt((hitPosition[0] - rayFrom[i][0]) ** 2 + (hitPosition[1] - rayFrom[i][1]) ** 2))

        self.euler_angle = euler_angle
        # print("euler_angle: ", euler_angle[2] * 180 / np.pi)

        ### BC-Model
        # returns the distance to the walls hit starting from 0 to 2pi counter clockwise so each of the 51 entries is
        # the length for one radial separation bin
        self.raysThatHit = rayReturn
        ###
        return rayReturn

    def getRays(self):
        return self.raysThatHit
