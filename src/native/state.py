import math

import numpy as np
import numpy.typing as npt

from .solver import ArmSolver


class ArmState:
    def __init__(self, joints: npt.NDArray[np.float64], solver: ArmSolver):
        self.joints = joints
        self.solver = solver
        self.pose = solver.forward(self.joints[:6])

    def set_with_ef_delta(self, ef_delta: npt.NDArray[np.float64]):
        next_pose = self.pose @ ef_delta
        self.joints[:6] = self.solver.backward(next_pose, self.joints[:6])
        self.pose = next_pose

    def set_with_joints(self, joints: npt.NDArray[np.float64]):
        pose = self.solver.forward(joints[:6])
        self.pose = pose
        self.joints = joints.copy()

    def open_gripper(self):
        self.joints[6] = math.pi

    def close_gripper(self):
        self.joints[6] = 0
