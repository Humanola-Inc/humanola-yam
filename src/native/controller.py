import math
# import multiprocessing as mp
import subprocess
import time
# from dataclasses import dataclass
# from typing import Optional

import numpy as np
import numpy.typing as npt
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

from .state import ArmState


# @dataclass
# class JointCtrl:
#     joints: npt.NDArray[np.float64]
#     duration: Optional[float]
#
#
# class GetJoint:
#     pass
#
#
# class Close:
#     pass
#
#
# def offshore_controller(
#     channel: str,
#     sim: bool,
#     is_ready,
#     ctrl_ev_queue,
#     joints_queue,
# ):
#     arm = get_yam_robot(
#         channel=channel,
#         arm_type=ArmType.YAM,
#         gripper_type=GripperType.LINEAR_4310,
#         zero_gravity_mode=True,
#         sim=sim,
#         enable_auto_recovery=True,
#     )
#     is_ready.set()
#     try:
#         while True:
#             ev = ctrl_ev_queue.get()
#             if isinstance(ev, Close):
#                 break
#             elif isinstance(ev, JointCtrl):
#                 if ev.duration is None:
#                     arm.command_joint_pos(ev.joints)
#                 else:
#                     arm.move_joints(ev.joints, ev.duration)
#             elif isinstance(ev, GetJoint):
#                 joints = np.zeros(7)
#                 obs = arm.get_observations()
#                 joints[:6] = obs["joint_pos"]
#                 joints[6] = obs["gripper_pos"]
#                 joints_queue.put(joints)
#     finally:
#         arm.close()
#
#
# class ArmController:
#     def __init__(
#         self,
#         channel: str,
#         state: ArmState,
#         init_state: bool = True,
#         sim: bool = False,
#     ):
#         self.channel = channel
#         self.sim = sim
#         self.state = state
#         self.__start_process()
#         if init_state:
#             joints = self.get_joints()
#             self.state.set_with_joints(joints)
#         else:
#             self.update()
#
#     def __start_process(self):
#         self.is_ready = mp.Event()
#         self.ctrl_ev_queue = mp.Queue()
#         self.joints_queue = mp.Queue()
#         self.process = mp.Process(
#             target=offshore_controller,
#             args=(
#                 self.channel,
#                 self.sim,
#                 self.is_ready,
#                 self.ctrl_ev_queue,
#                 self.joints_queue,
#             ),
#         )
#         self.process.start()
#         self.is_ready.wait()
#
#     def update(self):
#         self.ctrl_ev_queue.put(JointCtrl(self.state.joints, None))
#
#     def update_over_time(self, target_joints: npt.NDArray[np.float64], duration: float):
#         self.state.set_with_joints(target_joints)
#         self.ctrl_ev_queue.put(JointCtrl(target_joints, duration))
#
#     def get_joints(self) -> npt.NDArray[np.float64]:
#         self.ctrl_ev_queue.put(GetJoint())
#         return self.joints_queue.get()
#
#     def restart(self):
#         self.close()
#         subprocess.run(["ip", "link", "set", self.channel, "down"], check=True)
#         subprocess.run(
#             [
#                 "ip",
#                 "link",
#                 "set",
#                 self.channel,
#                 "up",
#                 "type",
#                 "can",
#                 "bitrate",
#                 "1000000",
#             ],
#             check=True,
#         )
#         self.__start_process()
#
#     def close(self):
#         self.update_over_time(np.array([0, 1, 1, -1, 1, 1, 0]) * math.pi / 180, 1)
#         self.ctrl_ev_queue.put(Close())
#         self.process.join()


class ArmController:
    def __init__(
        self,
        channel: str,
        state: ArmState,
        init_state: bool = True,
        sim: bool = False,
    ):
        self.channel = channel
        self.sim = sim
        self.state = state
        self.__start_arm()
        if init_state:
            joints = self.get_joints()
            self.state.set_with_joints(joints)
        else:
            self.update()

    def __start_arm(self):
        self.arm = get_yam_robot(
            channel=self.channel,
            arm_type=ArmType.YAM,
            gripper_type=GripperType.LINEAR_4310,
            zero_gravity_mode=True,
            sim=self.sim,
            enable_auto_recovery=True,
        )

    def update(self):
        self.arm.command_joint_pos(self.state.joints)

    def update_over_time(self, target_joints: npt.NDArray[np.float64], duration: float):
        self.state.set_with_joints(target_joints)
        self.arm.move_joints(target_joints, duration)

    def get_joints(self) -> npt.NDArray[np.float64]:
        joints = np.zeros(7)
        obs = self.arm.get_observations()
        joints[:6] = obs["joint_pos"]
        joints[6] = obs["gripper_pos"]
        return joints

    def restart(self):
        self.close()
        subprocess.run(["ip", "link", "set", self.channel, "down"], check=True)
        subprocess.run(
            [
                "ip",
                "link",
                "set",
                self.channel,
                "up",
                "type",
                "can",
                "bitrate",
                "1000000",
            ],
            check=True,
        )
        self.__start_arm()

    def close(self):
        try:
            self.update_over_time(np.array([0, 1, 1, -1, 1, 1, 0]) * math.pi / 180, 1)
        finally:
            self.arm.close()
