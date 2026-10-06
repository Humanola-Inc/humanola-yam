import math
import multiprocessing as mp
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

from .state import ArmState


@dataclass
class JointCtrl:
    joints: npt.NDArray[np.float64]
    duration: Optional[float]


class GetJoint:
    pass


class Close:
    pass


def offshore_controller(
    channel: str,
    sim: bool,
    is_ready,
    ctrl_rx,
    joints_rx,
):
    arm = get_yam_robot(
        channel=channel,
        arm_type=ArmType.YAM,
        gripper_type=GripperType.LINEAR_4310,
        zero_gravity_mode=True,
        sim=sim,
        enable_auto_recovery=True,
    )

    def joints_loop():
        while True:
            ev = joints_rx.recv()
            if isinstance(ev, Close):
                break
            elif isinstance(ev, GetJoint):
                joints = np.zeros(7)
                obs = arm.get_observations()
                joints[:6] = obs["joint_pos"]
                joints[6] = obs["gripper_pos"]
                joints_rx.send(joints)

    def ctrl_loop():
        while True:
            ev = ctrl_rx.recv()
            if isinstance(ev, Close):
                break
            elif isinstance(ev, JointCtrl):
                if ev.duration is None:
                    arm.command_joint_pos(ev.joints)
                else:
                    arm.move_joints(ev.joints, ev.duration)

    joints_thread = threading.Thread(target=joints_loop, daemon=True)
    ctrl_thread = threading.Thread(target=ctrl_loop, daemon=True)
    joints_thread.start()
    ctrl_thread.start()
    is_ready.set()
    try:
        joints_thread.join()
        ctrl_thread.join()
    finally:
        arm.close()


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
        self.__start_process()
        if init_state:
            joints = self.get_joints()
            self.state.set_with_joints(joints)
        else:
            self.update()

    def __start_process(self):
        self.is_ready = mp.Event()
        ctrl_rx, self.ctrl_tx = mp.Pipe(duplex=False)
        self.joints_tx, joints_rx = mp.Pipe()
        self.process = mp.Process(
            target=offshore_controller,
            args=(
                self.channel,
                self.sim,
                self.is_ready,
                ctrl_rx,
                joints_rx,
            ),
        )
        self.process.start()
        self.is_ready.wait()

    def update(self):
        self.ctrl_tx.send(JointCtrl(self.state.joints, None))

    def update_over_time(self, target_joints: npt.NDArray[np.float64], duration: float):
        self.state.set_with_joints(target_joints)
        self.ctrl_tx.send(JointCtrl(target_joints, duration))

    def get_joints(self) -> npt.NDArray[np.float64]:
        self.joints_tx.send(GetJoint())
        return self.joints_tx.recv()

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
        self.__start_process()

    def close(self):
        self.update_over_time(np.array([0, 1, 1, -1, 1, 1, 0]) * math.pi / 180, 1)
        self.joints_tx.send(Close())
        self.ctrl_tx.send(Close())
        self.process.join()
