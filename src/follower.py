from __future__ import annotations

import argparse
import math

import numpy as np
import numpy.typing as npt
from humanola import constants, dataset, robo
from native import Arm, YamNotSim, YamSim

LEADER_TOPIC = "ctl/yam-leader"

RUNTIME = None

ARM_FIELDS = [
    dataset.Field(
        name=f"arm.joint{i + 1}",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    )
    for i in range(6)
] + [
    dataset.Field(
        name="arm.joint7",
        dtype=dataset.LengthType(unit=dataset.LengthUnit.M, shape=[1]),
    ),
    dataset.Field(
        name="arm.joint8",
        dtype=dataset.LengthType(unit=dataset.LengthUnit.M, shape=[1]),
    ),
    dataset.Field(
        name="arm.gripper",
        dtype=dataset.AngleType(unit=dataset.AngleUnit.RAD, shape=[1]),
    ),
]


def joints_frame(joints: npt.NDArray[np.float64]) -> robo.DataFrame:
    frame = robo.DataFrame()
    for joint in joints[:6]:
        frame.attach(dataset.Num(joint))
    gripper = joints[6]
    frame.attach(dataset.Num(gripper * -0.04695))
    frame.attach(dataset.Num(gripper * -0.04695))
    frame.attach(dataset.Num(gripper))
    return frame


class FollowerAction:
    def __init__(self):
        self.stream = None

    def open(self) -> FollowerAction:
        self.stream = RUNTIME.get_hub().open_data(
            topic="act/arms",
            name="Yam Follower Action",
            desc="Records the joint targets sent to the yam follower arm",
            to=[robo.Addr.Dataset],
            fields=ARM_FIELDS,
        )
        return self

    def record(self, joints: npt.NDArray[np.float64]) -> None:
        self.stream.capture(joints_frame(joints).update_tp_us())

    def close_stream(self) -> None:
        self.stream.close()
        self.stream = None


class FollowerCtrl:
    QUERIES: list[robo.Query] = [
        robo.Query(kind=robo.SensorKind.RawDouble, name=name)
        for name in [f"joint{i + 1}" for i in range(6)] + ["gripper"]
    ]

    def __init__(self, arm: Arm):
        self.arm = arm
        self.action = FollowerAction()

    def open(self) -> FollowerCtrl:
        self.action.open()
        return self

    def recv_delta(self, prev: robo.Device, cur: robo.Device) -> None:
        sensors = [cur.get(q) for q in FollowerCtrl.QUERIES]
        if any(s is None for s in sensors):
            return
        joints = np.array([s.as_raw().value for s in sensors])
        self.arm.update(joints)
        self.action.record(joints)

    def close_stream(self) -> None:
        self.action.close_stream()


class FollowerData:
    def __init__(self, arm: Arm):
        self.arm = arm

    def open(self) -> FollowerData:
        return self

    def get_data(self) -> robo.DataFrame:
        return joints_frame(self.arm.get_joints())

    def close_stream(self) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--channel", default="can0")
    args = parser.parse_args()

    arm: Arm = YamSim(args.channel) if args.sim else YamNotSim(args.channel)
    arm.start()

    channel, runtime = (
        robo.RoboConfig(
            api_url="https://grpc.humanola.com",
            api_key="<YOUR_API_KEY>",
            robo_id="<YOUR_ROBO_ID>",
            name="Yam follower",
        )
        .attach_device_subscriber(
            topic=LEADER_TOPIC,
            name="Yam follower arm",
            desc="Mirrors the joints of the yam leader arm",
            rate=60,
            v=FollowerCtrl(arm),
        )
        .attach_data(
            topic=constants.SRC_DATA,
            name="Yam Follower",
            desc="Records the joint position of the yam follower arm",
            rate=120,
            v=FollowerData(arm),
            fields=ARM_FIELDS,
        )
        .run()
    )
    RUNTIME = runtime
    try:
        runtime.wait_for_interrupt()
    finally:
        arm.update_over_time(np.array([0, 1, 1, -1, 1, 1, 0]) * math.pi / 180, 1)
        arm.close()
