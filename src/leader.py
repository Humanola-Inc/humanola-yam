from __future__ import annotations

import argparse

from humanola import robo
from native import Arm, YamLeaderNotSim, YamSim

LEADER_TOPIC = "ctl/yam-leader"


class LeaderDevice:
    def __init__(self, arm: Arm):
        self.arm = arm

    def open(self) -> LeaderDevice:
        return self

    def get_device(self) -> robo.Device:
        device = robo.Device(kind="yam", name="leader")
        device.timestamp_now()
        joints = self.arm.get_joints()
        for i, joint in enumerate(joints[:6]):
            device.add_reading(f"joint{i + 1}", robo.RawDoubleState(value=float(joint)))
        device.add_reading("gripper", robo.RawDoubleState(value=float(joints[6])))
        return device

    def close_stream(self) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--channel", default="can0")
    args = parser.parse_args()

    arm: Arm = YamSim(args.channel) if args.sim else YamLeaderNotSim(args.channel)
    arm.start()

    channel, runtime = (
        robo.RoboConfig(
            api_url="https://grpc.humanola.com",
            api_key="<YOUR_API_KEY>",
            robo_id="<YOUR_ROBO_ID>",
            name="Yam leader",
        )
        .attach_device(
            topic=LEADER_TOPIC,
            name="Yam leader arm",
            desc="Publishes the joints of the yam leader arm",
            rate=60,
            v=LeaderDevice(arm),
        )
        .run()
    )
    try:
        runtime.wait_for_interrupt()
    finally:
        arm.close()
