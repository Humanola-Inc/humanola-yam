import pathlib

import casadi
import numpy as np
import numpy.typing as npt
import pinocchio as pin
import pinocchio.casadi as cpin


class ArmSolver:
    def __init__(self, urdf_file: pathlib.Path, compile: bool = True):
        self.model = pin.RobotWrapper.BuildFromURDF(
            str(urdf_file), package_dirs=str(urdf_file.parent)
        )
        self.model = self.model.buildReducedRobot(
            list_of_joints_to_lock=["joint7", "joint8"],
            reference_configuration=np.zeros(self.model.model.nq),
        )
        # End Effector Frame, we will protrude it a little from the
        # last motor (for grippers)
        self.ef_frame_id = self.model.model.addFrame(
            pin.Frame(
                "ee",
                self.model.model.getJointId("joint6"),
                pin.SE3(np.eye(3), np.array([0, 0, 0.2])),
                pin.FrameType.OP_FRAME,
            )
        )
        self.cmodel = cpin.Model(self.model.model)
        self.cdata = self.cmodel.createData()

        # Forwards and Inverse
        self.cq = casadi.SX.sym("q", 6, 1)  # 6 DOF (joint angles)
        self.cTf = casadi.SX.sym("tf", 4, 4)  # End effector transform
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # End effector error
        ef_tf_error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.ef_frame_id].inverse() * cpin.SE3(self.cTf)
                    ).vector
                )
            ],
        )

        # optimisation
        opti = casadi.Opti()
        var_q = opti.variable(6)
        prev_var_q = opti.parameter(6)
        param_tf = opti.parameter(4, 4)

        # optimiser error
        opti.subject_to(
            opti.bounded(
                self.model.model.lowerPositionLimit,
                var_q,
                self.model.model.upperPositionLimit,
            )
        )
        opti.minimize(
            20 * casadi.sumsqr(ef_tf_error(var_q, param_tf)[:3])
            + 2 * casadi.sumsqr(ef_tf_error(var_q, param_tf)[3:])
            + 0.5 * casadi.sumsqr(var_q - prev_var_q)
            + 0.02 * casadi.sumsqr(var_q)
        )
        opti.solver(
            "ipopt",
            {
                "ipopt": {"print_level": 0, "max_iter": 20, "tol": 1e-2},
                "print_time": False,
                "jit": compile,
                "jit_options": {
                    "flags": ["-Ofast", "-march=native"],
                    "compiler": "gcc",
                },
                "jit_cleanup": True,
                "expand": True,
            },
        )
        self.solve_fn = opti.to_function(
            "ik",
            [var_q, param_tf, prev_var_q],
            [var_q],
            ["q_init", "tf", "prev_q"],
            ["q_sol"],
        )

    def forward(self, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        data = self.model.model.createData()
        pin.forwardKinematics(self.model.model, data, joints)
        pin.updateFramePlacements(self.model.model, data)
        ef_frame_tf = data.oMf[self.ef_frame_id].homogeneous
        return ef_frame_tf

    def backward(
        self,
        pose: npt.NDArray[np.float64],
        prev_joints: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        if prev_joints is None:
            prev_joints = np.zeros(6)
        joints = self.solve_fn(prev_joints, pose, prev_joints)
        return np.array(joints).flatten()
