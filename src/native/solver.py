import pathlib
from typing import Protocol

import casadi
import mink
import mujoco
import numpy as np
import numpy.typing as npt
import pinocchio as pin
import pinocchio.casadi as cpin


class ArmSolver(Protocol):
    def forward(self, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

    def backward(
        self,
        pose: npt.NDArray[np.float64],
        prev_joints: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]: ...


class CasadiSolver:
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


class MujocoSolver:
    def __init__(self, xml_file: pathlib.Path, max_iters: int = 20):
        spec = mujoco.MjSpec.from_file(str(xml_file))
        # only kinematics and inertia are needed, skip loading the meshes
        for geom in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]:
            spec.delete(geom)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
        # End Effector Frame, we will protrude it a little from the
        # last motor (for grippers)
        spec.joint("joint6").parent.add_site(name="ee", pos=[0, 0, 0.2])
        self.model = spec.compile()
        self.configuration = mink.Configuration(self.model)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.max_iters = max_iters

        # position matters more than orientation, with a light pull towards
        # zero to keep away from odd configurations
        self.ef_task = mink.FrameTask(
            frame_name="ee",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.3,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=0.003)
        self.posture_task.set_target(np.zeros(self.model.nq))
        # l2 regularisation towards the previous solution, keeps consecutive
        # solves from jumping between ik branches
        self.smooth_task = mink.PostureTask(self.model, cost=0.1)

    def forward(self, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.configuration.update(joints)
        return self.configuration.get_transform_frame_to_world("ee", "site").as_matrix()

    def backward(
        self,
        pose: npt.NDArray[np.float64],
        prev_joints: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        if prev_joints is None:
            prev_joints = np.zeros(6)
        self.configuration.update(prev_joints)
        self.ef_task.set_target(mink.SE3.from_matrix(pose))
        self.smooth_task.set_target(prev_joints)
        tasks = [self.ef_task, self.posture_task, self.smooth_task]
        for _ in range(self.max_iters):
            # dt = 1 so each step is a full joint displacement
            vel = mink.solve_ik(
                self.configuration,
                tasks,
                1.0,
                "quadprog",
                damping=1e-3,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, 1.0)
            if np.linalg.norm(vel) < 1e-4:
                break
        return self.configuration.q.copy()
