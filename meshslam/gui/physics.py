"""
PyBullet physics integration for the Triangle SLAM GUI.

Provides rigid-body collisions (bouncy balls) and particle-based pseudo-liquid
simulation on top of the reconstructed mesh.
"""
import os
import tempfile
import numpy as np

import pybullet as p
import pybullet_data


class PhysicsEngine:
    """Wraps a PyBullet DIRECT server with the SLAM mesh as a static collision body."""

    def __init__(self):
        self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / 240, physicsClientId=self._client)

        self._mesh_body = None
        self._ball_ids = []
        self._liquid_ids = []
        self._liquid_types = []  # "liquid", "water", or "confetti" per particle
        self._tmp_obj = None  # temp file for mesh collision

        # Materials
        self._ball_radius = 0.02
        self._liquid_radius = 0.005

    def set_gravity(self, gravity):
        """Set gravity direction. gravity: [gx, gy, gz] in m/s^2."""
        p.setGravity(*gravity, physicsClientId=self._client)

    def set_mesh(self, vertices, triangles):
        """Load the triangle mesh as a static collision body.

        vertices: (V, 3) float64 numpy array
        triangles: (F, 3) int32 numpy array
        """
        # Remove old mesh
        if self._mesh_body is not None:
            p.removeBody(self._mesh_body, physicsClientId=self._client)
            self._mesh_body = None

        if self._tmp_obj is not None and os.path.exists(self._tmp_obj):
            os.unlink(self._tmp_obj)

        # Write mesh to temp OBJ for PyBullet
        fd, path = tempfile.mkstemp(suffix=".obj")
        with os.fdopen(fd, "w") as f:
            for v in vertices:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for tri in triangles:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        self._tmp_obj = path

        col_shape = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=path,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=self._client,
        )
        self._mesh_body = p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=col_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self._client,
        )
        p.changeDynamics(
            self._mesh_body, -1,
            restitution=0.2,
            lateralFriction=0.8,
            physicsClientId=self._client,
        )

    def spawn_ball(self, position, velocity=None, radius=None):
        """Spawn a bouncy ball at the given world position."""
        r = radius or self._ball_radius
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=self._client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1, 0.2, 0.2, 1],
                                  physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
            physicsClientId=self._client,
        )
        p.changeDynamics(body, -1, restitution=0.3, lateralFriction=0.5,
                         linearDamping=0.1, angularDamping=0.1,
                         physicsClientId=self._client)
        if velocity is not None:
            p.resetBaseVelocity(body, linearVelocity=list(velocity),
                                physicsClientId=self._client)
        self._ball_ids.append(body)
        return body

    def pour_liquid(self, center, n_particles=200, spread=0.05, velocity=None):
        """Spawn a cluster of small spheres as pseudo-liquid."""
        r = self._liquid_radius
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=self._client)

        vel = velocity if velocity is not None else [0, 0, -0.5]
        for _ in range(n_particles):
            offset = np.random.randn(3) * spread
            offset[2] = abs(offset[2])  # spawn above center
            pos = np.array(center) + offset
            body = p.createMultiBody(
                baseMass=0.002,
                baseCollisionShapeIndex=col,
                basePosition=pos.tolist(),
                physicsClientId=self._client,
            )
            p.changeDynamics(body, -1, restitution=0.1, lateralFriction=0.1,
                             rollingFriction=0.01, physicsClientId=self._client)
            p.resetBaseVelocity(body, linearVelocity=vel, physicsClientId=self._client)
            self._liquid_ids.append(body)
            self._liquid_types.append("liquid")

    def pour_water(self, center, n_particles=500, spread=0.02, velocity=None):
        """Spawn many tiny droplets that behave like water."""
        r = 0.002  # much smaller than regular liquid
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=self._client)
        vel = velocity if velocity is not None else [0, 0, -0.5]
        for _ in range(n_particles):
            offset = np.random.randn(3) * spread
            offset[2] = abs(offset[2])
            pos = np.array(center) + offset
            body = p.createMultiBody(
                baseMass=0.0005,
                baseCollisionShapeIndex=col,
                basePosition=pos.tolist(),
                physicsClientId=self._client,
            )
            p.changeDynamics(body, -1, restitution=0.05, lateralFriction=0.02,
                             rollingFriction=0.001, physicsClientId=self._client)
            jitter = (np.random.randn(3) * 0.1).tolist()
            v = [vel[0] + jitter[0], vel[1] + jitter[1], vel[2] + jitter[2]]
            p.resetBaseVelocity(body, linearVelocity=v, physicsClientId=self._client)
            self._liquid_ids.append(body)
            self._liquid_types.append("water")

    def pour_confetti(self, center, n_particles=200, spread=0.08, velocity=None):
        """Spawn lightweight confetti pieces that flutter and scatter."""
        r = 0.006  # visible but flat-ish
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=self._client)
        vel = velocity if velocity is not None else [0, 0, -0.3]
        for _ in range(n_particles):
            offset = np.random.randn(3) * spread
            offset[2] = abs(offset[2]) * 1.5  # spread more vertically
            pos = np.array(center) + offset
            body = p.createMultiBody(
                baseMass=0.0002,  # very light
                baseCollisionShapeIndex=col,
                basePosition=pos.tolist(),
                physicsClientId=self._client,
            )
            p.changeDynamics(body, -1, restitution=0.4, lateralFriction=0.8,
                             rollingFriction=0.1, linearDamping=0.3,
                             physicsClientId=self._client)
            # Random horizontal scatter
            jitter = np.random.randn(3) * 0.3
            v = [vel[0] + jitter[0], vel[1] + jitter[1], vel[2] + jitter[2] * 0.3]
            p.resetBaseVelocity(body, linearVelocity=v, physicsClientId=self._client)
            self._liquid_ids.append(body)
            self._liquid_types.append("confetti")

    def step(self, n=1):
        """Advance the simulation by n substeps."""
        for _ in range(n):
            p.stepSimulation(physicsClientId=self._client)

    def get_ball_positions(self):
        """Return (N, 3) array of ball positions."""
        if not self._ball_ids:
            return np.zeros((0, 3))
        positions = []
        for bid in self._ball_ids:
            pos, _ = p.getBasePositionAndOrientation(bid, physicsClientId=self._client)
            positions.append(pos)
        return np.array(positions)

    def get_liquid_positions(self):
        """Return (N, 3) array of liquid particle positions."""
        if not self._liquid_ids:
            return np.zeros((0, 3))
        positions = []
        for lid in self._liquid_ids:
            pos, _ = p.getBasePositionAndOrientation(lid, physicsClientId=self._client)
            positions.append(pos)
        return np.array(positions)

    def get_liquid_types(self):
        """Return list of type strings parallel to get_liquid_positions."""
        return list(self._liquid_types)

    def clear(self):
        """Remove all dynamic objects (balls + liquid)."""
        for bid in self._ball_ids:
            p.removeBody(bid, physicsClientId=self._client)
        for lid in self._liquid_ids:
            p.removeBody(lid, physicsClientId=self._client)
        self._ball_ids.clear()
        self._liquid_ids.clear()
        self._liquid_types.clear()

    def cleanup(self):
        """Disconnect PyBullet and remove temp files."""
        self.clear()
        if self._mesh_body is not None:
            p.removeBody(self._mesh_body, physicsClientId=self._client)
        p.disconnect(self._client)
        if self._tmp_obj is not None and os.path.exists(self._tmp_obj):
            os.unlink(self._tmp_obj)
