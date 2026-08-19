import numpy as np
import pyvista as pv


class AttitudeVisualizer:
    def __init__(
        self,
        times,
        quaternions,
        h_values,
        etas,
        omega_squared,
        actuator_effort,
        frame_step=20
    ):
        self.times = np.asarray(times)
        self.quaternions = np.asarray(quaternions)
        self.h_values = np.asarray(h_values)
        self.etas = np.asarray(etas)
        self.omega_squared = np.asarray(omega_squared)
        self.actuator_effort = np.asarray(actuator_effort)

        self.frame_step = frame_step

        self.plotter = pv.Plotter(
            window_size=(1800, 900)
        )

        # Charts
        self.chart_h = pv.Chart2D(
            size=(0.32, 0.20),
            loc=(0.66, 0.76)
        )

        self.chart_eta = pv.Chart2D(
            size=(0.32, 0.20),
            loc=(0.66, 0.52)
        )

        self.chart_omega = pv.Chart2D(
            size=(0.32, 0.20),
            loc=(0.66, 0.28)
        )

        self.chart_effort = pv.Chart2D(
            size=(0.32, 0.20),
            loc=(0.66, 0.04)
        )

        # Axis labels
        self.chart_h.x_axis.label = "Time [s]"
        self.chart_h.y_axis.label = "F h"

        self.chart_eta.x_axis.label = "Time [s]"
        self.chart_eta.y_axis.label = "eta"

        self.chart_omega.x_axis.label = "Time [s]"
        self.chart_omega.y_axis.label = "omega^T omega"

        self.chart_effort.x_axis.label = "Time [s]"
        self.chart_effort.y_axis.label = "Control effort"

        for chart in [
            self.chart_h,
            self.chart_eta,
            self.chart_omega,
            self.chart_effort
        ]:
            chart.x_axis.tick_label_size = 16
            chart.y_axis.tick_label_size = 16

            chart.x_axis.label_size = 18
            chart.y_axis.label_size = 18

        self.line_h = self.chart_h.line([], [])
        self.line_eta = self.chart_eta.line([], [])
        self.line_omega = self.chart_omega.line([], [])
        self.line_effort = self.chart_effort.line([], [])


        self.plotter.add_chart(
            self.chart_h,
            self.chart_eta,
            self.chart_omega,
            self.chart_effort
        )

        self.satellite = None
        self.reference_points = None
        self.body_axes = None

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion q = [eta, eps1, eps2, eps3]
        into a 3x3 rotation matrix.
        """
        q = q / np.linalg.norm(q)

        eta = q[0]
        eps = q[1:]

        S = np.array([
            [0.0, -eps[2], eps[1]],
            [eps[2], 0.0, -eps[0]],
            [-eps[1], eps[0], 0.0]
        ])

        return np.eye(3) + 2.0 * eta * S + 2.0 * S @ S

    def build_satellite(self):
        """
        Load the Galileo spacecraft STL model and prepare it
        for quaternion-based attitude animation.
        """

        self.satellite = pv.read("assets/Galileo.stl")

        # Smooth surface appearance
        self.satellite = self.satellite.compute_normals(
            point_normals=True,
            cell_normals=False,
            split_vertices=False
        )

        # The NASA STL is already centered approximately at
        # the spacecraft body-frame origin (0, 0, 0).
        self.reference_points = self.satellite.points.copy()

        # Body-frame axes
        axis_length = 5.0
        origin = np.array([0.0, 0.0, 0.0])

        x_axis = pv.Line(
            origin,
            origin + np.array([axis_length, 0.0, 0.0])
        )

        y_axis = pv.Line(
            origin,
            origin + np.array([0.0, axis_length, 0.0])
        )

        z_axis = pv.Line(
            origin,
            origin + np.array([0.0, 0.0, axis_length])
        )

        self.body_axes = [x_axis, y_axis, z_axis]

        self.body_axes_reference = [
            axis.points.copy() for axis in self.body_axes
        ]

    def animate(self):
        """
        Animate the satellite using the simulated quaternion trajectory.
        """

        if self.satellite is None:
            self.build_satellite()

        # Add spacecraft
        self.plotter.add_mesh(
            self.satellite,
            smooth_shading=True,
            show_edges=False
        )

        # Add body-frame axes
        self.plotter.add_axes()
        self.plotter.show_grid()

        self.plotter.add_mesh(self.body_axes[0], color="red", line_width=5)
        self.plotter.add_mesh(self.body_axes[1], color="green", line_width=5)
        self.plotter.add_mesh(self.body_axes[2], color="blue", line_width=5)

        # Default isometric camera view
        self.plotter.view_isometric()
        self.plotter.reset_camera()

        # Zoom out to fit the complete spacecraft
        self.plotter.camera.zoom(0.6)

        # Show window without blocking execution
        self.plotter.show(auto_close=False, interactive_update=True)

        # Animation loop
        for i in range(0, len(self.quaternions), self.frame_step):

            q = self.quaternions[i]

            R = self.quaternion_to_rotation_matrix(q)

            # Rotate satellite
            self.satellite.points = self.reference_points @ R.T

            # Rotate body-frame axes
            for axis, ref_points in zip(
                self.body_axes,
                self.body_axes_reference
            ):
                axis.points = ref_points @ R.T

            # Update charts
            self.line_h.update(
                self.times[:i + 1],
                self.h_values[:i + 1]
            )

            self.line_eta.update(
                self.times[:i + 1],
                self.etas[:i + 1]
            )

            self.line_omega.update(
                self.times[:i + 1],
                self.omega_squared[:i + 1]
            )

            self.line_effort.update(
                self.times[:i + 1],
                self.actuator_effort[:i + 1]
            )

            self.plotter.render()
            self.plotter.update()

        self.plotter.show()