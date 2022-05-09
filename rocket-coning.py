#!python
from math import cos, radians, sin
from numpy import array, diag, dot, cross, linspace, squeeze, degrees
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
import toml
import matplotlib.pyplot as plt

'''
Based on the following work:
[1] D. H. Platus, “Missile and spacecraft coning instabilities,” Journal of Guidance, Control, and Dynamics, vol. 17, no. 5, pp. 1011–1018, Sep. 1994, doi: 10.2514/3.21303.
'''
import vtkmodules.vtkRenderingOpenGL2  # DO NOT REMOVE --- will fail silently if so
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
)

""" 
What should this tool do?
    Input: rocket moments of inertia, pitch moment coefficient, free stream conditions
    Output: Some idea of stability regions of the rocket with regard to the coning mode
        Function of rocket physical characteristics and...
        1) pitch angle theta
        2) roll rate phidot
        3) free stream conditions
        Output can be quasi-static stability exponent surfaces for reasonable theta and phidot.
        If the user clicks anywhere in the graph, we can take those conditions and simulate the evolution to see the results.
    For controls stuff  -> can add in some callback for a control torque that we can try to use to stabilize and check the sim again
                        -> also can implement the papers suggestions and give specs needed to stabilize at given point
                
"""


class ProblemInstance:
    def __init__(self, config):

        self.config = config

        self.A = config["shape"]["A"]
        self.Cna = config["shape"]["Cnalpha"]
        self.Cma = config["shape"]["Cmalpha"]
        self.d = config["shape"]["d"]
        self.A = config["shape"]["A"]

        rho = config["freestream"]["rho"]
        v = config["freestream"]["v"]

        # Calculate pitching torque derivative and magnus torque derivative with angle of attack
        self.Na = 0.5 * rho * v**2 * self.A * self.d * self.Cna
        self.Ma = (
            lambda w: 0.5 * rho * self.A *
            (w * self.d / v) * self.d * v**2 * self.Cma
        )

        # Calculate axisymetric moment of inertia ratio
        self.mu = config['mass']['Ixx']/config['mass']['I']

        # Some values from [1]
        self.kappa = lambda Ma: Ma/config['mass']['I']
        self.sigma = lambda Na, Ma: Na/Ma

        # Formulas for the decay rate of the two pitching modes.
        # Positive values here will indicate unstable growth in angle of attack
        self.lam1 = lambda k, s, u, p: k*(u*p - (2*s*k/(u*p)))**-1
        self.lam2 = lambda k, s, u, p: -k*(u*p - (2*s*k/(u*p)))**-1

    def plotStabilityExponents(self):
        # Roll rate of the rocket in the body frame around the x axis
        # We plot the decay rate as a function of the roll rate
        p = linspace(0, self.config['parameters']['omega_max'],
                     self.config['parameters']['omega_step'])

        # Realizing the actual values here for values of p
        Ma = self.Ma(p)
        kappa = self.kappa(Ma)
        sigma = self.sigma(self.Na, Ma)

        # Decay rates
        lam1 = self.lam1(kappa, sigma, self.mu, p)
        lam2 = self.lam2(kappa, sigma, self.mu, p)

        # Plot the decay rates for the vs the roll rates
        plt.figure()
        plt.title('positive mode')
        plt.xlabel('roll rate (Hz)')
        plt.plot(p, lam1)

        plt.figure()
        plt.title('negative mode')
        plt.xlabel('roll rate (rad-s^-1)')
        plt.plot(p, lam2)

        plt.show()

    def omegaPrecession(self, omega_s, I_s, I_p, alpha):
        """Calculate the roll rate that will result in pure precession for a given angle of attack and precession rate.

        Args:
            omega_s (float): precession rate
            I_s (float): axisymetric moment of inertia
            I_p (float): perpendicular moment of inertia
            alpha (float): angle of attack

        Returns:
            float: roll rate
        """
        return I_s * omega_s / (I_p * cos(alpha))


class DynamicsEulerRocket:
    """Initializes and calculates state space representation of rocket rotational dynamics only.
    """

    def __init__(self, m, I) -> None:
        self.m = m
        self.I = I

    def __call__(self, t, y, u) -> array:
        """Get the state derivative.

        Args:
            t (float): simulation time
            y (float): state; this is the quaternion that orients vectors in the body frame to the world frame and the angular rate in the body frame.
            u (float): input to the system; this is the torque on the system in the body frame

        Returns:
            array: state derivative
        """
        q0, q1, q2, q3, wx, wy, wz = y

        q = array([q0, q1, q2, q3])
        qw = array([0, wx, wy, wz])
        qdot = 0.5 * self.quat_mult(q, qw)

        w = array([wx, wy, wz])

        # TODO Torque needs to come from world frame?
        wdot = inv(self.I) @ (u + cross(self.I @ w, w))

        return array(
            [
                qdot[0],
                qdot[1],
                qdot[2],
                qdot[3],
                wdot[0],
                wdot[1],
                wdot[2],
            ]
        )

    def quat_mult(self, q, p):
        """Multiplies two quaternions

        Args:
            q (array): q1
            p (array): q2

        Returns:
            array: q1 multiplied with q2
        """
        q0 = q[0]
        p0 = p[0]
        qvec = q[1:]
        pvec = p[1:]

        res0 = q0 * p0 - dot(pvec, qvec)
        resvec = q0 * pvec + p0 * qvec + cross(qvec, pvec)
        return array([res0, resvec[0], resvec[1], resvec[2]])

    def makeInitialConditions(self, p_initial):
        """Makes initial conditions for the dynamics. Assumes a small angle of attack. Gives a starting rotation.

        Args:
            p_initial (float): initial roll rate in the body frame

        Returns:
            array: initial state for the solver
        """

        phi = radians(0)
        psi = radians(0)
        theta = radians(5)

        thetadot = 0.0
        phidot = p_initial
        psidot = 0.0

        winitial = array(
            [
                psidot * cos(theta) + phidot,
                thetadot * cos(phi) + psidot * sin(theta) * sin(phi),
                psidot * sin(theta) * cos(phi) - thetadot * sin(phi),
            ]
        )

        RB2N = Rotation.from_euler("XYX", [psi, theta, phi])
        qinitial = RB2N.as_quat()

        return array(
            [
                qinitial[3],
                qinitial[0],
                qinitial[1],
                qinitial[2],
                winitial[0],
                winitial[1],
                winitial[2],
            ]
        )


class vtkSimulationCallback:
    def __init__(self, config, problem, rocketActor, renderInteractor, dt=0.1):

        self.problem = problem
        self.rocketActor = rocketActor
        self.renderInteractor = renderInteractor
        self.dt = dt

        m = config["mass"]["m"]
        Ixx = config["mass"]["Ixx"]
        I = config["mass"]["I"]

        self.dyn = DynamicsEulerRocket(m, diag([Ixx, I, I]))
        self.y0 = self.dyn.makeInitialConditions(
            config['parameters']['omega_max']/2)

    def execute(self, obj, event):
        # Get the moment derivatives
        Na = self.problem.Na
        Ma = self.problem.Ma

        # Get the euler rotations from the current state. yaw, pitch, roll
        _, theta, phi = Rotation.from_quat(
            [self.y0[1], self.y0[2], self.y0[3], self.y0[0]]).as_euler("XYX")

        # Get a rotation just for the roll
        # This is because the moments we calculate are actually in an intermediate frame and we need to bring them to the body frame.
        RB2I = Rotation.from_euler('X', [phi])

        # Calculate the moments
        N = Na*theta * array([0, 1, 0])
        p_current = norm(self.y0[4:])
        M = Ma(p_current)*theta * array([0, 0, 1])

        # Calculate the control torques in the body frame
        u_interval = squeeze(RB2I.inv().apply(N+M))

        # Solve the dynamics integration step
        sol = solve_ivp(
            self.dyn, (0.0, self.dt), self.y0, t_eval=(self.dt,), args=(u_interval,)
        )
        y1 = sol.y[:, -1]
        RB2N = Rotation.from_quat([y1[1], y1[2], y1[3], y1[0]])
        eulerB2N = RB2N.as_euler("ZXY", True)

        # Update the visualization
        self.rocketActor.SetOrientation(
            eulerB2N[1], eulerB2N[2], eulerB2N[0])

        self.renderInteractor = obj
        self.renderInteractor.GetRenderWindow().Render()

        self.y0 = y1

        # Stop if our angle of attack reaches the threshold (we are bound for a flat spin)
        if degrees(theta) > config['simulation']['alpha_stop']:
            self.renderInteractor.DestroyTimer()


class SimulationRenderWindow(vtkRenderWindow):
    def __init__(self, config) -> None:
        super().__init__()

        dt = config['simulation']['dt']

        self.colors = vtkNamedColors()

        self.reader = vtkSTLReader()
        self.reader.SetFileName(config['model']['path'])

        self.mapper = vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())

        self.currentOrientationActor = vtkActor()
        self.currentOrientationActor.GetProperty().SetSpecular(0.6)
        self.currentOrientationActor.GetProperty().SetSpecularPower(30)
        self.currentOrientationActor.SetMapper(self.mapper)
        self.currentOrientationActor.GetProperty().SetColor(
            self.colors.GetColor3d("BurlyWood"))
        # currentOrientationActor.GetProperty().EdgeVisibilityOn()
        # currentOrientationActor.GetProperty().SetEdgeColor(colors.GetColor3d("Red"))

        # Setup a renderer, render window, and interactor
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(self.colors.GetColor3d("MistyRose"))

        self.SetWindowName("Animation")
        self.AddRenderer(self.renderer)
        self.SetSize(1920, 1080)

        self.renderWindowInteractor = vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self)
        self.style = vtkInteractorStyleTrackballCamera()
        self.renderWindowInteractor.SetInteractorStyle(self.style)

        # Add the actor to the scene
        self.renderer.AddActor(self.currentOrientationActor)

        # Render and interact
        self.renderer.GetActiveCamera().SetPosition(2.0, 2.0, 2.0)
        self.renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.Render()

        # Initialize must be called prior to creating timer events.
        self.renderWindowInteractor.Initialize()

        axes = vtkAxesActor()

        self.renderer.AddActor(axes)

        # Sign up to receive TimerEvent
        cb = vtkSimulationCallback(
            config, problem, self.currentOrientationActor, self.renderWindowInteractor, dt=dt
        )
        self.renderWindowInteractor.AddObserver("TimerEvent", cb.execute)
        cb.timerId = self.renderWindowInteractor.CreateRepeatingTimer(
            int(1000 * dt))

    def Start(self):
        self.Render()
        self.renderWindowInteractor.Start()


if __name__ == '__main__':
    config = toml.load("rocket-config.toml")
    problem = ProblemInstance(config)
    problem.plotStabilityExponents()
    simulationViz = SimulationRenderWindow(config)
    simulationViz.Start()
