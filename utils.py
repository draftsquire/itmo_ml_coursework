import matplotlib.pyplot as plt
import numpy as np

def plot_tracking(simout, q, qdes):
    q = np.asarray(q)
    qdes = np.asarray(qdes)
    x, y, phi = q[:,0], q[:,1], q[:,2]
    x_des, y_des, phi_des = qdes[:,0], qdes[:,1], qdes[:,2]
    # Plot trajectory in the Oxy plane

    # Create a figure with two subplots (1 column, 2 rows)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2 rows, 1 column

    axs[0][0].plot(simout.times, x, label="X", color="blue")
    axs[0][0].plot(simout.times, x_des, label="X_des", color="red", linestyle="--")
    axs[0][0].set_xlabel("Time")
    axs[0][0].set_ylabel("X")
    axs[0][0].set_title("X Coordinate over Time")
    axs[0][0].grid(True)
    axs[0][0].legend()

    # Subplot 2: Y coordinates over time (top-right)

    axs[0][1].plot(simout.times, y, label="Y", color="blue")
    axs[0][1].plot(simout.times, y_des, label="Y_des", color="red", linestyle="--")
    axs[0][1].set_xlabel("Time")
    axs[0][1].set_ylabel("Y")
    axs[0][1].set_title("Y Coordinate over Time")
    axs[0][1].grid(True)
    axs[0][1].legend()

    # Subplot 3: Robot trajectory in the Oxy plane
    axs[1][0].plot(x, y, label="$Traj$", color="blue")
    axs[1][0].scatter(x_des, y_des, label="$Traj_{des}$", color="red", s=50, marker="x")
    axs[1][0].set_xlabel("X")
    axs[1][0].set_ylabel("Y")
    axs[1][0].set_title("Robot Trajectory in the Oxy Plane")
    axs[1][0].grid(True)
    axs[1][0].axis('equal')
    axs[1][0].legend()

    # Subplot 4: Desired and actual angles of rotation over time
    axs[1][1].plot(simout.times, phi_des , label="$\\varphi_{des}$", color="green", linestyle="--")
    axs[1][1].plot(simout.times, phi, label="$\\varphi_{act}$", color="orange")
    axs[1][1].set_xlabel("Time")
    axs[1][1].set_ylabel("Angle (deg)")
    axs[1][1].set_title("Desired and Actual Angles of Rotation")
    axs[1][1].grid(True)
    axs[1][1].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()