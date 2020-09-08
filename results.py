import numpy as np
import matplotlib.pyplot as plt

def simulateRun(model, all_x0, xT, dt = 0.01, steps=1000):

    #Create trajectories
    traj = []
    cur_x = all_x0
    for i in range(steps):
        traj.append(cur_x)
        cur_v = model.predict(cur_x - xT)
        cur_x = cur_x + cur_v * dt
    traj = np.stack(traj, axis=-1)

    #Plot trajectories
    plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(traj.shape[0]):
        ax.plot3D(traj[i, 0], traj[i, 1], traj[i, 2])
    plt.show()