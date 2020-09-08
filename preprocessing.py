import numpy as np
import matplotlib.pyplot as plt
import glob, os

def preprocessing(demos, tol_cutting=0.005, showPlot=False):
    if showPlot:
        plt.figure()
        ax = plt.axes(projection='3d')

    X, Y, x0, xT = [], [], [], []
    for i in range(len(demos)):
        demo = demos[i]
        time = demo[0]
        pos = demo[1:] #DxN

        #Velocities
        vel = np.diff(pos) / np.diff(time) #Dx(N-1)/(N-1)

        #Trimming (Only save when moving)
        ind = np.nonzero(np.linalg.norm(vel, axis=0) > tol_cutting)
        pos = pos[:, np.min(ind):np.max(ind)+1]
        vel = vel[:, np.min(ind):np.max(ind)]
        vel = np.pad(vel, [(0,0),(0,1)], mode='constant', constant_values=0).T #NxD

        #Plotting position
        if showPlot:
            ax.plot3D(pos[0], pos[1], pos[2])

        #Saving first and final position
        x0.append(pos[:, 0])
        xT.append(pos[:, -1])
        pos = pos.T - xT[i] #NxD
        
        X.append(pos)
        Y.append(vel)
    if showPlot:
        plt.show()

    X = np.concatenate(X, axis=0) #NxD
    Y = np.concatenate(Y, axis=0)
    all_x0 = np.stack(x0)
    xT = np.mean(np.stack(xT), axis=0)
    return X, Y, all_x0, xT

def load_all_demos(dirName = "./demonstrations_project"):
    os.chdir(dirName)
    demos = []
    for file in glob.glob("*.txt"):
        demos.append(np.loadtxt(file).T)
    return demos

def load_demos(name = "./demonstrations_project/two_fingers", num_demos=3):
    demos = []
    for i in range(num_demos):
        demos.append(np.loadtxt("%s_%d.txt" % (name, i+1)).T)
    return demos