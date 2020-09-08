import os
import mogptk
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocessing, load_demos

def initModel(data, Q=3):
    model = mogptk.MOSM(data, Q)
    # model.init_parameters("SM")
    # model.save_parameters('model_init')
    # model.predict(X[:,:6])
    # model.plot_prediction(title='Untrained model')
    return model

def simulateRun(model, all_x0, xT, dt = 0.01, steps=1000):
    #Create trajectories
    traj = []
    cur_x = all_x0
    for i in range(steps):
        traj.append(cur_x)
        cur_v = model.predict(cur_x - xT)
        cur_v = np.stack(cur_v[0]).T 
        cur_x = cur_x + cur_v * dt
    traj = np.stack(traj, axis=-1)

    #Plot trajectories
    plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(all_x0.shape[0]):
        ax.plot3D(traj[i, 0], traj[i, 1], traj[i, 2])
    plt.show()


#Load data
demos = load_demos(num_demos=5)
X, Y, all_x0, xT = preprocessing(demos)

#Creating dataset for mogptk
names=["pose_x", "pose_y", "pose_z", "or_x", "or_y", "or_z", 
       "j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9","Pressure"]
data = []
for i in range(6): #Y.shape[1]
    data.append(mogptk.Data(X[:,:6], Y[:, i], 
                name="vel_" + names[i], x_labels=names))
    data[i].remove_randomly(pct=0.7)

# model = initModel(data)
# model.train( method='L-BFGS-B', tol=1e-6, maxiter=1000, verbose=True)
# model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/mogp_only_pose")
# model.save_parameters(model_name)

# load model
model = mogptk.MOSM(data, Q=3)
model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/mogp_only_pose")
params = model.load_parameters(model_name)

#Plot results
simulateRun(model, all_x0[:, :6], xT[:6], steps=200)