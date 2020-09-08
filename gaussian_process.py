import os
import sklearn.metrics
from joblib import dump, load
from results import simulateRun
from preprocessing import preprocessing, load_demos
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, ConstantKernel

#Load data
demos = load_demos(num_demos=3)
X, Y, all_x0, xT = preprocessing(demos)
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y)

# kernel = ConstantKernel(61.5**2) * RBF(0.0794) + WhiteKernel(0.000332)
# kernel = ConstantKernel(63.9**2) * Matern(0.147, nu=1.5) + WhiteKernel(1e-5)
kernel = ConstantKernel(0.00316**2) + Matern(length_scale=8.7, nu=3/2) + WhiteKernel(noise_level=1.47e+3)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
model.fit(X[:,:6], Y[:,:6])
print(model.score(X[:,:6], Y[:,:6]))

#Save model
model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/gpr_only_pose.joblib")
dump(model, model_name) 

#Load model
# model = load('./models/gpr.joblib') 

#Plot results
simulateRun(model, all_x0[:,:6], xT[:6])





