import sklearn.datasets
import sklearn.metrics
import autosklearn.regression
from joblib import dump, load
from results import simulateRun
from preprocessing import preprocessing, load_demos

#Load data
demos = load_demos()
X, Y, all_x0, xT = preprocessing(demos)
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y)

#Train model
model = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=3600,
    per_run_time_limit=500,
    tmp_folder='/tmp/autosklearn_regression_sr_tmp',
    output_folder='/tmp/autosklearn_regression_sr_out',
)
model.fit(X, Y)
print(model.show_models())
predictions = model.predict(X)
print("R2 score:", sklearn.metrics.r2_score(Y, predictions))

#Save model
dump(model, 'rf_robot_model_2.joblib') 

#Load model
load('./models/robot_model_2.joblib') 
# model.refit(X, Y)

#Plot results
simulateRun(model, all_x0, xT)