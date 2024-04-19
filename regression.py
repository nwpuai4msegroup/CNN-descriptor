import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib
import xgboost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import GridSearchCV
def build_model_re(X_train, X_test, y_train, y_test, model='RF', seed=1):
    param_grid = {}
    if model == 'RF':
        model = RandomForestRegressor(random_state=1) #,criterion = 'mae')n_estimators=100, max_depth = 5, min_samples_split = 3

        param_grid = {
            "max_depth": np.arange(1, 11, step=1),
            "n_estimators": np.arange(10, 51, step=5),
        }
    elif model == 'SVR':
        model = svm.SVR()
        param_grid = {
            "C": np.arange(1, 100, step=1),
            "kernel": ['rbf', 'sigmoid'],#, 'precomputed'
            "gamma": np.arange(0.01, 10, step=0.05),

        }
    elif model == 'GBM':
        model = GradientBoostingRegressor()
        param_grid = {
            'n_estimators': np.arange(50, 301, step=50),
            'max_depth': np.arange(1, 11, step=1),
            #'min_samples_split': np.arange(2, 11, step=1),
            #'learning_rate': [0.1, 0.75, 0.05,0.04, 0.03,0.02,0.01, 0.05]
        }
    # elif model == 'NN':
    #     model = MLPRegressor()
    #     param_grid = {
    #         'hidden_layer_sizes': [(32,64,32), (64,64,64), (32,32,32)],
    #         'activation': ['relu','tanh','logistic'],
    #         'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    #         'learning_rate': ['constant','adaptive'],
    #         'solver': ['adam'],
    #         'max_iter':[1000, 2000]
    #     }
    # elif model == 'LR':
    #     model = LinearRegression()
    #     param_grid = {
    #         'fit_intercept': ['True','False']
    #     }
    # elif model == 'KNN':
    #     model = KNeighborsRegressor()
    #     param_grid = {
    #         'n_neighbors': np.arange(5, 11, step=1)
    #     }
    elif model == 'XGB':
        model = xgboost.XGBRegressor()
        param_grid = {
        "max_depth": np.arange(1, 11, step=1),
        "n_estimators": np.arange(10, 51, step=5),
    }
    elif model == 'AdaBoost':
        model = AdaBoostRegressor()
        param_grid = {
        "n_estimators": np.arange(10, 101, step=10),
        "learning_rate": [ 0.01, 0.1, 1.0],
        "loss": ['linear', 'square', 'exponential'],
    }
    #elif model == 'GPR':
    #     model = GaussianProcessRegressor()
    #     param_grid = {
    #     "kernel": [1.0 * RBF(), ConstantKernel(), Matern()],
    #     "n_restarts_optimizer": np.arange(1, 6),
    # }

    #metrics.get_scorer_names()
    random_cv =  GridSearchCV(
        model, param_grid, cv=5, scoring="r2", n_jobs=-1,
    )
    random_cv.fit(X_train, y_train)

    #regressor = RandomForestRegressor(**random_cv.best_params_)
    #regressor.fit(X_train, y_train)

    y_hat_train = random_cv.predict(X_train)  # Training set predictions
    y_hat_test = random_cv.predict(X_test)  # Test set predictions
    R2_train = r2(y_hat_train, y_train)
    R2_test = r2(y_hat_test, y_test)
    MSE_train = mean_squared_error(y_hat_train, y_train)
    MSE_test = mean_squared_error(y_hat_test, y_test)
    MAE_train = mean_absolute_error(y_hat_train, y_train)
    MAE_test = mean_absolute_error(y_hat_test, y_test)
    RMSE_train = np.sqrt(MSE_train)
    RMSE_test = np.sqrt(MSE_test)
    #print(R2_train, R2_test)
    return random_cv, y_hat_train, y_hat_test, R2_train, R2_test, MSE_train, MSE_test, MAE_train, MAE_test,  RMSE_test, RMSE_train
matplotlib.rcParams['font.family']='Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size']=12
matplotlib.rcParams['mathtext.fontset'] ='custom'

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import StandardScaler

def y_inverse_transform(y_list, std_scalery):
    #y_orignal = np.exp(std_scalery.inverse_transform(y_list.reshape(-1,1)))
    y_orignal = std_scalery.inverse_transform(y_list.reshape(-1,1))
    return y_orignal
def data_preprocessing(X, y):
    std_scalerXc = StandardScaler()
    std_scaleryc = StandardScaler()
    Xc = np.array(X.values)
    yc = np.array(y.values)
    Xc_norm = std_scalerXc.fit_transform(Xc)
    yc_norm = std_scaleryc.fit_transform(yc.reshape(-1, 1))
    X_train = Xc_norm[:65, :34]

    y_train = yc_norm[:65, 0].ravel()

    X_test = Xc_norm[65:, :34]

    y_test = yc_norm[65:, 0].ravel()

    return X_train, X_test, y_train, y_test, std_scalerXc, std_scaleryc
def plot(y_hat_train, y_hat_test, y_train, y_test, std_scaleryc, title):
    y_hat_train = y_inverse_transform(y_hat_train, std_scaleryc)
    y_hat_test = y_inverse_transform(y_hat_test, std_scaleryc)

    y_train = y_inverse_transform(y_train, std_scaleryc)
    y_test = y_inverse_transform(y_test, std_scaleryc)
    matplotlib.rcParams['font.size']=20
    x = [np.min(y_train), np.max(y_train)]
    y = [np.min(y_train), np.max(y_train)]

    R2_train = r2(y_hat_train, y_train)
    R2_test = r2(y_hat_test, y_test)
    MSE_train = mean_squared_error(y_hat_train, y_train)
    MSE_test = mean_squared_error(y_hat_test, y_test)
    MAE_train = mean_absolute_error(y_hat_train, y_train)
    MAE_test = mean_absolute_error(y_hat_test, y_test)
    RMSE_train = np.sqrt(MSE_train)
    RMSE_test = np.sqrt(MSE_test)
    print(R2_train, R2_test, MSE_train, MSE_test, MAE_train, MAE_test, RMSE_test, RMSE_train)
    plt.rcParams['axes.linewidth']=3
    plt.figure(figsize=(8, 7), dpi=150)#7.4
    plt.scatter(y_train, y_hat_train, label="Training",marker='*',
                c="blue", alpha=0.7,s=60)
    plt.scatter(y_test, y_hat_test, label="Test",
                c="green", s=60, alpha=0.9,marker='*')
    x = [-3, 4]
    y = [-3, 4]
    plt.plot(x, y, c='red',linestyle = '--')
    plt.text(3.5, -1.5, f'R² = {R2_test:.2f}', fontsize=20, ha='right' )
    plt.text(3.5, -2, f'MSE = {MSE_test:.2f}', fontsize=20, ha='right')
    plt.text(3.5, -2.5, f'MAE = {MAE_test:.2f}', fontsize=20, ha='right')
    plt.xlabel('lg(Measured creep rupture life)(h)', fontproperties='Arial', fontsize=24)
    plt.ylabel('lg(Predicted creep rupture life)(h)', fontproperties='Arial', fontsize=24)
    plt.xlim(-3, 4)
    plt.ylim(-3, 4)
    plt.yticks(np.arange(-3,5,step=1), fontproperties='Arial', size=24)  # 设置大小及加粗
    plt.xticks(np.arange(-3,5,step=1), fontproperties='Arial', size=24)
    plt.legend(loc='best', frameon=False,fontsize=25,handletextpad=0)
    plt.tick_params(length=7, width=2)
    plt.title(title)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    return plt.show()
if __name__ == '__main__':
    data_pd = pd.read_csv("C:/Users/zbt/Desktop/Ti-alloy/train_test_Data/temp650/stress320/stress320.csv")
    data_pd = data_pd.drop(['class'],axis=1)
    X = data_pd.iloc[:, :34]
    print(X.shape)
    y = data_pd.iloc[:, 34]
    print(y.shape)
    X_train, X_test, y_train, y_test, std_scalerXc, std_scaleryc = data_preprocessing(X, y)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    best_model, y_hat_train, y_hat_test, R2_train, R2_test, MSE_train, MSE_test, MAE_trian, MAE_test, RMSE_test, RMSE_train = build_model_re(
        X_train, X_test, y_train, y_test, model='XGB')
    plot(y_hat_train, y_hat_test, y_train, y_test, std_scaleryc, title='XGBoost')
