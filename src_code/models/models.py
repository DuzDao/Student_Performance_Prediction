from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
import lightgbm
import xgboost
from sklearn.metrics import r2_score as r2, mean_squared_error as mse
import numpy as np

class MyEnsembleModel:
    def __init__(self, X, y):
        self.lgm = lightgbm.LGBMRegressor()
        self.dtr = DecisionTreeRegressor()
        self.gbr = GradientBoostingRegressor()
        self.rfr = RandomForestRegressor()
        self.xg = xgboost.XGBRegressor()
        self.X = X
        self.y = y
    
    def evaluate(self, yVal, pred):
        return r2(yVal, pred), mse(yVal, pred)

    def lgbm_dtr(self):
        X_train, X_val, y_train, y_val = self.X[:800], self.X[800:], self.y[:800], self.y[800:]
        self.dtr.fit(X_train, y_train)
        dtr_pred = self.dtr.predict(X_val)
        dtr_r2, dtr_mse = self.evaluate(y_val, dtr_pred)

        self.lgm.fit(X_train, y_train)
        lgbm_pred = self.lgm.predict(X_val)
        lgbm_r2, lgbm_mse = self.evaluate(y_val, lgbm_pred)

        stacked_X_val = np.column_stack((dtr_pred, lgbm_pred))
        stacked_model = LinearRegression()
        stacked_model.fit(stacked_X_val, y_val)
        stacked_pred = stacked_model.predict(stacked_X_val)
        pred_r2, pred_mse = self.evaluate(y_val, stacked_pred)

        return dtr_r2, dtr_mse, lgbm_r2, lgbm_mse, pred_r2, pred_mse

    def gbr_rfr(self):
        kf = KFold(n_splits=5, shuffle=True)
        pred_scores = []

        for _, (train_index, val_index) in enumerate(kf.split(self.X, self.y)):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]

            self.gbr.fit(X_train, y_train)
            self.rfr.fit(X_train, y_train)

            ensemble_pred = (self.gbr.predict(X_val) + self.rfr.predict(X_val)) / 2

            score = r2(y_val, ensemble_pred)
            pred_scores.append(score)
        return pred_scores, np.mean(pred_scores)

    def lgbm_xg(self):
        kf = KFold(n_splits=5, shuffle=True)
        pred_scores = []

        for _, (train_index, val_index) in enumerate(kf.split(self.X, self.y)):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]
        
            self.xg.fit(X_train, y_train)
            self.lgm.fit(X_train, y_train)
        
            ensemble_pred = (self.xg.predict(X_val) + self.lgm.predict(X_val)) / 2
        
            score = r2(y_val, ensemble_pred)
            pred_scores.append(score)

        return pred_scores, np.mean(pred_scores)
