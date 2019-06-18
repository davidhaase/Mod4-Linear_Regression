import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression
from sklearn.feature_selection import RFECV

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

class LinearModeler():

    def __init__(self, data_file=''):
        self.df = None
        self.target = None
        self.features = None
        self.data_file = data_file
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scale = True
        self.model = None
        self.y_train_pred = None
        self.train_mae = None
        self.train_mse = None
        self.train_rmse = None
        self.y_pred = None
        self.test_mae = None
        self.test_rmse = None
        self.target_std = None
        self.results = {}
        self.history = []

    def read_csv(self, data_file=''):
        if data_file != '':
            self.data_file = data_file
        try:
            self.df = pd.read_csv(self.data_file,index_col='id')
            return self.df.shape

        except Exception as e:
            print(e)

    def explore_data(self):
        self.df.shape

        sns.set(style="white")

        # Compute the correlation matrix
        corr = self.df.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

    def plot_pairs(self, cols=[]):
        if cols == []:
            cols = self.df.columns

        sns.pairplot(self.df, vars=cols)

    def missing_data(self):
        #missing data
        total = self.df.isnull().sum().sort_values(ascending=False)
        percent = (self.df.isnull().sum()/self.df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data.head(20)

    def set_target_and_features(self, target, features):
        self.target = self.df[target]
        self.features = self.df[features]
        self.base_features = features

    def get_drop_list_corrs(self, threshold=0.95):

        # Create correlation matrix
        # Select upper triangle of correlation matrix
        corr_matrix = self.X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        return [column for column in upper.columns if any(upper[column] > threshold)]

    def get_drop_list_f_test(self, k=10):
        selector = SelectKBest(f_regression, k)
        selector.fit(self.X_train, self.y_train)
        return self.X_train.columns[~selector.get_support()]

    def get_drop_list_rfecv(self):
        ols = linear_model.LinearRegression()
        # Create recursive feature eliminator that scores features by mean squared errors
        selector = RFECV(estimator=ols, step=1, cv=20, scoring='neg_mean_squared_error')

        # Fit recursive feature eliminator
        selector.fit(self.X_train, self.y_train)
        return list(self.X_train.columns[~selector.support_])


    def get_drop_list_f_test(self, k=10):
        selector = SelectKBest(f_regression, k)
        selector.fit(self.X_train, self.y_train)
        return list(self.X_train.columns[~selector.get_support()])

    def split_data(self, i=34, j=0.2):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target, random_state=i, test_size=j)
        except Exception as e:
            print(e)

        if (self.scale):
            scaler = StandardScaler()
            scaler.fit(self.X_train)
            self.X_train =pd.DataFrame(data=scaler.transform(self.X_train), columns=self.features.columns)
            self.X_test =pd.DataFrame(data=scaler.transform(self.X_test), columns=self.features.columns)


    def evaluate_lm(self, model_type='Normal'):
        #instantiate a linear regression object
        self.model = linear_model.LinearRegression()

        #fit the linear regression to the data
        self.model = self.model.fit(self.X_train, self.y_train)

        self.get_metrics(model_type)


    def get_metrics(self, model_type):

        attempt = {}
        print("Training set - Features: ", self.X_train.shape, "Target: ", self.y_train.shape)
        print("Test set - Features: ", self.X_test.shape, "Target: ",self.y_test.shape)




        print('\nTRAINING STATS')
        print('Training Intercept: {}'.format(self.model.intercept_))
        print('Training Coefficients:\n{}'.format(self.model.coef_))
        print ("Training R^2 Score:", self.model.score(self.X_train, self.y_train))

        self.y_train_pred = self.model.predict(self.X_train)
        self.train_mae = metrics.mean_absolute_error(self.y_train, self.y_train_pred)
        self.train_mse = metrics.mean_squared_error(self.y_train, self.y_train_pred)
        self.train_rmse = np.sqrt(metrics.mean_squared_error(self.y_train, self.y_train_pred))


        print('****\nTRAINING ERRORS')
        print('Mean Absolute Error:', self.train_mae )
        print('Mean Squared Error:',  self.train_mse)
        print('Root Mean Squared Error:' , self.train_rmse)

        self.target_std = self.target.std()

        print('\nBy Deviation of Target\n')
        print('Mean Absolute Error (Z):', self.train_mae/self.target_std)
        print('Root Mean Squared Error (Z):' , self.train_rmse/self.target_std)

        training = {'result_type':'Training Evaluation',
                   'Training Intercept' : self.model.intercept_,
                   'Training Coefficients' : self.model.coef_,
                   'Training R^2 Score' : self.model.score(self.X_train, self.y_train),
                   'Mean Absolute Error' : self.train_mae,
                   'Mean Squared Error' : self.train_mse,
                   'Root Mean Squared Error' : self.train_rmse,
                   'Mean Absolute Error' : self.train_mae/self.target_std,
                   'Root Mean Squared Error' : self.train_rmse/self.target_std}

        attempt['Training'] = training
        self.y_pred = self.model.predict(self.X_test)

        plt.scatter(self.y_test, self.y_pred)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        sns.residplot(self.y_pred, self.y_test, lowess=True, color="g")

        print('****\nTESTING ERRORS')
        print ("Test R^2 Score:", self.model.score(self.X_test, self.y_test))

        self.test_mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
        self.test_mse = metrics.mean_squared_error(self.y_test, self.y_pred)
        self.test_rmse = np.sqrt(metrics.mean_squared_error(self.y_test, self.y_pred))


        print('Mean Absolute Error:' + str(self.test_mae))
        print('Mean Squared Error:' + str(self.test_mse))
        print('Root Mean Squared Error:' + str(self.test_rmse))

        print('\nBy Deviation of Target\n')
        print('Mean Absolute Error (Z):', self.test_mae/self.target_std )
        print('Root Mean Squared Error (Z):' , self.test_rmse/self.target_std)

        print('\n***\nRMSE\nTraining: ', self.train_rmse, "vs. Testing: ", self.test_rmse)

        prediction = {'Test R^2 Score' : self.model.score(self.X_test, self.y_test),
                   'Mean Absolute Error' : self.test_mae,
                   'Mean Squared Error' : self.test_mse,
                   'Root Mean Squared Error' : self.test_rmse,
                   'Mean Absolute Error  Z' : self.test_mae/self.target_std,
                   'Root Mean Squared Error Z' : self.test_rmse/self.target_std}

        attempt['Prediction'] = prediction
        attempt['RMSE Comparison'] = [self.train_rmse, self.test_rmse]
        attempt['Model Type'] = model_type
        self.history.append(attempt)

        lm_coef_ = attempt['Training']['Training Coefficients']
        try:
            coef = pd.DataFrame(data=lm_coef_).T
            coef.columns = self.features.columns

            model_coef = coef.T.sort_values(by=0).T
            model_coef.plot(kind='bar', title='Modal Coefficients', legend=True, figsize=(16,8))
        except Exception as e:
            print(e)

    def set_polynomial(self, degree):
        #instantiate this class
        poly = PolynomialFeatures(degree, interaction_only=False)
        df_poly = pd.DataFrame(poly.fit_transform(self.features), columns=poly.get_feature_names(self.features.columns))
        self.features = df_poly
        self.split_data()

        self.evaluate_lm('Poly_' + str(degree))


    def evaluate_ridge(self, alpha):
        ridgeReg = Ridge(alpha, normalize=True)
        self.model = ridgeReg.fit(self.X_train, self.y_train)
        self.y_pred = ridgeReg.predict(self.X_test)

        self.get_metrics('Ridge_' + str(alpha))

    def evaluate_lasso(self, alpha):
        lassoReg = Lasso(alpha, normalize=True)
        self.model = lassoReg.fit(self.X_train, self.y_train)
        self.y_pred = lassoReg.predict(self.X_test)

        self.get_metrics('Lasso_' + str(alpha))

    def plot_rmse_history(self):

        errors = [attempt['RMSE Comparison'] for attempt in self.history]
        x_ticks = [attempt['Model Type'] for attempt in self.history]
        df_error = pd.DataFrame(errors, index=x_ticks, columns=['train_error', 'test_error'])


        df_error.plot.line(figsize=(16,8))
        plt.xticks(np.arange(len(x_ticks)), (x_ticks))


    def plot_mae_history(self):
        errors = []
        x_ticks = []

        for attempt in self.history:
            train_mae = attempt['Training']['Mean Absolute Error']
            test_mae = attempt['Prediction']['Mean Absolute Error']
            x_ticks.append(attempt['Model Type'])
            errors.append([train_mae, test_mae])

        df_error = pd.DataFrame(errors, index=x_ticks, columns=['train_error', 'test_error'])

        df_error.plot.line(figsize=(16,8))
        plt.xticks(np.arange(len(x_ticks)), (x_ticks))

    def find_lowest(self, category):
        min_attempt = None
        min_value = 300.00

        for attempt in self.history:
            test_value = attempt['Prediction'][category]
            if test_value < min_value:
                min_value = test_value
                min_attempt = attempt

        return min_attempt

    def find_highest(self, category='Test R^2 Score'):
        max_attempt = None
        max_value = 0.00

        for attempt in self.history:
            test_value = attempt['Prediction'][category]
            if test_value > max_value:
                max_value = test_value
                max_attempt = attempt

        return max_attempt
