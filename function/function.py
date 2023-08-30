#!pip install scorecardpy
#!pip install catboost
#!pip install pygam


from google.colab import files
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import missingno as msno
from sklearn.model_selection import train_test_split
import scorecardpy as sc
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import time
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from pygam import LogisticGAM 
from typing import List, Dict, Any, Tuple

import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.offline as py

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc




def get_column_types(data_frame):
    """
    Get the names of qualitative (categorical) and quantitative (numerical) columns from a DataFrame.
    
    Parameters:
        data_frame (pd.DataFrame): The DataFrame for which column types need to be determined.
        
    Returns:
        qualitative_cols (list): A list containing the names of qualitative columns.
        quantitative_cols (list): A list containing the names of quantitative columns.
    """
    # Initialize empty lists to store column names
    qualitative_cols = []  # For qualitative (categorical) columns
    quantitative_cols = []  # For quantitative (numerical) columns

    # Loop through each column in the DataFrame
    for column in data_frame.columns:
        # Check if the column's data type is 'object'
        if data_frame[column].dtype == 'object':
            # If data type is 'object', consider it a qualitative column
            qualitative_cols.append(column)
        else:
            # If data type is not 'object', consider it a quantitative column
            quantitative_cols.append(column)

    # Return the lists of column names
    return qualitative_cols, quantitative_cols




def custom_boxplot(df, column_list):
    """
    Create customized boxplots for specified columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_list (list): List of column names for which boxplots will be generated.

    Returns:
        None
    """
    for col in column_list:
        plt.boxplot(x=df[col],
                    whis=3,
                    patch_artist=True,
                    showmeans=True,
                    boxprops={'color': 'black', 'facecolor': '#9999ff'},
                    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
                    meanprops={'marker': 'D', 'markerfacecolor': 'indianred'},
                    medianprops={'linestyle': '--', 'linewidth': 2.5, 'color': 'orange'})
        plt.tick_params(top='off', right='off')
        plt.xlabel("" + col)
        plt.title("BoxPlot on: " + col)
        plt.show()


def plot_density_with_normal(df):
    """
    Plot density plots with overlaid normal distribution curves for each column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    sns.set(style="ticks")

    for column in df.columns:
        # Get cleaned data of the column
        cleaned_data = df[column]

        # Exclude non-finite values from the calculation
        cleaned_data = cleaned_data[np.isfinite(cleaned_data)]

        # Calculate parameters of the normal distribution (mean and standard deviation)
        mean, std = norm.fit(cleaned_data)

        # Generate a sample of values for the probability density function of the normal distribution
        x = np.linspace(cleaned_data.min(), cleaned_data.max(), 100)
        y = norm.pdf(x, mean, std)

        # Plot the density plot with the normal distribution
        plt.figure()
        sns.kdeplot(data=cleaned_data, label="Density Plot")
        plt.plot(x, y, 'r-', label="Normal Distribution")
        plt.xlabel(column)
        plt.ylabel("Density")
        plt.legend()
        plt.title('Density Plot for ' + column)
        plt.show()



def custom_pieplot(df, column_list):
    """
    Create customized pie plots for specified qualitative columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_list (list): List of column names for which pie plots will be generated.

    Returns:
        None
    """
    for col in column_list:
        value_counts = df[col].value_counts()
        labels = value_counts.index
        sizes = value_counts.values

        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Pie Chart for: " + col)
        plt.show()




def analyze_missing_values(df):
    """
    Analyze missing values in a DataFrame and categorize variables based on missing value percentages.

    Parameters:
    df (DataFrame): The input DataFrame to analyze.

    Returns:
    tuple: A tuple containing lists of variable names for different missing value categories:
        - no_missing (list): Variables with no missing values.
        - less_than_5 (list): Variables with missing values less than or equal to 5%.
        - between_5_and_25 (list): Variables with missing values between 5% and 25%.
        - over_25 (list): Variables with missing values over 25%.
    Example:
        no_missing, less_than_5, between_5_and_25, over_25 = analyze_missing_values(df_concat)
        print("Variables with no missing values:", no_missing)
        print("Variables with missing values less than or equal to 5%:", less_than_5)
        print("Variables with missing values between 5% and 25%:", between_5_and_25)
        print("Variables with missing values over 25%:", over_25)
    """

    # Calculate the percentage of missing values for each column
    missing_percent = (df.isnull().sum() / len(df)) * 100

    # Categorize variables based on missing value percentages
    less_than_5 = missing_percent[missing_percent <= 5].index.tolist()
    between_5_and_25 = missing_percent[(missing_percent > 5) & (missing_percent <= 25)].index.tolist()
    over_25 = missing_percent[missing_percent > 25].index.tolist()

    # Variables with no missing values
    no_missing = missing_percent[missing_percent == 0].index.tolist()

    # Visualize missing values
    msno.matrix(df, sparkline=False, figsize=(10, 5), fontsize=8, color=(0.27, 0.52, 1.0))
    plt.show()

    sns.set(style="white")
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cmap="RdYlGn", cbar=False)
    plt.show()

    msno.dendrogram(df, figsize=(10, 5), fontsize=8)
    plt.show()

    return no_missing, less_than_5, between_5_and_25, over_25





def detecter_corriger_valeurs_manquantes(data, strategy='median'):
    """
    Detect and correct missing values in the DataFrame.

    Parameters:
        data (DataFrame): Input data.
        strategy (str): Strategy for replacing missing values. Options: 'mean', 'median', 'mode', 'variance'.

    Returns:
        data_corrigee (DataFrame): Corrected data.
    """
    data_corrigee = data.copy()

    quantitative_vars = data.select_dtypes(include=['number']).columns
    qualitative_vars = data.select_dtypes(include=['object']).columns

    # Detect and correct missing values based on strategy
    for column in data.columns:
        missing_percentage = data[column].isnull().mean() * 100

        if missing_percentage > 0 and missing_percentage <= 5:
            data_corrigee.dropna(subset=[column], inplace=True)
        elif missing_percentage > 5 and missing_percentage < 25:
            if column in qualitative_vars:
                if strategy == 'mode':
                    replacement_value = data[column].mode().iloc[0]
                else:
                    raise ValueError(f"Invalid strategy '{strategy}' for qualitative variable.")
            elif column in quantitative_vars:
                if strategy == 'mean':
                    replacement_value = data[column].mean()
                elif strategy == 'median':
                    replacement_value = data[column].median()
                elif strategy == 'variance':
                    replacement_value = data[column].var()
                else:
                    raise ValueError(f"Invalid strategy '{strategy}' for quantitative variable.")
            else:
                raise ValueError("Unknown variable type.")
                
            data_corrigee[column].fillna(replacement_value, inplace=True)
        elif missing_percentage >= 25:
            data_corrigee.drop(columns=[column], inplace=True)

    return data_corrigee 




def split_data_to_train_test(data, ratio=0.8):
    """
    Split a DataFrame into training and test datasets.

    Parameters:
        data (DataFrame): Input data to be split.
        ratio (float): The proportion of data to be assigned to the training dataset.
                      Default is 0.8 (80% training, 20% test).

    Returns:
        train_data (DataFrame): Training dataset.
        test_data (DataFrame): Test dataset.
    """
    train_data, test_data = train_test_split(data, train_size=ratio, test_size=1 - ratio, random_state=42)
    return train_data, test_data


def feature_selection(X, y):
    """
    Perform feature selection using various methods.

    Parameters:
        X (DataFrame): Independent variables.
        y (Series): Categorical target variable.

    Returns:
        feature_selection_df (DataFrame): DataFrame indicating feature selection results.
    """
    # Convert categorical y into numerical values
    #y_numerical = y.map({'good': 0, 'bad': 1})


    mapping_target = {'good':0, 'bad':1}
    # Use the map function to replace values in the 'target' column
    y_numerical = y['target'].map(mapping_target)



    cor_list = []

    # Pearson Correlation
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y_numerical)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-100:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in X.columns]

    # Chi-2
    k= len(X.columns)
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=k)
    chi_selector.fit(X_norm, y_numerical)
    chi_support = chi_selector.get_support()

    # Wrapper (RFE & RFEcv)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=20, step=10, verbose=5)
    rfe_selector.fit(X_norm, y_numerical)
    rfe_support = rfe_selector.get_support()

    # Embedded (L1)
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"))
    embeded_lr_selector.fit(X_norm, y_numerical)
    embeded_lr_support = embeded_lr_selector.get_support()

    # Random Forest
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=17))
    embeded_rf_selector.fit(X, y_numerical)
    embeded_rf_support = embeded_rf_selector.get_support()

    # LightGBM
    lgbc = LGBMClassifier(n_estimators=18, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    embeded_lgb_selector = SelectFromModel(lgbc)
    embeded_lgb_selector.fit(X, y_numerical)
    embeded_lgb_support = embeded_lgb_selector.get_support()

    # Summary
    feature_selection_df = pd.DataFrame({'Feature': X.columns,
                                         'Pearson': cor_support,
                                         'Chi-2': chi_support,
                                         'RFE': rfe_support,
                                         'Logistics': embeded_lr_support,
                                         'Random Forest': embeded_rf_support,
                                         'LightGBM': embeded_lgb_support})

    feature_selection_df['Total'] = np.sum(feature_selection_df.iloc[:, 1:], axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)

    return feature_selection_df



def select_features_by_vote(feature_selection_df, vote):
    """
    Select features based on a vote threshold.

    Parameters:
        feature_selection_df (DataFrame): DataFrame indicating feature selection results.
        vote (int): The vote threshold.

    Returns:
        selected_features (list): List of selected features.
    """
    selected_features = feature_selection_df.loc[feature_selection_df['Total'] >= vote, 'Feature'].tolist()
    #selected_features.append('target')
    return selected_features




def resample_data(sampling_type: str, X_train: BaseEstimator, y_train: BaseEstimator) -> tuple:
    """
    Resamples the training data based on the specified sampling type.

    Parameters:
    sampling_type (str): The resampling technique to use ('SMOTE', 'Random', or 'ADASYN').
    X_train (BaseEstimator): The feature matrix of the training data.
    y_train (BaseEstimator): The target vector of the training data.

    Returns:
    tuple: A tuple containing the resampled feature matrix (X_train_sampling) and the resampled target vector (y_train_sampling).
    """

    allowed_sampling_types = ['SMOTE', 'Random', 'ADASYN']
    if sampling_type not in allowed_sampling_types:
        raise ValueError("Invalid sampling type. Allowed values are: 'SMOTE', 'Random', 'ADASYN'.")

    X_train, y_train = check_X_y(X_train, y_train)
    
    if sampling_type == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif sampling_type == 'Random':
        sampler = RandomUnderSampler(random_state=0)
    elif sampling_type == 'ADASYN':
        sampler = ADASYN()

    X_train_sampling, y_train_sampling = sampler.fit_resample(X_train, y_train)
    return X_train_sampling, y_train_sampling



def train_and_save_models(model_types: List[str], X_train: BaseEstimator, X_test: BaseEstimator,
                          y_train: BaseEstimator, sampling_type: str, n_est: int = 100, database_name: str = '') -> Dict[str, Tuple[Any, float]]:
    """
    Trains multiple models on resampled data and saves the trained models.

    Parameters:
    model_types (List[str]): List of model types to train ('Logistic', 'RandomForest', etc.).
    X_train (BaseEstimator): The feature matrix of the training data.
    X_test (BaseEstimator): The feature matrix of the test data.
    y_train (BaseEstimator): The target vector of the training data.
    sampling_type (str): The resampling technique to use ('SMOTE', 'Random', or 'ADASYN').
    n_est (int): Number of estimators for RandomForest and GradientBoosting models (default=100).

    Returns:
    Dict[str, Tuple]: A dictionary containing trained models and their respective training times.
    """
    allowed_model_types = ['Logistic', 'RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting', 'CatBoost', 'gam']
    invalid_model_types = [model for model in model_types if model not in allowed_model_types]
    if invalid_model_types:
        raise ValueError(f"Invalid model type(s): {', '.join(invalid_model_types)}. Allowed values are: {', '.join(allowed_model_types)}")

    results = {}

    for model_type in model_types:
        start_time = time.time()

        X_train_sampling, y_train_sampling = resample_data(sampling_type, X_train, y_train)
        
        if model_type == 'Logistic':
            model = LogisticRegression()
        elif model_type == 'RandomForest':
            model = RandomForestClassifier(n_estimators=n_est)
        elif model_type == 'XGBoost':
            model = XGBClassifier()
        elif model_type == 'LightGBM':
            model = lgb.LGBMClassifier()
        elif model_type == 'GradientBoosting':
            model = GradientBoostingClassifier(n_estimators=n_est)
        elif model_type == 'CatBoost':
            model = CatBoostClassifier()
        elif model_type == 'gam':
            model = LogisticGAM()

        model_fit = model.fit(X_train_sampling, y_train_sampling)
        t_training = time.time() - start_time
        
        model_filename = f'm_{model_type.lower()}_{database_name.lower()}.sav'
        pickle.dump(model_fit, open(model_filename, 'wb'))



        results[model_type] = (model_fit, t_training)

    return results


def fit_multiple_algorithms(model_types: List[str], X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, sampling_type: str, n_est: int = 100, database_name: str = '') -> Dict[str, Any]:
    """
    Fits and saves multiple machine learning algorithms.

    Parameters:
    model_types (List[str]): List of model types to train ('Logistic', 'RandomForest', etc.).
    X_train (pd.DataFrame): The feature matrix of the training data.
    X_test (pd.DataFrame): The feature matrix of the test data.
    y_train (pd.Series): The target vector of the training data.
    sampling_type (str): The resampling technique to use ('SMOTE', 'Random', or 'ADASYN').
    n_est (int): Number of estimators for RandomForest and GradientBoosting models (default=100).

    Returns:
    Dict[str, Any]: A dictionary containing trained models.
    """
    trained_models = {}

    for model_type in model_types:
        trained_model = train_and_save_models([model_type], X_train, X_test, y_train, sampling_type, n_est, database_name)
        trained_models[model_type] = trained_model[model_type][0]

    return trained_models



def load_model(model_type: str, database_name: str = '') -> BaseEstimator:
    """
    Load a trained model from a saved file.

    Parameters:
    model_type (str): The type of model to load ('Logistic', 'RandomForest', etc.).

    Returns:
    BaseEstimator: The loaded trained model.
    """
    model_filename = f'm_{model_type.lower()}_{database_name.lower()}.sav'
    loaded_model = pickle.load(open(model_filename, 'rb'))
    return loaded_model



def predict_probabilities(models: Dict[str, BaseEstimator], X_test: BaseEstimator) -> Dict[str, List[float]]:
    """
    Predict the probabilities of the positive class using multiple trained models.

    Parameters:
    models (Dict[str, BaseEstimator]): A dictionary containing model types as keys and loaded trained models as values.
    X_test (BaseEstimator): The feature matrix of the test data.

    Returns:
    Dict[str, List[float]]: A dictionary containing model types as keys and lists of predicted probabilities as values.
    """
    predicted_probabilities = {}
    
    for model_type, model in models.items():
        positive_class_probabilities = model.predict_proba(X_test)[:, 1]
        predicted_probabilities[model_type] = positive_class_probabilities.tolist()

    df = pd.DataFrame(predicted_probabilities)
    
    return df



def calculate_gains_table(df: pd.DataFrame, model_col_list: list, target_col: str) -> pd.DataFrame:
    """
    Calculate summary statistics for deciles based on predicted probabilities.

    This function calculates various summary statistics for each decile based on the predicted probabilities.

    Parameters:
    df (pd.DataFrame): DataFrame containing predicted values, target column, and decile.
    model_col_list (list): List of column names containing the predicted probabilities for each model.
    target_col (str): Name of the column containing the target values.

    Returns:
    pd.DataFrame: DataFrame containing summary statistics for each decile.

    Input:
    - df: A DataFrame with columns for predicted probabilities, target values, and decile.
    - model_col_list: List of column names containing the predicted probabilities for each model.
    - target_col: Name of the column containing the target values.

    Output:
    - A DataFrame containing the following columns:
        - 'model': Name of the model.
        - 'OBS': Number of records in each decile.
        - 'MIN_SCORE': Minimum predicted probability in each decile.
        - 'MAX_SCORE': Maximum predicted probability in each decile.
        - 'FAILURES': Number of failures in each decile.
        - 'FAILURE_RATE': Actual failure rate for each decile.
        - 'PCT_OF_FAILURES': Percentage of failures in each decile.
        - 'CUML_FAILURES': Cumulative number of failures by decile.
        - 'CUML_PCT_OF_FAILURES': Cumulative percentage of failures by decile.
    Example usage :
     model_col_list = ['Logistic', 'RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting', 'CatBoost']
     gains_summary = calculate_gains_table(df, model_col_list, 'target')
    """
    def create_deciles(df: pd.DataFrame, model_col: str, target_col: str) -> pd.DataFrame:
        """
        Create deciles based on the predicted probabilities of a model.

        This function sorts the data by the predicted probabilities in descending order,
        adds a small random number to break ties, and then creates deciles based on the sorted probabilities.

        Parameters:
        df (pd.DataFrame): DataFrame containing predicted values and target column.
        model_col (str): Name of the column containing the predicted probabilities.
        target_col (str): Name of the column containing the target values.

        Input:
        - df: A DataFrame with columns for predicted probabilities and the target values.
        - model_col: Name of the column containing the predicted probabilities.
        - target_col: Name of the column containing the target values.

        Returns:
        pd.DataFrame: DataFrame with added columns 'wookie', 'DECILE' based on the model's predicted probabilities.
        """
        df_sorted = df.sort_values(by=[model_col], ascending=[False])
        df_sorted['wookie'] = np.random.randint(0, 100, df_sorted.shape[0]) / 100000000000000000
        df_sorted[model_col] = df_sorted[model_col] + df_sorted['wookie']
        df_sorted['DECILE'] = pd.qcut(df_sorted[model_col], 10, labels=np.arange(100, 0, -10))
        return df_sorted

    gains_tables = []

    for model_col in model_col_list:
        df_with_deciles = create_deciles(df, model_col, target_col)

        gains_table = df_with_deciles.groupby(['DECILE']).agg(
            MIN_SCORE=(model_col, 'min'),
            MAX_SCORE=(model_col, 'max'),
            FAILURE_RATE=(target_col, 'mean'),
            FAILURES=(target_col, 'sum'),
            OBS=(target_col, 'count')
        ).sort_values(by=['DECILE'], ascending=[False])

        gains_table['CUML_FAILURES'] = gains_table['FAILURES'].cumsum()
        gains_table['PCT_OF_FAILURES'] = (gains_table['FAILURES']) / (df[target_col].sum()) * 100
        gains_table['CUML_PCT_OF_FAILURES'] = gains_table['PCT_OF_FAILURES'].cumsum()

        gains_table['model'] = model_col
        gains_table = gains_table[['model', 'OBS', 'MIN_SCORE', 'MAX_SCORE', 'FAILURES', 'FAILURE_RATE', 'PCT_OF_FAILURES', 'CUML_FAILURES', 'CUML_PCT_OF_FAILURES']]
        gains_tables.append(gains_table)

    all_gains_tables = pd.concat(gains_tables, ignore_index=True)
    
    return all_gains_tables





def calculate_cutoff_metrics_gain(df: pd.DataFrame, model_cols: list, target_col: str, cut: int) -> pd.DataFrame:
    """
    Calculate cutoff metrics for different probability cut-off points for multiple models.

    This function calculates various metrics for different models, probability cut-off points, and target values.

    Parameters:
    df (pd.DataFrame): DataFrame containing predicted probabilities and target values for multiple models.
    model_cols (list): List of column names containing the predicted probabilities for each model.
    target_col (str): Name of the column containing the target values.
    cut (int): Number of cut-off points for qcut.

    Returns:
    pd.DataFrame: DataFrame containing cutoff metrics for different models and probability cut-off points.
    """
    results = []

    for model_col in model_cols:
        model_df = df[[model_col, target_col]].copy()
        model_df[model_col] += np.random.randint(0, 100, model_df.shape[0]) / 100000000000000000
        model_df['GROUPS'] = pd.qcut(model_df[model_col], cut, labels=False, duplicates='drop')

        tips_summedb = model_df.groupby(['GROUPS'])[model_col].min()
        tips_summedz = model_df.groupby(['GROUPS'])[target_col].sum()
        tips_summeda = model_df.groupby(['GROUPS'])[target_col].count()

        tips = pd.concat([tips_summedb, tips_summedz, tips_summeda], axis=1)
        tips.columns = ['CUT-OFF', 'FAILURES', 'OBS']
        tips['NON_FAILURES'] = tips.OBS - tips.FAILURES

        tips.reset_index(level=0, inplace=True)
        tips = tips.sort_values(by=['GROUPS'], ascending=[False])

        tips['INV_CUM_FAILURES'] = tips.FAILURES.cumsum()
        tips['INV_CUM_NON_FAILURES'] = tips.NON_FAILURES.cumsum()
        tips['TOTAL_OBS'] = tips.OBS.sum()

        tips = tips.sort_values(by=['GROUPS'], ascending=[True])

        tips['CUM_FAILURES'] = tips.FAILURES.cumsum()
        tips['CUM_NON_FAILURES'] = tips.NON_FAILURES.cumsum()
        tips['TOTAL_FAILURES'] = tips.FAILURES.sum()
        tips['TOTAL_NON_FAILURES'] = tips.NON_FAILURES.sum()

        tips['TRUE_POSITIVES'] = tips.INV_CUM_FAILURES
        tips['FALSE_POSITIVES'] = tips.INV_CUM_NON_FAILURES
        tips['TRUE_NEGATIVES'] = tips.CUM_NON_FAILURES - tips.NON_FAILURES
        tips['FALSE_NEGATIVES'] = tips.CUM_FAILURES - tips.FAILURES

        tips['OBS2'] = tips.TRUE_POSITIVES + tips.FALSE_POSITIVES + tips.TRUE_NEGATIVES + tips.FALSE_NEGATIVES

        tips['SENSITIVITY'] = tips['TRUE_POSITIVES'] / (tips['TRUE_POSITIVES'] + tips['FALSE_NEGATIVES'])
        tips['SPECIFICITY'] = tips['TRUE_NEGATIVES'] / (tips['FALSE_POSITIVES'] + tips['TRUE_NEGATIVES'])
        tips['FALSE_POSITIVE_RATE'] = 1 - tips['SPECIFICITY']
        tips['FALSE_NEGATIVE_RATE'] = 1 - tips['SENSITIVITY']

        gains = tips[['GROUPS', 'CUT-OFF', 'TRUE_POSITIVES', 'FALSE_POSITIVES', 'TRUE_NEGATIVES', 'FALSE_NEGATIVES',
                      'SENSITIVITY', 'SPECIFICITY', 'FALSE_POSITIVE_RATE', 'FALSE_NEGATIVE_RATE']]
        gains.insert(0, 'model', model_col)
        results.append(gains)

    result_df = pd.concat(results, ignore_index=True)
    return result_df


def calculate_cutoff_metricsMS(df: pd.DataFrame, model_cols: list, target_col: str, cut: int) -> pd.DataFrame:
    """
    Calculate cutoff metrics for different probability cut-off points for multiple models.

    This function calculates various metrics for different models, probability cut-off points, and target values.

    Parameters:
    df (pd.DataFrame): DataFrame containing predicted probabilities and target values for multiple models.
    model_cols (list): List of column names containing the predicted probabilities for each model.
    target_col (str): Name of the column containing the target values.
    cut (int): Number of cut-off points for qcut.

    Returns:
    pd.DataFrame: DataFrame containing cutoff metrics for different models and probability cut-off points.
    """
    results = []

    for model_col in model_cols:
        model_df = df[[model_col, target_col]].copy()
        model_df[model_col] += np.random.randint(0, 100, model_df.shape[0]) / 100000000000000000
        model_df['GROUPS'] = pd.qcut(model_df[model_col], cut, labels=False, duplicates='drop')

        tips_summedb = model_df.groupby(['GROUPS'])[model_col].min()
        tips_summedz = model_df.groupby(['GROUPS'])[target_col].sum()
        tips_summeda = model_df.groupby(['GROUPS'])[target_col].count()

        tips = pd.concat([tips_summedb, tips_summedz, tips_summeda], axis=1)
        tips.columns = ['CUT-OFF', 'FAILURES', 'OBS']
        tips['NON_FAILURES'] = tips.OBS - tips.FAILURES

        tips.reset_index(level=0, inplace=True)
        tips = tips.sort_values(by=['GROUPS'], ascending=[False])

        tips['INV_CUM_FAILURES'] = tips.FAILURES.cumsum()
        tips['INV_CUM_NON_FAILURES'] = tips.NON_FAILURES.cumsum()
        tips['TOTAL_OBS'] = tips.OBS.sum()

        tips = tips.sort_values(by=['GROUPS'], ascending=[True])

        tips['CUM_FAILURES'] = tips.FAILURES.cumsum()
        tips['CUM_NON_FAILURES'] = tips.NON_FAILURES.cumsum()
        tips['TOTAL_FAILURES'] = tips.FAILURES.sum()
        tips['TOTAL_NON_FAILURES'] = tips.NON_FAILURES.sum()

        tips['TRUE_POSITIVES'] = tips.INV_CUM_FAILURES
        tips['FALSE_POSITIVES'] = tips.INV_CUM_NON_FAILURES
        tips['TRUE_NEGATIVES'] = tips.CUM_NON_FAILURES - tips.NON_FAILURES
        tips['FALSE_NEGATIVES'] = tips.CUM_FAILURES - tips.FAILURES

        tips['OBS2'] = tips.TRUE_POSITIVES + tips.FALSE_POSITIVES + tips.TRUE_NEGATIVES + tips.FALSE_NEGATIVES

        tips['SENSITIVITY'] = tips['TRUE_POSITIVES'] / (tips['TRUE_POSITIVES'] + tips['FALSE_NEGATIVES'])
        tips['SPECIFICITY'] = tips['TRUE_NEGATIVES'] / (tips['FALSE_POSITIVES'] + tips['TRUE_NEGATIVES'])
        tips['FALSE_POSITIVE_RATE'] = 1 - tips['SPECIFICITY']
        tips['FALSE_NEGATIVE_RATE'] = 1 - tips['SENSITIVITY']

        tipsx = tips
        tipsx['FALSE_CLASSIFICATIONS'] = tipsx.FALSE_POSITIVES + tipsx.FALSE_NEGATIVES
        tipsx['FALSE_CLASSIFICATION_RATE'] = tipsx.FALSE_CLASSIFICATIONS / tipsx.TOTAL_OBS

        gains = tipsx[['GROUPS', 'CUT-OFF', 'TRUE_POSITIVES', 'FALSE_POSITIVES', 'TRUE_NEGATIVES', 'FALSE_NEGATIVES',
                      'SENSITIVITY', 'SPECIFICITY', 'FALSE_POSITIVE_RATE', 'FALSE_NEGATIVE_RATE',
                      'FALSE_CLASSIFICATIONS', 'FALSE_CLASSIFICATION_RATE']]
        gains.insert(0, 'model', model_col)
        results.append(gains)

    result_df = pd.concat(results, ignore_index=True)
    return result_df





def plot_misclassification_subplots(df: pd.DataFrame):
    unique_models = df['model'].unique()
    num_models = len(unique_models)

    # Calculate the number of rows and columns for subplots
    num_rows = (num_models + 1) // 2
    num_cols = 2
    
    fig = sp.make_subplots(rows=num_rows, cols=num_cols, subplot_titles=unique_models)

    for i, model in enumerate(unique_models, start=1):
        model_df = df[df['model'] == model]
        
        row = (i + 1) // 2
        col = (i % 2) + 1
        
        x1 = model_df['CUT-OFF']
        y1 = model_df['FALSE_POSITIVE_RATE']
        y2 = model_df['FALSE_NEGATIVE_RATE']
        y3 = model_df['FALSE_CLASSIFICATION_RATE']
        
        trace1 = go.Scatter(
            x=x1,
            y=y1,
            name='False Positive Rate'
        )
        trace2 = go.Scatter(
            x=x1,
            y=y2,
            name='False Negative Rate'
        )
        trace3 = go.Scatter(
            x=x1,
            y=y3,
            name='False Classification Rate'
        )
        
        for trace in [trace1, trace2, trace3]:
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        title='Mis-Classification Rates BY CUT OFF SCORE',
        showlegend=False
    )
    
    fig.update_xaxes(title_text='CUT OFF SCORE', row=num_rows, col=1)
    fig.update_yaxes(title_text='False Positive and False Negative Rates', row=num_rows, col=1)
    
    py.iplot(fig, filename='misclassification-subplots')





def get_best_cutoffs_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the best cutoff rows for each model.

    Parameters:
    df (pd.DataFrame): DataFrame containing cutoff metrics.

    Returns:
    pd.DataFrame: DataFrame containing the best cutoff rows for each model.
    """
    grouped = df.groupby('model')
    best_cutoffs = []

    for model, group_df in grouped:
        best_cut_off_row = group_df.sort_values(by=['FALSE_CLASSIFICATION_RATE'], ascending=True)
        best_cut_off_row = best_cut_off_row.head(1)[['model', 'CUT-OFF']]
        best_cutoffs.append(best_cut_off_row)

    best_cutoffs_df = pd.concat(best_cutoffs, ignore_index=True)
    return best_cutoffs_df




def find_ks_thresholds_try(df_evaluation: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Find the KS threshold for each model in the DataFrame.

    This function calculates the KS threshold for each predicted value in the evaluation DataFrame.

    Parameters:
    df_evaluation (pd.DataFrame): DataFrame containing predicted values for different models.
    df_labels (pd.DataFrame): DataFrame containing actual class labels.

    Returns:
    pd.DataFrame: DataFrame containing the KS threshold for each model.
    """

    def find_ks_threshold(y_train: np.array, y_train_predict: np.array) -> tuple:
        """
        Find the threshold that maximizes the Kolmogorov-Smirnov (KS) statistic for class separation.

        This function calculates the KS statistic and identifies the threshold that maximizes
        the separation between the predicted probability distributions of two classes.

        Parameters:
        y_train (np.array): Actual class labels (0 or 1).
        y_train_predict (np.array): Predicted probabilities for the positive class.

        Returns:
        tuple: A tuple containing the KS threshold and the calculated KS statistic.
        """
        y_train = np.array(y_train)
        y_train_predict = np.array(y_train_predict)

        y_1 = y_train_predict[y_train == 1]
        y_0 = y_train_predict[y_train == 0]

        if len(y_1) != len(y_0):
            min_len = min(len(y_1), len(y_0))
            y_1 = np.random.choice(y_1, min_len, replace=False)
            y_0 = np.random.choice(y_0, min_len, replace=False)

        ecdf_1 = np.arange(1, len(y_1) + 1) / len(y_1)
        ecdf_0 = np.arange(1, len(y_0) + 1) / len(y_0)

        ks_statistic = np.max(np.abs(ecdf_1 - ecdf_0))
        ks_threshold = y_1[np.argmax(np.abs(ecdf_1 - ecdf_0))]

        return ks_threshold, ks_statistic
    ks_thresholds = {}

    for model_col in df_evaluation.columns:
        y_train_predict = df_evaluation[model_col]
        ks_threshold, _ = find_ks_threshold(df_labels['target'], y_train_predict)
        ks_thresholds[model_col] = ks_threshold

    ks_thresholds_df = pd.DataFrame(ks_thresholds.items(), columns=['Model', 'KS_Threshold'])
    return ks_thresholds_df






def find_ks_thresholds(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Find the KS threshold for each model in the DataFrame.

    This function calculates the Kolmogorov-Smirnov (KS) threshold for each predicted value in the combined DataFrame.

    Parameters:
    df_combined (pd.DataFrame): DataFrame containing predicted values for different models and actual class labels.

    Returns:
    pd.DataFrame: DataFrame containing the KS threshold for each model.

    Definitions:
    - KS threshold: The value that maximizes the KS statistic, indicating the separation between predicted class distributions.

    Input:
    - df_combined: A DataFrame with columns containing predicted values for different models and a 'target' column
                   containing actual class labels (0 or 1).

    Output:
    - ks_thresholds_df: A DataFrame with two columns: 'Model' and 'KS_Threshold'.
      - 'Model': Name of the model.
      - 'KS_Threshold': The calculated KS threshold for each model.
    """
    def cdf(sample, x, sort=False):
        # Sorts the sample, if unsorted
        if sort:
            sample.sort()
        # Counts how many observations are below x
        cdf = sum(sample <= x)
        # Divides by the total number of observations
        cdf = cdf / len(sample)
        return cdf

    def ks_2samp(sample1, sample2):
        # Gets all observations
        observations = np.concatenate((sample1, sample2))
        observations.sort()
        # Sorts the samples
        sample1_sorted = np.sort(sample1)
        sample2_sorted = np.sort(sample2)
        # Evaluates the KS statistic
        D_ks = []  # KS Statistic list
        for x in observations:
            cdf_sample1 = cdf(sample=sample1_sorted, x=x)
            cdf_sample2 = cdf(sample=sample2_sorted, x=x)
            D_ks.append(abs(cdf_sample1 - cdf_sample2))
        ks_stat = max(D_ks)
        # Calculates the P-Value based on the two-sided test
        # The P-Value comes from the KS Distribution Survival Function (SF = 1-CDF)
        m, n = float(len(sample1_sorted)), float(len(sample2_sorted))
        en = m * n / (m + n)
        p_value = stats.kstwo.sf(ks_stat, np.round(en))
        return ks_stat, p_value

    ks_thresholds = {}

    for model_col in df_combined.columns:
        if model_col != 'target':
            y_train_predict = df_combined[model_col]
            ks_threshold, _ = ks_2samp(df_combined['target'], y_train_predict)
            ks_thresholds[model_col] = ks_threshold

    ks_thresholds_df = pd.DataFrame(ks_thresholds.items(), columns=['Model', 'KS_Threshold'])
    return ks_thresholds_df




def transform_binary_class(dataframe: pd.DataFrame, cut_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a dataframe into binary class based on cutoff values for each model.

    This function takes a dataframe and a cutoff dataframe as inputs and transforms the
    dataframe into binary class based on the specified cutoff values for each model.

    Parameters:
    dataframe (pd.DataFrame): The original dataframe containing model columns.
    cut_dataframe (pd.DataFrame): The dataframe containing model names and their cutoff values.

    Returns:
    pd.DataFrame: A new dataframe with transformed binary class values.
    """

    transformed_dataframe = dataframe.copy()
    
    for model_name, cut_off in cut_dataframe.values:
        if model_name in transformed_dataframe.columns:
            transformed_dataframe[model_name] = (transformed_dataframe[model_name] > cut_off).astype(int)
    
    return transformed_dataframe





def plot_roc_curve(cut_dataframe: pd.DataFrame, target: pd.DataFrame, title: str):
    """
    Plot ROC curves for multiple models against the target variable.

    Parameters:
    cut_dataframe (pd.DataFrame): DataFrame containing cut-off values for each model.
    target (pd.DataFrame): DataFrame containing the target variable.
    title (str): Title for the plot.

    Returns:
    None (displays the plot)
    """
    plt.figure()

    for name in cut_dataframe.columns:
        cut_off = cut_dataframe[name]  # Assuming you want the first cut-off value
        fpr, tpr, _ = roc_curve(target['target'], cut_off)
        plt.plot(fpr, tpr, label=name)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()







def plot_confusion_matrices(models: list, cut_dataframe: pd.DataFrame, target: pd.DataFrame, title: str):
    """
    Plot confusion matrices for multiple models against the target variable.

    Parameters:
    models (list): List of model names.
    cut_dataframe (pd.DataFrame): DataFrame containing cut-off values for each model.
    target (pd.DataFrame): DataFrame containing the target variable and transformed predictions.
    title (str): Title for the plot.

    Returns:
    None (displays the plot)
    """
    plt.figure(figsize=(12, 8))

    for name in models:
        y_pred = cut_dataframe[name]
        cm = confusion_matrix(target['target'], y_pred)
        
        plt.subplot(2, 3, models.index(name) + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()




def transform_to_odds(probability_df, epsilon=1e-8):
    """
    Transform probability values to odds using the formula (1 - p) / (p + epsilon).

    Parameters:
    probability_df (pd.DataFrame): DataFrame containing probability values.
    epsilon (float): Small constant added to the denominator to avoid division by zero.

    Returns:
    pd.DataFrame: DataFrame containing odds values.
    """
    odds_df = (1 - probability_df) / (probability_df + epsilon)
    return odds_df







def min_max_scaling(dataframe):
    """
    Perform Min-Max scaling on each column of the DataFrame.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing the data.

    Returns:
    pd.DataFrame: DataFrame with scaled values using Min-Max scaling.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataframe)
    scaled_dataframe = pd.DataFrame(scaled_data, columns=dataframe.columns)
    return scaled_dataframe



def transform_dataframe(df: pd.DataFrame, offset: float, factor: float) -> pd.DataFrame:
    """
    Transform the values in specified columns of the DataFrame using the formula: new_value = offset + factor * log(value).

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns to be transformed.
    offset (float): Offset value to be added to the transformed values.
    factor (float): Factor to be multiplied with the logarithm of the values.

    Returns:
    pd.DataFrame: DataFrame with transformed values.
    """
    transformed_df = df.copy()

    for col in df.columns:
        transformed_df[col] = offset + factor * np.log(transformed_df[col])

    return transformed_df




def plot_density_subplots(dataframe: pd.DataFrame, model_columns: list, target_col: str, num_cols: int = 2) -> None:
    """
    Plot density plots as subplots for specified model columns against the target variable.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing the data to be plotted.
    model_columns (list): List of column names representing the model prediction columns.
    target_col (str): Name of the target column in the DataFrame.
    num_cols (int, optional): Number of subplot columns. Default is 2.

    Returns:
    None (displays the plot).
    """
    num_models = len(model_columns)
    num_rows = (num_models + num_cols - 1) // num_cols
    figsize = (12, 6 * num_rows)

    plt.figure(figsize=figsize)

    for i, col in enumerate(model_columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.kdeplot(dataframe[dataframe[target_col] == 0][col], label=f'{col} (target==0)')
        sns.kdeplot(dataframe[dataframe[target_col] == 1][col], label=f'{col} (target==1)')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()

    plt.tight_layout()
    plt.show()



























