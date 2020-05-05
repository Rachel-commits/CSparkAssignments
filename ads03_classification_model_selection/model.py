
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def time_buckets(row):
    '''Creates two hour bins from the launch_hour column'''

    if row['deadline_hour'] in (11, 12, 13, 14):
        return '11am-2pm'
    elif row['deadline_hour'] in (15, 16, 17, 18):
        return '3pm-6pm'
    elif row['deadline_hour'] in (19, 20, 21, 22):
        return '7pm-10pm'
    elif row['deadline_hour'] in (23, 0, 1, 2):
        return '11pm-2am'
    elif row['deadline_hour'] in (3, 4, 5, 6):
        return '3am-6am'
    else:
        return None


def get_slug(cat):
    loc_json = json.loads(cat)
    return loc_json["slug"]


def get_pos(cat):
    loc_json = json.loads(cat)
    return loc_json["position"]


def new_cat(row):
    if row['subcat'] in ('rock', 'indie rock'):
        return 'music-rock'
    elif row['subcat'] == 'country & folk':
        return 'music-country'
    elif row['subcat'] == 'nonfiction':
        return 'publishing-nonfiction'
    elif row['subcat'] in ('documentary', 'shorts'):
        return 'film-doc'
    else:
        return row['cat']


def preprocess(df):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains columns used for training (features)
    as well as the target column.

    It also contains some rows for which the target column is unknown.
    Those are the observations you will need to predict for KATE
    to evaluate the performance of your model.

    Here you will need to return the training set: X and y together
    with the preprocessed evaluation set: X_eval.

    Make sure you return X_eval separately! It needs to contain
    all the rows for evaluation -- they are marked with the column
    evaluation_set. You can easily select them with pandas:

         - df.loc[df.evaluation_set]

    For y you can either return a pd.DataFrame with one column or pd.Series.

    :param df: the dataset
    :type df: pd.DataFrame
    :return: X, y, X_eval
    """

    # convert goal to usd
    df['goal_usd'] = df['goal'] * df['static_usd_rate']

    # convert to log usd
    df['log_usd'] = np.log(df['goal_usd'])

    # convert time
    date_cols = ['created_at', 'deadline', 'launched_at']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], origin='unix', unit='s')

    # df['deadline_created_diff'] = (df['deadline'] - df['created_at']).dt.days
    df['launch_created_diff'] = (df['launched_at'] - df['created_at']).dt.days
    df['deadline_launch_diff'] = (df['deadline'] - df['launched_at']).dt.days

    # Extract the year, month, day, time columns
    # Year
    df['deadline_year'] = df['deadline'].dt.year

    # Month
    df['deadline_month'] = df['deadline'].dt.month_name()

    # Day of Week
    df['deadline_day'] = df['deadline'].dt.weekday_name

    # Time of Day
    df['deadline_hour'] = df['deadline'].dt.hour

    # Calculates bins from launch_time
    df['deadline_time'] = df.apply(time_buckets, axis=1)
    df['slug'] = df['category'].apply(get_slug)
    df['position'] = df['category'].apply(get_pos)

    df['cat'] = df['slug'].str.split('/').str[0]
    df['subcat'] = df['slug'].str.split('/').str[1]
    df['cat2'] = df.apply(new_cat, axis=1)

    cols_to_drop = ['id', 'photo', 'name', 'blurb', 'slug', 'friends',
                    'is_starred', 'is_backing', 'permissions', 'goal',
                    'static_usd_rate', 'currency_symbol',
                    'disable_communication', 'currency_trailing_code',
                    'currency', 'creator', 'location', 'profile', 'urls',
                    'source_url', 'created_at', 'deadline', 'launched_at',
                    'deadline', 'created_at', 'deadline_hour', 'subcat',
                    'cat', 'category', 'goal_usd',
                    'deadline_time', 'deadline_day', 'deadline_month',
                    'deadline_year', 'position']

    df.drop(cols_to_drop, axis=1, inplace=True)

    df_transformed = pd.get_dummies(df)

    # save labels to know what rows are in evaluation set
    # evaluation_set is a boolean so we can use it as mask
    msk_eval = df_transformed.evaluation_set

    X = df_transformed[~msk_eval].drop(["state"], axis=1)
    y = df_transformed[~msk_eval]["state"]
    X_eval = df_transformed[msk_eval].drop(["state"], axis=1)

    # standardise
    scaler = StandardScaler()  # create an instance
    scaler.fit(X)  # fit to the data

    # recreate a features data frame
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_eval = pd.DataFrame(scaler.transform(X_eval), columns=X.columns)

    return X, y, X_eval


def train(X, y):
    """Trains a new model on X and y and returns it.

    :param X: your processed training data
    :type X: pd.DataFrame
    :param y: your processed label y
    :type y: pd.DataFrame with one column or pd.Series
    :return: a trained model
    """

    bestc = 1
    bestpen = 'l1'
    model = LogisticRegression(solver='liblinear', C=bestc, penalty=bestpen)
    # model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X, y)
    return model


def predict(model, X_test):
    """This functions takes your trained model as well
    as a processed test dataset and returns predictions.

    On KATE, the processed test dataset will be the X_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with one
    column or a pd.Series

    :param model: your trained model
    :param X_test: a processed test set (on KATE it will be X_eval)
    :return: y_pred, your predictions
    """

    y_pred = model.predict(X_test)
    return y_pred

df = pd.read_csv("data/kickstarter.csv")
X, y, X_eval = preprocess(df)
print(X.columns)
# model = train(X, y)
# y_pred = predict(model, X_eval)
# print(y_pred)
