from fbprophet import Prophet
import pandas as pd


def preprocess(df):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains the time axis and the target column.

    It also contains some rows for which the target column is unknown.
    Those are the observations you will need to predict for KATE
    to evaluate the performance of your model.

    Here you will need to return the training time serie: ts together
    with the preprocessed evaluation time serie: ts_eval.

    Make sure you return ts_eval separately! It needs to contain
    all the rows for evaluation -- they are marked with the column
    evaluation_set. You can easily select them with pandas:

         - df.loc[df.evaluation_set]


    :param df: the dataset
    :type df: pd.DataFrame
    :return: ts, ts_eval
    """

    df['day'] = pd.to_datetime(df['day'])
    df = df.rename(columns={'day': 'ds', 'consumption': 'y'})
    ts = df[~df['evaluation_set']].drop('evaluation_set', axis=1)
    ts_eval = df[df['evaluation_set']].drop('evaluation_set', axis=1)
    return ts, ts_eval


def train(ts):
    """Trains a new model on ts and returns it.

    :param ts: your processed training time serie
    :type ts: pd.DataFrame
    :return: a trained model
    """

<<<<<<< HEAD
    forecast_model = Prophet(growth='linear',  weekly_seasonality=5, yearly_seasonality=5, )
=======
    forecast_model = Prophet(growth='linear',  weekly_seasonality=4, yearly_seasonality=4,)
>>>>>>> 6e2ca7d494076b0136aa642e2abcc76df5a6786f
    forecast_model.fit(ts)
    return forecast_model


def predict(model, ts_test):
    """This functions takes your trained model as well
    as a processed test time serie and returns predictions.

    On KATE, the processed testt time serie will be the ts_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with one column
    or a pd.Series

    :param model: your trained model
    :param ts_test: a processed test time serie (on KATE it will be ts_eval)
    :return: y_pred, your predictions
    """
<<<<<<< HEAD

=======
    
>>>>>>> 6e2ca7d494076b0136aa642e2abcc76df5a6786f
    dates = ts_test[['ds']]
    predictions = model.predict(dates)['yhat']
    return predictions
