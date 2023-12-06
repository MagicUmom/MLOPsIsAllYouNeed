import os

import mlflow
from mlflow.models import infer_signature
from prefect import flow, task

# ======= requirements.txt ====== #
import dill
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
# ================================== #

@task
def set_experiment(EXPERMENT_NAME = "nyc-taxi-experiment"):
    # setting experiment name 
    existing_exp = mlflow.get_experiment_by_name(EXPERMENT_NAME)
    if not existing_exp:
        mlflow.create_experiment(EXPERMENT_NAME, "s3://"+os.getenv('MLFLOW_BUCKET_NAME')+"/")
    mlflow.set_experiment(EXPERMENT_NAME)

def preprocessor(df):
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df.loc[:,'duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.loc[:,'duration'] = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df.loc[:,'PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    
    target = 'duration'
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    x = df[categorical + numerical]
    y = df[target].values
    
    return x, y
preprocessor_task = task(preprocessor)

@task
def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                # num_boost_round=100,
                num_boost_round=2,
                evals=[(valid, 'validation')],
                # early_stopping_rounds=20
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )

    return best_result

@task
def train_best_model(train, valid, y_val, 
                     dv, best_result, X_train): # for log_model
    with mlflow.start_run() as run:
        # parameters 
        best_params = {
            'learning_rate': best_result['learning_rate'],
            'max_depth': int(best_result['max_depth']),
            'min_child_weight': best_result['min_child_weight'],
            'objective': 'reg:linear',
            'reg_alpha': best_result['reg_alpha'],
            'reg_lambda': best_result['reg_lambda'],
            'seed': 42
        }

        mlflow.log_params(best_params)

        # model
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            # num_boost_round=100,
            num_boost_round=2,
            evals=[(valid, 'validation')],
            # early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)

        # artifacts
        artifacts = { # this dict will server to XGBmodel as 'context.artifacts'
            # artifacts should not contain functions decorated with prefect.task/flow
            'preprocessor': 'preprocessor.b',
            'dv': 'dv.b',
            'model':'xgb.json'
        }

        with open("preprocessor.b", "wb+") as f_out:
            dill.dump(preprocessor, f_out)
        with open('dv.b', 'wb+') as f_out:
            dill.dump(dv, f_out)
        booster.save_model('xgb.json')
        
        # signature
        signature = infer_signature(X_train, y_pred) # sample of input, output # not support pandas.dtyp.object
        pip_requirements = ["-r requirements.txt"]

        # customized model class 
        class XGBmodel(mlflow.pyfunc.PythonModel):
            def load_context(self, context): # called when load_model
                import dill
                import xgboost as xgb
                import pandas as pd

                with open(context.artifacts["preprocessor"], "rb") as f:
                    self.preprocessor = dill.load(f)
                with open(context.artifacts["dv"], "rb") as f:
                    self.dv = dill.load(f)
                    
                self.model = xgb.Booster()
                self.model.load_model(context.artifacts["model"])

            def predict(self, context, model_input): # called when model.predict 
                # suppose input type is pd.dataframe
                X_test = model_input.to_dict(orient='records')
                X_test = self.dv.transform(X_test)
                X_test = xgb.DMatrix(X_test)
                return self.model.predict(X_test)

        # logModel
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=XGBmodel(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=pip_requirements
        )

@flow
def main(train_path: str="./data/green_tripdata_2021-01.parquet",
        val_path: str="./data/green_tripdata_2021-02.parquet",
        EXPERMENT_NAME = "nyc-taxi-experiment"):
    
    set_experiment(EXPERMENT_NAME)

    # prepare data
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    X_train, y_train = preprocessor_task(df_train)
    X_val, y_val = preprocessor_task(df_val)

    dv = DictVectorizer()

    X_train_trans = dv.fit_transform(X_train.to_dict(orient='records'))
    X_val_trans = dv.transform(X_val.to_dict(orient='records'))

    train = xgb.DMatrix(X_train_trans, label=y_train)
    valid = xgb.DMatrix(X_val_trans, label=y_val)

    best_result = train_model_search(train, valid, y_val)
    train_best_model(train, valid, y_val, dv, best_result, X_train)


if __name__ == "__main__":
    main()

