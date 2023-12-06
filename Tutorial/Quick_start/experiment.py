import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from dotenv import load_dotenv
import mlflow
import os
from datetime import datetime
import gdown


def main():
    '''
    開發實驗階段
    - 請先完成快速安裝
    - 此為開發實驗階段主要跟大家分享如何將過程紀錄在MLflow中，並將每次的實驗紀錄(模型參數, Loss曲線, 評估指標…等)儲存起來，方便之後多實驗結果比較。
    功能介紹
    - 紀錄模型超參數及訓練結果、並將模型存到 Minio裡面
    '''

    # 使用 Gdown 獲取資料
    # 資料下載 url
    url = "https://drive.google.com/file/d/13_yil-3-ihA_px4nFdWq8KVoQWxxffHm/view?usp=sharing"
    gdown.download(url, output='data/titanic_data.csv', quiet=False, fuzzy=True)

    # 資料讀取
    data = pd.read_csv("data/titanic_data.csv")

    # 將 Age 的缺失值補 Age 的平均數
    data['Age'].fillna(data['Age'].mean(), inplace = True)
    # 資料 Ground Truth 設定
    y_train = data.Survived
    X_train = data.drop(columns='Survived')
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    X_train = X_train[numerical_features]

    # 將連續變項歸一化(MinMaxScaler): 將數值壓縮到0~1之間
    scaler = MinMaxScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    
    # 建立模型
    model_svc = SVC(C=1.0,        # Regularization parameter
                    kernel='rbf') # kernel

    model_xgb = XGBClassifier(max_depth=2,
                            learning_rate=0.1)
    
    # 訓練模型
    model_svc.fit(X_train, y_train)
    model_xgb.fit(X_train, y_train)

    # 評估指標
    y_pred = model_svc.predict(X_train)
    accuracy_svc = (y_pred == y_train).sum()/y_train.shape[0]

    y_pred = model_xgb.predict(X_train)
    accuracy_xgb = (y_pred == y_train).sum()/y_train.shape[0]
    
    # MLflow 環境設定
    load_dotenv('.env')
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('MINIO_ROOT_USER')
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('MINIO_ROOT_PASSWORD')
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    # MLflow 實驗名稱設定
    experiment_name = 'Titanic'
    existing_exp = mlflow.get_experiment_by_name(experiment_name)

    if not existing_exp:
        mlflow.create_experiment(experiment_name, "s3://mlflow/")
    mlflow.set_experiment(experiment_name)

    # MLflow 上面訓練結果紀錄
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H-%M-%S")
    with mlflow.start_run(run_name='Run_%s' % dt_string):
        # 設定開發者名稱
        mlflow.set_experiment_tag('developer', 'GU')

        # 設定需要被紀錄的參數
        mlflow.log_params({
            'Model': "XGboost",
            'Learning rate': 0.1,
        })

        # 設定需要被紀錄的評估指標
        mlflow.log_metric("Test Accuracy", accuracy_xgb)

        # 上傳訓練好的模型
        mlflow.xgboost.log_model(model_xgb, artifact_path='Model')

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H-%M-%S")
    with mlflow.start_run(run_name='Run_%s' % dt_string):
        # 設定開發者名稱
        mlflow.set_experiment_tag('developer', 'GU')

        # 設定需要被紀錄的參數
        mlflow.log_params({
            'Model': 'SVC',
            'C': 1,
            'kernel':'rbf',
        })

        # 設定需要被紀錄的評估指標
        mlflow.log_metric("Test Accuracy", accuracy_svc)
        # 上傳訓練好的模型
        mlflow.sklearn.log_model(model_svc, artifact_path='Model')

if __name__=="__main__":
    main()