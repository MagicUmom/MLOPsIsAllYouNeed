import pandas as pd
from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
import os


def main():
    '''
    - 請先完成快速安裝與開發實驗階段
    - 此部署階段主要跟大家分享如何將訓練好的模型進行部署，一般來說會有兩道手續：

    1. 從眾多實驗中找出要將哪個模型進行部署，需要對該模型進行"註冊"(Register)
    2. 使用註冊後的進行部署，並實際進行資料推論

    - 因為部署階段需要使用到前面安裝步驟的相關套件，所以請先確保有確實完成快速安裝
    - 此階段需要幾個訓練完成的模型並上傳至 MLflow，也請確定"開發實驗階段"有確實完成
    
    功能介紹
    1. 註冊模型(Register model)
    2. 模型部署預測
    '''
    # MLflow 環境設定
    load_dotenv('.env')
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('MINIO_ROOT_USER')
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('MINIO_ROOT_PASSWORD')
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv('MLFLOW_S3_ENDPOINT_URL')

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    # 獲得實驗編號
    target_experiments = {}
    for rm in mlflow.search_experiments(filter_string="name = 'Titanic'"):
        target_experiments = dict(rm)

    experiment_id = target_experiments['experiment_id']

    # 透過實驗編號取得每一次的模型紀錄
    runs_df = mlflow.search_runs(experiment_ids=experiment_id)
    runs_df = runs_df.sort_values(by=['metrics.Test Accuracy'], ascending=False)
    runs_df.reset_index(inplace=True)

    # 將評估指標表現最好的模型進行”註冊“
    best_run = runs_df.iloc[0]
    best_run_id = best_run["run_id"]
    mv = mlflow.register_model(model_uri="runs:/%s/Model"%best_run_id, 
                            name="Titanic_model")
    
    # 將註冊後的模型加入版本號(Staging, Production, Archived)
    client = MlflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))
    client.transition_model_version_stage(
        name="Titanic_model", version=int(mv.version), stage="Production"
    )

    # 下載註冊後的模型, 並使用MLflow 讀取模型
    model_name = "Titanic_model"
    stage = "Production"

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    
    # 建立一筆測試資料，並進行預測
    test = pd.DataFrame()
    test["Age"] = [0.28]
    test["SibSp"] = [0.0]
    test["Parch"] = [0.0]
    test["Fare"] = [0.014151]

    print(test)
    result = model.predict(test)
    if result :
        print("Survived")
    else:
        print("Dead")

if __name__=="__main__":
    main()