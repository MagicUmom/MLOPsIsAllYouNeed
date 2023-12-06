import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms
import numpy as np
from prefect import task, flow
import mlflow
from datetime import datetime
import yaml

class Net(nn.Module):
    """
    針對手寫數字辨識建立簡單的CNN模型
    """
    def __init__(self):
        super(Net, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding='same'
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=4,
                kernel_size=5,
                stride=1,
                padding='same'
            ),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()
        self.classify = nn.Linear(
            in_features=4 * 14 * 14,
            out_features=10
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.flatten(x)
        output = self.classify(x)

        return output

@task(name='Pulling dataset', log_prints=True)
def fetch_dataset(remote):
    print(f'Pulling data from remote {remote}...')
    # DVC pulls the data from remote
    os.chdir('./data')
    os.system('git init')
    os.system('dvc init')
    os.system(f'dvc remote add remote {remote} -f')
    os.system('dvc remote modify remote endpointurl ${MLFLOW_S3_ENDPOINT_URL}')
    os.system('dvc pull -r remote')

    # 取得資料標籤並紀錄
    data_version = os.popen('git describe --tags').read()[:-1]
    print(f'Current data version: {data_version}')

    os.chdir('../')

    return data_version

@task(name='Preprocessing')
def preprocessing(root=f'./data/MNIST/train', shuffle_data=True, batch_size=64, random_seed=1, val_size=0.2):
    transform_list = transforms.Compose(
        [
            transforms.Grayscale(),  # MNIST需要將圖片轉換為灰階，否則預設是3通道圖片
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
        ]
    )  # 圖片前處理

    train_data = torchvision.datasets.ImageFolder(
        root=root,
        transform=transform_list
    )  # 讀取root路徑底下的圖片資料

    indices = list(range(len(train_data)))
    split_point = int(np.floor(val_size * len(train_data)))

    if shuffle_data:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split_point:], indices[:split_point]
    print(f'Training set size: {len(train_indices)}, validation set size: {len(val_indices)}')

    # 訓練與驗證集的DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=4
    )
    val_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=4
    )

    return train_loader, val_loader

@task(name='Model training', log_prints=True)
def model_training(train_loader, val_loader, model, data_version='1.0', n_epochs=5, learning_rate=0.01, device_name='cpu'):
    date_time_now = datetime.now()
    mlflow_run_name = f'Run_{str(date_time_now.strftime("%Y-%m-%d"))}-{str(date_time_now.strftime("%H-%M-%S"))}'

    with mlflow.start_run(run_name=mlflow_run_name):
        # 紀錄模型框架跟訓練裝置
        mlflow.set_tags(
            {
                'Training device': device_name,
                'Phase': 'Retraining'
            }
        )

        device = torch.device(device_name)
        print(f'Training on {device_name}.')

        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        loss_fn = nn.CrossEntropyLoss()

        # 紀錄實驗參數
        mlflow.log_params({
            'Data version': data_version,
            'Model': model.__class__.__name__,
            'Number of epochs': n_epochs,
            'Optimizer': type(optimizer).__name__,
            'Initial learning rate': learning_rate,
        })

        for epoch in range(n_epochs):
            train_loss_list = []
            n_correct_train = 0
            n_train_data = 0

            # 訓練階段
            model.train()
            for idx, (imgs, train_true_labels) in enumerate(train_loader):
                imgs = imgs.to(device)
                train_true_labels = train_true_labels.to(device)

                optimizer.zero_grad()
                train_outputs = model(imgs)

                train_loss = loss_fn(train_outputs, train_true_labels)
                train_loss_list.append(train_loss.item() * len(train_true_labels))
                train_loss.backward()
                optimizer.step()

                train_outputs_label = torch.argmax(train_outputs, 1)
                n_correct_train = n_correct_train + len(torch.where(train_outputs_label == train_true_labels)[0])
                n_train_data = n_train_data + len(train_true_labels)

            mlflow.log_metric('Training loss', np.sum(train_loss_list) / n_train_data, step=epoch)
            mlflow.log_metric('Training accuracy', n_correct_train / n_train_data, step=epoch)

            val_loss_list = []
            n_correct_val = 0
            n_val_data = 0

            # 驗證階段
            model.eval()
            with torch.no_grad():
                for _idx, (imgs, val_true_labels) in enumerate(val_loader):
                    imgs = imgs.to(device)
                    val_true_labels = val_true_labels.to(device)

                    val_outputs = model(imgs)

                    val_loss_list.append(loss_fn(val_outputs, val_true_labels).item() * len(val_true_labels))

                    val_outputs_label = torch.argmax(val_outputs, 1)
                    n_correct_val = n_correct_val + len(torch.where(val_outputs_label == val_true_labels)[0])
                    n_val_data = n_val_data + len(val_true_labels)

                mlflow.log_metric('Validation loss', np.sum(val_loss_list) / n_val_data, step=epoch)
                mlflow.log_metric('Validation accuracy', n_correct_val / n_val_data, step=epoch)

            print(
                f'{epoch + 1}/{n_epochs} {idx + 1}/{len(train_loader)}, \
                train loss: {np.sum(train_loss_list) / n_train_data:8.5f}, \
                train acc: {n_correct_train / n_train_data:6.4f}, \
                val loss: {np.sum(val_loss_list) / n_val_data:8.5f}, \
                val acc: {n_correct_val / n_val_data:6.4f}           ',
                end='\r')

        mlflow.pytorch.log_model(model, artifact_path='Model')  # 儲存模型，未來可隨時用來進行推論

@flow(name=f'MNIST', log_prints=True)
def main():
    with open('flow.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # 設定環境變數
    # 這邊很重要，如果程式沒有正確讀取這些環境變數，可能會造成MLflow無法追蹤實驗，或無法執行
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('MINIO_ROOT_USER')
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('MINIO_ROOT_PASSWORD')
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')
    os.environ["LOGNAME"] = cfg['developer_name']  # 設定要紀錄在實驗的使用者名稱

    # 設定MLflow用來儲存實驗結果的路徑
    # mlflow.set_tracking_uri(os.getenv('MLFLOW_SERVER'))
    print(f"MLflow server: {mlflow.get_tracking_uri()}")

    # 設定實驗名稱，如果該實驗不存在則建立
    if not mlflow.get_experiment_by_name(cfg['experiment_name']):  # 確認'experiment_name'為名的實驗存在與否
        mlflow.create_experiment(cfg['experiment_name'], f"s3://{os.getenv('MLFLOW_BUCKET_NAME')}/")
    mlflow.set_experiment(cfg['experiment_name'])

    # 從DVC remote下載資料
    data_version = fetch_dataset(cfg['dvc_remote'])

    # 資料前處理
    train_loader, val_loader = preprocessing(
        root=cfg['train_data_path'],
        shuffle_data=cfg['hyperparameter']['shuffle_data'],
        batch_size=cfg['hyperparameter']['batch_size'],
        random_seed=cfg['hyperparameter']['random_seed'],
        val_size=cfg['hyperparameter']['val_size']
    )

    # 模型訓練
    model_training(
        train_loader=train_loader,
        val_loader=val_loader,
        model=Net(),
        data_version=data_version,
        n_epochs=cfg['hyperparameter']['n_epochs'],
        learning_rate=cfg['hyperparameter']['learning_rate'],
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    )


if __name__ == '__main__':
    main()
