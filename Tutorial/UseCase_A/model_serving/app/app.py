import os
import numpy as np
import torch
from flask import Flask, request, redirect, url_for, render_template
import mlflow
from torchvision import transforms
from dotenv import load_dotenv
import csv
from datetime import datetime
from PIL import Image

UPLOAD_FOLDER = 'static/candidates/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']  # 允許的副檔名清單，在清單裡面的才能上傳

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
load_dotenv('../.env')

app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def write_detection_result(path, cls, prob):  # 將圖片路徑、類別以及機率值寫入在output.csv，後續可讓工程師進行標籤確認
    if os.path.exists('detection_result.csv'):
        with open('detection_result.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([path, cls, prob])
    else:
        with open('detection_result.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            fieldnames = ['image', 'class', 'probability']
            writer.writerow(fieldnames)
            writer.writerow([path, cls, prob])  # path + datetime


def write_user_feedback(path, feedback, message):  # 將圖片路徑、使用者回饋以及文字建議紀錄到user_feedback.csv，作為效能追蹤的根據
    if os.path.exists('user_feedback.csv'):
        with open('user_feedback.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([path, feedback, message])
    else:
        with open('user_feedback.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            fieldnames = ['image', 'feedback', 'message']
            writer.writerow(fieldnames)
            writer.writerow([path, feedback, message])

def allowed_file(filename):  # 確認副檔名是否在ALLOWED_EXTENSIONS允許的範圍內
    return '.' in filename and \
           filename.rsplit('.', 1)[-1] in ALLOWED_EXTENSIONS


prediction_class, prediction_prob, full_filename = 0, 0, 'static/image/example_img.png'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global prediction_class, prediction_prob, full_filename

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            full_filename = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename + f'_{str(datetime.now().strftime("%Y-%m-%d"))}-{str(datetime.now().strftime("%H-%M-%S"))}'
            ) + f'.{file.filename.rsplit(".")[-1]}'

            file.save(full_filename)  # 將使用者上傳的圖片儲存在UPLOAD_FOLDER
            print(f'Uploaded file saved to {full_filename}.')

            prediction_class, prediction_prob = prediction(full_filename)

        return redirect(
            url_for(
                'prediction',
                filename=full_filename
            )
        )

    context = {
        'model_version': logged_model_info.version,
        'image_path': full_filename,
        'prediction_class': prediction_class,
        'prediction_prob': np.round(prediction_prob, 4)
    }

    return render_template('index.html', **context)


@app.route('/', methods=['POST'])
def prediction(filename):
    global loaded_model

    # 讀取檔案以及前處理
    image = Image.open(filename)

    infer_transform = transforms.Compose(
        [
            transforms.Grayscale(),  # MNIST需要將圖片轉換為灰階，否則預設是3通道圖片
            transforms.Resize([28, 28]),
            transforms.ToTensor(),
        ]
    )

    image = infer_transform(image)

    # 推論
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prediction = loaded_model(torch.unsqueeze(image, 0).to(device)).float().detach().numpy()[0]
    prediction = np.exp(prediction) / sum(np.exp(prediction))
    print(f'Prediction result: {prediction}')

    return np.argmax(prediction), np.max(prediction)


@app.route('/user_feedback', methods=['POST'])
def user_feedback():
    global prediction_class, prediction_prob, full_filename
    if request.method == 'POST':
        write_user_feedback(
            path=full_filename,
            feedback=request.form.get('feedback'),
            message=request.form.get('describe')
        )

    prediction_class, prediction_prob, full_filename = 0, 0, 'static/image/example_img.png'

    return redirect(url_for('upload_file'))


if __name__ == '__main__':
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('AWS_ACCESS_KEY_ID')
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('AWS_SECRET_ACCESS_KEY')

    mlflow.set_tracking_uri(os.getenv('MLFLOW_SERVER'))
    print(f'MLflow tracking URI: {mlflow.get_tracking_uri()}')

    cli = mlflow.tracking.MlflowClient()  # 建立MlflowClient來搜尋模型

    logged_model_info = cli.get_latest_versions(name='MNIST', stages=["Production"])[0]  # 搜尋目前為Production的模型
    print(f'Model version: {logged_model_info.version}')
    logged_model_run_id = logged_model_info.run_id  # 模型的run_id

    logged_model_path = f'runs:/{logged_model_run_id}/Model'  # 格式'runs:/<model_run_id>/Model'為MLflow固定的寫法

    # 讀取PyTorch模型
    loaded_model = mlflow.pytorch.load_model(
        model_uri=logged_model_path,
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    loaded_model.eval()

    app.run(port=10000, host='0.0.0.0', debug=True)
