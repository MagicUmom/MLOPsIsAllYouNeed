unzip MNIST.zip  # 解壓縮MNIST.zip，如果已經解壓縮過，這條可以註解掉

# 製作v1.0的訓練資料，並讓DVC開始追蹤
git init  # 需要先以git對資料夾進行初始化
dvc init  # DVC對資路夾進行初始化
dvc add MNIST  # 將MNIST資料夾以DVC追蹤
git add .gitignore MNIST.dvc  # git add 後面的檔案順序可對調
git commit -m "First version of training data."  # 以git對.dvc進行版控
git tag -a "v1.0" -m "Created MNIST."  # 建立標籤，未來要重回某個版本時比較方便

dvc remote add remote s3://dvcmnist/  # remote為自定義的遠端名稱
dvc remote modify remote endpointurl http://localhost:9000
export AWS_ACCESS_KEY_ID=admin
export AWS_SECRET_ACCESS_KEY=adminsecretkey
dvc push -r remote  # 推送至名為remote的遠端

python3 expand_train_data.py  # 將額外的訓練資料加入train裡面

# 製作v2.0的訓練資料
dvc add MNIST
git add MNIST.dvc
git commit -m "Add some images"
git tag -a "v2.0" -m "More images added."
dvc push -r remote
#git push  # 如果有遠端的git repo才需要執行
