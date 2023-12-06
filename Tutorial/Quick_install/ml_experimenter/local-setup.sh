#! /bin/bash

set -eu

echo "Step 1. Installing packages..."
pip install -r requirements_sys.txt

echo "step 2. Setting the environment variables..."
source .env
export AWS_ACCESS_KEY_ID=$MINIO_ROOT_USER
export AWS_SECRET_ACCESS_KEY=$MINIO_ROOT_PASSWORD

echo "Step 3. Initializing Git and DVC ..."
if [ ! -e .git ]; then
    git init
else
    echo "Git is already initialized."
fi 

if [ ! -e .dvc ]; then 
    dvc init
else
    echo "DVC is already initialized."
fi 

dvc remote add -f minio_s3 s3://$DVC_BUCKET_NAME/$PROJECT_NAME
dvc remote modify minio_s3 endpointurl $MLFLOW_S3_ENDPOINT_URL

cat <<EOF

==========================
Done & Enjoy Your Project!
==========================
Note. DVC remote strage name: minio_s3
EOF