import os
from prefect.filesystems import RemoteFileSystem


minio_block = RemoteFileSystem(
    basepath="s3://" + os.getenv('PREFECT_BUCKET_NAME'),
    settings={
        "key": os.getenv('MINIO_ROOT_USER'),
        "secret": os.getenv('MINIO_ROOT_PASSWORD'),
        "client_kwargs": {"endpoint_url": "http://" + os.getenv('RFS_REMOTE_SERVER') + ":9000"},
    },
)

minio_block.save(os.getenv('RFS_NAME'), overwrite=True)
