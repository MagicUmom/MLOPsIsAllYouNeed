import os
import yaml

# yaml to dict
with open('flow.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# scheduler 
scheduler_method = data['scheduler']['method']

if scheduler_method == 'interval':
    schedule = f"--{scheduler_method} {data['scheduler']['interval']}"
elif scheduler_method == 'cron':
    schedule = f"--{scheduler_method} '{data['scheduler']['cron']}'"
elif scheduler_method == 'rrule':
    schedule = f"--{scheduler_method} '{data['scheduler']['rrule']}'"
else:
    raise ValueError('scheduler.method is invalid!')

# cli 
os.system(f"prefect work-pool create --type prefect-agent {data['deployment']['pool_name']}")
os.system(f"prefect deployment build -n {data['deployment']['deploy_name']} \
                                     -p {data['deployment']['pool_name']} \
                                     -q {data['deployment']['queue_name']} \
                                     --timezone {data['scheduler']['timezone']} \
                                     {schedule} \
                                     --override env.EXTRA_PIP_PACKAGES='dill' \
                                     -sb remote-file-system/{data['RFS_NAME']}/{data['deployment']['flow_name']}/{data['deployment']['deploy_name']} \
                                     --apply \
                                     /root/flows/{data['deployment']['flow_py_name']}:{data['deployment']['flow_name']}")