import os
from sagemaker.sklearn import SKLearn
from getconfig import CLOUD_CONFIG
# import boto3

LOCAL_MODE = os.getenv("LOCAL", False)


def submit_job():
    print("local mode", LOCAL_MODE)
    # Initialise SDK
    sklearn_estimator = SKLearn(
        entry_point='../src/train_and_deploy.py',
        role=CLOUD_CONFIG['sagemaker_role_id'],
        instance_type='local' if LOCAL_MODE else 'ml.m4.large',
        hyperparameters={
            'sagemaker_submit_directory': f"s3://{CLOUD_CONFIG['s3_bucket']}",
        },
        framework_version='1.2-1',
        metric_definitions=[
            {'Name': 'train:score', 'Regex': 'train:score=(\S+)'}]
    )
    print("estimator created")
    # Run model training job
    sklearn_estimator.fit({
        'train': "file://../train/data.csv" if LOCAL_MODE else f"s3://{CLOUD_CONFIG['s3_bucket']}/data.csv"
    })
    print("fit finished")
    # Deploy trained model to an endpoint
    sklearn_estimator.deploy(
        instance_type='local' if LOCAL_MODE else 'ml.t2.medium',
        initial_instance_count=1,
    )
    print("estimator deployed")


if __name__ == '__main__':
    submit_job()
