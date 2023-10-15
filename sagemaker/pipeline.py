from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.sklearn import SKLearn
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TrainingInput
from getconfig import CLOUD_CONFIG
import sagemaker

local_pipeline_session = LocalPipelineSession()

sklearn_estimator = SKLearn(
    sagemaker_session=local_pipeline_session,
    role=sagemaker.get_execution_role(),
    instance_type="ml.m4.large",
    instance_count=1,
    framework_version="1.3.1",
    py_version="py36",
    entry_point="src/train_and_deploy.py",
)

step = TrainingStep(
    name="MyTrainingStep",
    step_args=sklearn_estimator.fit(
        inputs=TrainingInput(s3_data=f"s3://{CLOUD_CONFIG['s3_bucket']}/data.csv"),
    )
)

pipeline = Pipeline(
    name="MyPipeline",
    steps=[step],
    sagemaker_session=local_pipeline_session
)

pipeline.create(
    role_arn=sagemaker.get_execution_role(),
    description="local pipeline example"
)

execution = pipeline.start()

steps = execution.list_steps()

training_job_name = steps['PipelineExecutionSteps'][0]['Metadata']['TrainingJob']['Arn']

step_outputs = local_pipeline_session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
