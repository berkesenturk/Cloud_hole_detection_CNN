from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
import os


# Connect to Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
)

# Submit training job
job = command(
    code="./src",
    command="""
        python train_azure.py
        --data_path ${{inputs.dataset}}
        --freeze_backbone
        --epochs 50
        --batch_size 32
        --learning_rate 1e-4
        --patience 10
        --augment
    """,
    inputs={
        "dataset": Input(
            type="uri_folder",
            path="azureml://datastores/gold/paths/datasets/v1.0_baseline"
        )
    },
    environment="azureml:cloud-detector-env:1",
    compute="gpu-cluster",
    display_name="cloud-hole-frozen-backbone",
    experiment_name="cloud-hole-detection"
)

# Run job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.studio_url}")
