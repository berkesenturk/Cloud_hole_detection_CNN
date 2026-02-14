#!/usr/bin/env python3
"""
Register custom environment in Azure ML
"""

import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential

# Connect to Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
    resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
    workspace_name=os.getenv('AZURE_ML_WORKSPACE')
)

print("Creating custom environment...")

# Create environment from local files
env = Environment(
    name="cloud-detector-env",
    description="Custom environment for cloud hole detection",
    build=BuildContext(path="./src"),  # Contains environment.yml, conda.yml, requirements.txt
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
)

# Register environment
env = ml_client.environments.create_or_update(env)

print(f"✓ Environment registered: {env.name}:{env.version}")
print(f"  Use in job: azureml:{env.name}:{env.version}")