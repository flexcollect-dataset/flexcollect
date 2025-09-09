import boto3
import os

ecs = boto3.client("ecs")


ecs.run_task(
  cluster="flexcollect-cluster",
  taskDefinition="flexcollect-task",  # family only -> latest ACTIVE revision
  launchType="FARGATE",
  networkConfiguration={
    "awsvpcConfiguration": {
      "subnets": ["subnet-0e75e6788e3ada572","subnet-0b299bc07e425f4dd"],
      "securityGroups": ["sg-0ea71df72778eab13"],
      "assignPublicIp": "ENABLED"
    }
  }
)
