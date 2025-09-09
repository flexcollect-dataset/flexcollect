import boto3
import os

ecs = boto3.client("ecs")

def handler(event, context):
    """
    Tiny dispatcher: starts a one-off Fargate task that runs your long job.
    Optionally pass parameters via event; they’ll be injected as env var FC_EVENT_JSON.
    """
    overrides = []
    if event:
        overrides = [{
            "name": "flexcollect",  # must match your container name in the task def
            "environment": [
                {"name": "FC_EVENT_JSON", "value": json_dumps_safe(event)}
            ]
        }]

    resp = ecs.run_task(
        cluster="flexcollect-cluster",
        taskDefinition="flexcollect-task",  # family only -> latest ACTIVE revision
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": "subnet-0e75e6788e3ada572,subnet-0b299bc07e425f4dd",
                "securityGroups": "sg-0ea71df72778eab13",
                "assignPublicIp": "ENABLED"
            }
        },
        overrides={"containerOverrides": overrides} if overrides else {}
    )
    return resp

def json_dumps_safe(obj):
    import json
    try:
        return json.dumps(obj)
    except Exception:
        return "{}"
