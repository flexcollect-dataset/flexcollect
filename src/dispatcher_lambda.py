import boto3
import os

ecs = boto3.client("ecs")

CLUSTER = os.getenv("ECS_CLUSTER", "flexcollect-cluster")
TASK_FAMILY = os.getenv("ECS_TASK_FAMILY", "flexcollect-task")

# Accept comma-separated env vars but convert to proper Python lists
def _split_ids(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]

SUBNETS = _split_ids(os.getenv("ECS_SUBNETS", "subnet-0e75e6788e3ada572,subnet-0b299bc07e425f4dd"))
SECURITY_GROUPS = _split_ids(os.getenv("ECS_SECURITY_GROUPS", "sg-0ea71df72778eab13"))


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
        cluster=CLUSTER,
        taskDefinition=TASK_FAMILY,  # family only -> latest ACTIVE revision
        launchType="FARGATE",
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": SUBNETS,
                "securityGroups": SECURITY_GROUPS,
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
