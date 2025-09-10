# flexcollect

Repo for collecting data and add it to dataset

## Run as ECS task (no Lambda)

This project now runs directly as an ECS Fargate task. The container executes `src.lambda_function_runner` which invokes your main logic in `src.lambda_function.lambda_handler` once.

### Steps

1. Build and push the image:
   - Authenticate to ECR and set your `IMAGE_URI`.
   - Build and push:
     ```bash
     docker build -t "$IMAGE_URI" .
     docker push "$IMAGE_URI"
     ```
2. Update `infra/ecrtask.json`:
   - Set `containerDefinitions[0].image` to your `IMAGE_URI`.
   - Keep `entryPoint` as `["python","-u","-m","src.lambda_function_runner"]`.
3. Register the task definition:
   - ```bash
     aws ecs register-task-definition --cli-input-json file://infra/ecrtask.json | cat
     ```
4. Run a one-off task:
   - ```bash
     aws ecs run-task \
       --cluster <your-cluster> \
       --launch-type FARGATE \
       --task-definition flexcollect-task \
       --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxx,subnet-yyyy],securityGroups=[sg-zzzz],assignPublicIp=ENABLED}" | cat
     ```

Optional: To pass parameters to the task, set the env var `FC_EVENT_JSON` as a container override when calling `run-task`. The runner will decode it and forward it to your handler.
