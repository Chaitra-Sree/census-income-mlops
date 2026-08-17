def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.t3.medium",
    training_instance_type="ml.m5.large",
):
    """Gets a SageMaker ML Pipeline instance working with income data."""

    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # Pipeline parameters
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=1,
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )

    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/income.txt",
    )

    # ---------------- PREPROCESS ----------------

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-income-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )

    step_process = ProcessingStep(
        name="PreprocessincomeData",
        step_args=step_args,
    )

    # ---------------- TRAIN ----------------

    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/IncomeTrain"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/income-train",
        sagemaker_session=pipeline_session,
        role=role,
    )

    xgb_train.set_hyperparameters(
        max_depth="5",
        eta="0.2",
        gamma="4",
        min_child_weight="6",
        subsample="0.7",
        objective="binary:logistic",
        num_round="50",
    )

    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    step_train = TrainingStep(
        name="TrainincomeModel",
        step_args=step_args,
    )

    # ---------------- EVALUATE ----------------

    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-income-eval",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            ),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # ---------------- REGISTER MODEL ----------------

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    step_register = ModelStep(
        name="RegisterincomeModel",
        step_args=step_args,
    )

    # ---------------- CONDITION ----------------

    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=0.7,
    )

    step_cond = ConditionStep(
        name="CheckaccuracyEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )

    # ---------------- PIPELINE ----------------

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            model_approval_status,
            input_data,
        ],
        steps=[
            step_process,
            step_train,
            step_eval,
            step_cond,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline