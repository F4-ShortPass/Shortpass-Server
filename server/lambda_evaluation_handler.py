"""
AWS Lambda entrypoint for running the evaluation pipeline.

Expected event payload (SQS, EventBridge, or direct invoke):
{
  "company_id": 1,
  "job_id": 1,
  "applicant_id": 1,
  "interview_id": 123
}

Environment:
- USE_AWS_S3=true
- S3_BUCKET_NAME, AWS_REGION set
- AWS credentials via role

This reuses the existing EvaluationPipelineService and S3 selector without
modifying MAS logic or prompts.
"""
from typing import Any, Dict

from services.s3_service_factory import get_s3_service
from services.evaluation_pipeline_service import EvaluationPipelineService


def _extract(key: str, event: Dict[str, Any]) -> Any:
    """
    Small helper to read key from event or nested detail.
    """
    if key in event:
        return event[key]
    detail = event.get("detail", {}) if isinstance(event, dict) else {}
    return detail.get(key)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    company_id = _extract("company_id", event)
    job_id = _extract("job_id", event)
    applicant_id = _extract("applicant_id", event)
    interview_id = _extract("interview_id", event)

    if None in (company_id, job_id, applicant_id, interview_id):
        return {
            "statusCode": 400,
            "body": {
                "message": "Missing required ids (company_id, job_id, applicant_id, interview_id)"
            }
        }

    s3 = get_s3_service()
    pipeline = EvaluationPipelineService(s3_service=s3)
    result = pipeline.run_pipeline(
        company_id=int(company_id),
        job_id=int(job_id),
        applicant_id=int(applicant_id),
        interview_id=int(interview_id),
    )

    return {"statusCode": 200, "body": result}
