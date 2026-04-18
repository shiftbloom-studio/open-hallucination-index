"""Wave 3 Stream G.1 — Lambda image rollback utility.

Rolls ``ohi-api`` Lambda back to the previous ECR image by:

1. Querying the current image URI via ``aws lambda get-function
   --query 'Code.ResolvedImageUri'`` (Decision I — returns the
   digest-form URI, NOT ``get-function-configuration --query
   'ImageUri'`` which returns the literal string ``"None"`` for
   image-backed Lambdas).
2. Listing the ECR image tags for ``ohi-api`` sorted by pushed-at
   descending; picking the second-newest tag (the one before
   ``:prod``).
3. Calling ``update-function-code`` with that previous image.
4. Polling ``LastUpdateStatus`` until not ``InProgress``.
5. Writing the chosen previous URI to stdout for the caller's audit
   log.

Used by:
* ``v2-post-deploy-verify.yml`` as the auto-rollback on blocking
  verify failure.
* ``v2-auto-rollback.yml`` for alarm-triggered + manual rollbacks.

Idempotent: if run twice in a row, the second run will roll back to
the pre-rollback digest (i.e. restore the originally-deployed image).
If that's not desired, pass ``--dry-run`` first to inspect the chosen
target.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import boto3

logger = logging.getLogger("rollback_lambda")


def _resolve_previous_image_uri(
    ecr_client, lambda_client, function_name: str
) -> tuple[str, str]:
    """Return ``(previous_image_uri, reason)``.

    The digest of the currently-running Lambda image is the anchor.
    We list ECR images sorted by ``imagePushedAt`` desc and pick the
    first one whose digest ≠ current. That gives us the last image
    that was deployed before the current ``:prod`` retag.
    """
    current = lambda_client.get_function(FunctionName=function_name)["Code"][
        "ResolvedImageUri"
    ]
    logger.info("Current image: %s", current)
    current_digest = current.split("@sha256:")[-1] if "@sha256:" in current else None

    repo_name = "ohi-api"
    paginator = ecr_client.get_paginator("describe_images")
    images = []
    for page in paginator.paginate(repositoryName=repo_name):
        images.extend(page.get("imageDetails", []))
    if not images:
        raise RuntimeError(f"No ECR images found for {repo_name}")
    images.sort(key=lambda d: d.get("imagePushedAt"), reverse=True)

    for img in images:
        digest = img.get("imageDigest", "")
        if current_digest and current_digest in digest:
            continue
        registry = (
            f"{lambda_client.meta.client_config.user_agent_extra or ''}"
            if False
            else None
        )
        # Registry ID + URL reconstructed from repo URI rather than the
        # caller identity (avoids an extra STS call in IAM-constrained
        # environments).
        repo_uri = _describe_repo_uri(ecr_client, repo_name)
        return f"{repo_uri}@{digest}", "previous-tag-by-pushed-at"

    raise RuntimeError("No previous image found — all ECR images match current digest")


def _describe_repo_uri(ecr_client, repo_name: str) -> str:
    resp = ecr_client.describe_repositories(repositoryNames=[repo_name])
    repos = resp.get("repositories") or []
    if not repos:
        raise RuntimeError(f"Repository {repo_name} not found")
    return repos[0]["repositoryUri"]


def _wait_until_successful(lambda_client, function_name: str, timeout_s: float = 120.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cfg = lambda_client.get_function_configuration(FunctionName=function_name)
        status = cfg.get("LastUpdateStatus", "Unknown")
        if status != "InProgress":
            return status
        time.sleep(3.0)
    raise TimeoutError(f"LastUpdateStatus did not leave InProgress within {timeout_s}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", required=True, help="Lambda function name (e.g. ohi-api)")
    parser.add_argument("--reason", required=True, help="Audit reason for the rollback")
    parser.add_argument("--dry-run", action="store_true", help="Print target URI without applying")
    parser.add_argument("--region", default="eu-central-1")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    lambda_client = boto3.client("lambda", region_name=args.region)
    ecr_client = boto3.client("ecr", region_name=args.region)

    previous_uri, source = _resolve_previous_image_uri(ecr_client, lambda_client, args.function)
    logger.info("Rollback target: %s (source=%s)", previous_uri, source)
    logger.info("Reason: %s", args.reason)

    if args.dry_run:
        logger.info("Dry-run — not invoking update-function-code")
        print(previous_uri)
        return 0

    resp = lambda_client.update_function_code(
        FunctionName=args.function, ImageUri=previous_uri
    )
    logger.info("update-function-code → CodeSha256=%s", resp.get("CodeSha256"))
    status = _wait_until_successful(lambda_client, args.function)
    if status != "Successful":
        logger.error("Rollback failed — LastUpdateStatus=%s", status)
        return 1

    logger.info("Rollback successful — Lambda now running %s", previous_uri)
    print(previous_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
