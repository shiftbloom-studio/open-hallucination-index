# Rollback a deploy

If `release.yml` deploys a bad image, rollback automatically via the `if: failure()`
step. If it didn't (e.g., the image works but the behavior is wrong), manual rollback:

## Option A — re-tag a known-good commit

release.yml triggers on tag push. To redeploy a prior version:

```bash
# Delete the bad tag locally and remotely (ONLY if you own the tag)
git tag -d v0.2.1
git push origin :refs/tags/v0.2.1   # delete remote (CAREFUL — ensure no-one else used it)

# Re-tag a known-good commit
git tag v0.2.1-rollback <commit-sha-that-was-last-known-good>
git push origin v0.2.1-rollback
```

release.yml fires on the new tag push; Lambda picks up the corresponding image.

## Option B — direct Lambda update (emergency only)

```bash
# Get the prior digest
PREV_DIGEST=$(aws ecr describe-images --repository-name ohi-api \
  --query 'reverse(sort_by(imageDetails[?imageTags!=null],&imagePushedAt))[1].imageDigest' \
  --output text)
REPO_URL=$(aws ecr describe-repositories --repository-names ohi-api --query 'repositories[0].repositoryUri' --output text)

aws lambda update-function-code \
  --function-name ohi-api \
  --image-uri "${REPO_URL}@${PREV_DIGEST}"
```

Note: this puts Terraform state out-of-sync. Run `terraform plan` on compute
afterward and re-apply with the correct `image_tag` to re-sync.
