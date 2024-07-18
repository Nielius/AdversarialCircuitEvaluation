SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export TAG=${TAG-$(git rev-parse --short HEAD)}
export COMMIT_FULL=$(git rev-parse HEAD)
export BRANCH_NAME=$(git branch --show-current)

echo "Tag: ${TAG}"
echo "Commit: ${COMMIT_FULL}"
echo "Branch: ${BRANCH_NAME}"

envsubst < "$SCRIPT_DIR/kaniko-build.yaml" |  kubectl create -f -

until kubectl logs -f job/build-image
do
  echo "No pods found yet. Sleeping..."
  sleep 5
  echo "Trying again."
done

# Here GIT_TOKEN should be a PAT that can read the repo (only needed if the repo is private)


# Try to run locally
# doesn't work, but it can't find the docker credentials to push, unfortunately
#      docker run gcr.io/kaniko-project/executor:latest \
#        --context git://${GIT_TOKEN}@github.com/FlyingPumba/circuits-benchmark.git#refs/heads/${BRANCH_NAME}#${COMMIT_FULL} \
#        --destination=ghcr.io/nielius/cb:${TAG} \
#        --destination=ghcr.io/nielius/cb:latest

