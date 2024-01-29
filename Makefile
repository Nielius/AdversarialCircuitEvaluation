APPLICATION_NAME ?= acdc
APPLICATION_URL ?= ghcr.io/alignmentresearch/${APPLICATION_NAME}
DOCKERFILE ?= Dockerfile

export APPLICATION_NAME
export APPLICATION_URL
export DOCKERFILE

# By default, the base tag is the short git hash
BASE_TAG ?= $(shell git rev-parse --short HEAD)
COMMIT_FULL ?= $(shell git rev-parse HEAD)
BRANCH_NAME ?= $(shell git branch --show-current)

# Release tag or latest if not a release
RELEASE_TAG ?= latest

default: release-${BASE_TAG}

# Re-usable functions
.PHONY: build
build:
	docker pull ${APPLICATION_URL}:${TAG} || true
	docker build --tag "${APPLICATION_URL}:${TAG}" \
		--build-arg "APPLICATION_NAME=${APPLICATION_NAME}" \
		--target ${TARGET} \
		-f "${DOCKERFILE}" .

.PHONY: build-main-% push-% release-%

build-%: ${DOCKERFILE}
	$(MAKE) build "TAG=$*" TARGET=main

push-%: build-%
	docker push "${APPLICATION_URL}:$*"

release-%: push-%
	docker tag ${APPLICATION_URL}:$* ${APPLICATION_URL}:${RELEASE_TAG}
	docker push ${APPLICATION_URL}:${RELEASE_TAG}

.PHONY: format check-codestyle
format:
	# Sort imports. (F401) delete unused imports
	ruff --fix .
	# Reformat using black
	black .

check-codestyle:
	# Sort imports. (F401) delete unused imports
	ruff --select I --select F401 . --fix
	# Reformat using black
	black --check .

test:
	pytest tests -W error

.PHONY: remote-build
remote-build:
	git push
	# Kaniko didn't correctly load the new submodule, unless one specifies a branch in which that module is available
	python -c "print(open('k8s/kaniko-build.yaml').read().format(COMMIT='${BASE_TAG}', BRANCH_NAME='${BRANCH_NAME}', COMMIT_FULL='${COMMIT_FULL}'))" | kubectl create -f -
