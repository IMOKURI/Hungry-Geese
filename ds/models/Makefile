.DEFAULT_GOAL := help

NOW = $(shell date '+%Y%m%d-%H%M%S')
HEAD_COMMIT = $(shell git rev-parse HEAD)
MODEL_VERSION := $(shell ls -t1 ../../models/ | head -2 | grep -v latest)
push: ## Push models as dataset
	@cp ../../models/latest.pth ../../models/$(MODEL_VERSION) .
	@kaggle datasets version -m $(HEAD_COMMIT)-$(MODEL_VERSION)-$(NOW) -r zip

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "\033[38;2;98;209;150m%-20s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
