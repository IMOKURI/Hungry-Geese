.DEFAULT_GOAL := help

NOW = $(shell date '+%Y%m%d-%H%M%S')
HEAD_COMMIT = $(shell git rev-parse HEAD)
push: ## Push models as dataset
	@cp ../../models/latest.pth .
	@kaggle datasets version -m $(HEAD_COMMIT)-$(NOW) -r zip

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "\033[38;2;98;209;150m%-20s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
