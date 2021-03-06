.DEFAULT_GOAL := help

NOW = $(shell date '+%Y%m%d-%H%M%S')
HEAD_COMMIT = $(shell git rev-parse HEAD)
source: ## Push source.
	@kaggle datasets version -m $(HEAD_COMMIT)-$(NOW) -r zip

model: ## Push model.
	@$(MAKE) -C ds/models/ push

agent: ## Push agent.
	@$(MAKE) -C handyrl/envs/kaggle/geese/ push

submit: ## Push submission alpha kernel.
	@$(MAKE) -C ds/submit/ alpha_

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "\033[38;2;98;209;150m%-20s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
