.DEFAULT_GOAL := help

model: ## Push model.
	@$(MAKE) -C ds/models/ push

eval: ## Push evaluation kernel.
	@$(MAKE) -C ds/eval/ push

submit: ## Push submission kernel.
	@$(MAKE) -C ds/submit/ push

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "\033[38;2;98;209;150m%-20s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
