.DEFAULT_GOAL := help

base_: ## Push base kernel
	@cd base && kaggle kernels push

ensemble_: ## Push ensemble kernel
	@cd ensemble && kaggle kernels push

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "\033[38;2;98;209;150m%-20s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
