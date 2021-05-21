.DEFAULT_GOAL := help

alpha_: ## Push alpha kernel
	@cd alpha && kaggle kernels push

dual_: ## Push dual kernel
	@cd dual && kaggle kernels push

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{printf "\033[38;2;98;209;150m%-20s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
