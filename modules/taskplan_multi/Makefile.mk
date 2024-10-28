help::
	@echo "Multi agent anticipatory taskplanning in a restaurant setting (multi-ap):"

MUL_AP_BASENAME ?= restaurant-multi
EXPERIMENT_NAME = beta-v0

# Target for a demo
.PHONY: multi-agent-demo
multi-agent-demo:
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.demo_pddl