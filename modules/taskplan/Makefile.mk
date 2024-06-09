help::
	@echo "Taskplanning in restaurant(lsp-tp):"
	@echo "  lsp-tp-gen-data	  Generate graph data from procthor maps."
	@echo "  lsp-tp-eval-learned  Evaluates learned planner."
	@echo "  lsp-tp-eval-known	  Evaluates known planner."
	@echo "  lsp-tp-eval-naive	  Evaluates naive planner."

LSP_TP_BASENAME ?= taskplan
LSP_TP_NUM_TRAINING_SEEDS ?= 200
LSP_TP_NUM_TESTING_SEEDS ?= 50
LSP_TP_NUM_EVAL_SEEDS ?= 2#100


# Target for a demo
.PHONY: taskplan-demo
taskplan-demo:
	@$(DOCKER_PYTHON) -m taskplan.scripts.demo_pddl

lsp-tp-seeds-naive = \
	$(shell for ii in $$(seq 7000 $$((7000 + $(LSP_TP_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_TP_BASENAME)/results/$(EXPERIMENT_NAME)/naive_$${ii}.png"; done)
$(lsp-tp-seeds-naive): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-tp-seeds-naive):
	@echo "Evaluating Data [$(LSP_TP_BASENAME) | seed: $(seed) | Naive"]
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_TP_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan.scripts.evaluate \
		--save_dir /data/$(LSP_TP_BASENAME)/results/$(EXPERIMENT_NAME) \
	 	--current_seed $(seed) \
	 	--image_filename naive_$(seed).png \
	 	--logfile_name naive_logfile.txt

.PHONY: lsp-tp-result-naive
lsp-tp-result-naive:
	@$(DOCKER_PYTHON) -m procgraph.scripts.result \
		--data_file /data/$(LSP_TP_BASENAME)/results/$(EXPERIMENT_NAME)/naive_logfile.txt \
		--output_image_file /data/$(LSP_TP_BASENAME)/results/naive_$(EXPERIMENT_NAME).png \
		--naive

.PHONY: lsp-tp-eval-naive
lsp-tp-eval-naive: $(lsp-tp-seeds-naive)
#	$(MAKE) lsp-tp-result-naive

lsp-tp-seeds-pddl = \
	$(shell for ii in $$(seq 7000 $$((7000 + $(LSP_TP_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(LSP_TP_BASENAME)/results/$(EXPERIMENT_NAME)/pddl_$${ii}.png"; done)
$(lsp-tp-seeds-pddl): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(lsp-tp-seeds-pddl):
	@echo "Evaluating Data [$(LSP_TP_BASENAME) | seed: $(seed) | PDDL"]
	@mkdir -p $(DATA_BASE_DIR)/$(LSP_TP_BASENAME)/results/$(EXPERIMENT_NAME)
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan.scripts.eval_pddl \
		--save_dir /data/$(LSP_TP_BASENAME)/results/$(EXPERIMENT_NAME) \
	 	--current_seed $(seed) \
	 	--image_filename pddl_$(seed).png \
	 	--logfile_name pddl_logfile.txt

.PHONY: eval-pddl
eval-pddl: $(lsp-tp-seeds-pddl)

# Target for downloading sbert
.PHONY: download-sbert
download-sbert:
	@mkdir -p $(DATA_BASE_DIR)/sentence_transformers/
	@$(DOCKER_PYTHON) -m taskplan.scripts.sentence_bert