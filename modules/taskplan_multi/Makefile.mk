help::
	@echo "Multi agent anticipatory taskplanning in a restaurant setting (multi-ap):"

MA_AP_BASENAME ?= exp-v0
MA_AP_NUM_TRAINING_SEEDS ?= 0
MA_AP_NUM_TESTING_SEEDS ?= 0
MA_AP_NUM_EVAL_SEEDS ?= 3
EXPERIMENT_NAME = beta-v0
EXP_NUM = 1

# Target for a demo
.PHONY: multi-agent-demo
multi-agent-demo: build
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/ma_taskplan_demo/
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.demo_pddl \
		--output_image_file /data/$(MA_AP_BASENAME)/ma_taskplan_demo/task_plan_myopic.png \
		--save_dir /data/$(MA_AP_BASENAME)/results/$(EXP_NUM) \
		--cook_network /data/models/ap_cook.pt \
		--cleaner_network /data/models/ap_cleaner.pt \
		--server_network /data/models/ap_server.pt


ma-data-gen-seeds = \
	$(shell for ii in $$(seq 0 $$((0 + $(MA_AP_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(MA_AP_BASENAME)/data_completion_logs/data_training_$${ii}.png"; done) \

$(ma-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(ma-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(ma-data-gen-seeds):
	@echo "Generating Data [$(MA_AP_BASENAME) | seed: $(seed) | $(traintest)"]
	@-rm -f $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/data_$(traintest)_$(seed).csv
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/pickles
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/data_completion_logs
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.gen_data \
		--current_seed $(seed) \
		--agent server_bot \
	 	--data_file_base_name data_$(traintest) \
		--save_dir /data/$(MA_AP_BASENAME)/ 

.PHONY: ma-data-gen
ma-data-gen: $(ma-data-gen-seeds)

ma-train-file = $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/logs/$(EXPERIMENT_NAME)/anticipategcn.pt 
$(ma-train-file): 
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.train \
		--num_steps 80 \
		--learning_rate 0.01 \
		--learning_rate_decay_factor 0.99 \
		--epoch_size 200 \
		--agent cleaner \
		--save_dir /data/$(MA_AP_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(MA_AP_BASENAME)/ 

.PHONY: ma-train
ma-train: $(ma-train-file)

ma-eval-seeds := \
	$(shell for ii in $$(seq 10 $$((10 + $(MA_AP_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(MA_AP_BASENAME)/figure/eval_$${ii}.png"; done)

$(ma-eval-seeds): eval_seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(ma-eval-seeds): 
	@echo "Evaluation Data"
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/figure
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/results
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/results/$(eval_seed)
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.eval_demo \
		--current_seed $(eval_seed) \
		--save_dir /data/$(MA_AP_BASENAME)/results/$(eval_seed) \
		--cook_network /data/models/ap_cook.pt \
		--cleaner_network /data/models/ap_cleaner.pt \
		--server_network /data/models/ap_server.pt

.PHONY: ma-eval-demo
ma-eval-demo: $(ma-eval-seeds)

.PHONY: ma-result-demo
ma-result-demo: 
	@echo "Result"
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/results/figure
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.results_demo \
		--save_dir /data/$(MA_AP_BASENAME)/results/ \