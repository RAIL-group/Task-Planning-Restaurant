help::
	@echo "Multi agent anticipatory taskplanning in a restaurant setting (multi-ap):"

MA_AP_BASENAME ?= restaurant-multi
MA_AP_NUM_TRAINING_SEEDS ?= 5000
MA_AP_NUM_TESTING_SEEDS ?= 0
MA_AP_NUM_EVAL_SEEDS ?= 10
EXPERIMENT_NAME = beta-v0

# Target for a demo
.PHONY: multi-agent-demo
multi-agent-demo: build
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/ma_taskplan_demo/
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.demo_pddl \
		--output_image_file /data/$(MA_AP_BASENAME)/ma_taskplan_demo/ma_taskplan.png


ma-data-gen-seeds = \
	$(shell for ii in $$(seq 0 $$((0 + $(MA_AP_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(MA_AP_BASENAME)/data_completion_logs/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 1500 $$((1500 + $(MA_AP_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(MA_AP_BASENAME)/data_completion_logs/data_testing_$${ii}.png"; done) \

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
		--agent tiny \
	 	--data_file_base_name data_$(traintest) \
		--save_dir /data/$(MA_AP_BASENAME)/ 

.PHONY: ma-data-gen
ma-data-gen: $(ma-data-gen-seeds)

ma-train-file = $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/logs/$(EXPERIMENT_NAME)/anticipategcn.pt 
$(ma-train-file): 
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.train \
		--num_steps 5000 \
		--learning_rate 0.05 \
		--learning_rate_decay_factor 0.5 \
		--epoch_size 500 \
		--save_dir /data/$(MA_AP_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(MA_AP_BASENAME)/ 

.PHONY: ma-train
ma-train: $(ma-train-file)

.PHONY: ma-eval-demo
ma-eval-demo: 
	@echo "Evaluation Data"
	@mkdir -p $(DATA_BASE_DIR)/$(MA_AP_BASENAME)/results 
	@$(DOCKER_PYTHON) -m taskplan_multi.scripts.eval_demo \
		--current_seed 1 \
		--save_dir /data/$(MA_AP_BASENAME)/results \
		--tall_network /data/restaurant-multi-tall/logs/beta-v0/ap_tall.pt \
		--tiny_network /data/restaurant-multi-tiny/logs/beta-v0/ap_tiny.pt