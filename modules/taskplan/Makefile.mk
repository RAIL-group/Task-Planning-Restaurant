help::
	@echo "Anticipatory Taskplanning in restaurant (ap-res):"
	@echo "  ap-res-gen-data	  Generate graph data from restaurants."
	@echo "  ap-res-eval-learned  Evaluates Anticipatory planner."
	@echo "  ap-res-eval-naive	  Evaluates Myopic planner."

AP_RES_BASENAME ?= restaurant
AP_RES_NUM_TRAINING_SEEDS ?= 28
AP_RES_NUM_TESTING_SEEDS ?= 12
AP_RES_NUM_EVAL_SEEDS ?= 1
EXPERIMENT_NAME = v0


ap-res-data-gen-seeds = \
	$(shell for ii in $$(seq 0 $$((0 + $(AP_RES_NUM_TRAINING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(AP_RES_BASENAME)/data_completion_logs/data_training_$${ii}.png"; done) \
	$(shell for ii in $$(seq 1500 $$((1500 + $(AP_RES_NUM_TESTING_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(AP_RES_BASENAME)/data_completion_logs/data_testing_$${ii}.png"; done) \

$(ap-res-data-gen-seeds): traintest = $(shell echo $@ | grep -Eo '(training|testing)' | tail -1)
$(ap-res-data-gen-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(ap-res-data-gen-seeds):
	@echo "Generating Data [$(AP_RES_BASENAME) | seed: $(seed) | $(traintest)"]
	@-rm -f $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/data_$(traintest)_$(seed)_*.csv
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/pickles
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/data_completion_logs
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan.scripts.gen_anticip_data \
		--current_seed $(seed) \
	 	--data_file_base_name data_$(traintest) \
		--save_dir /data/$(AP_RES_BASENAME)/ 

.PHONY: ap-res-gen-data
ap-res-gen-data: $(ap-res-data-gen-seeds)

ap-res-train-file = $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/logs/$(EXPERIMENT_NAME)/anticipategcn.pt 
$(ap-res-train-file): 
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/logs/$(EXPERIMENT_NAME)
	@$(DOCKER_PYTHON) -m taskplan.scripts.train \
		--num_steps 2000 \
		--learning_rate 0.01 \
		--learning_rate_decay_factor 0.5 \
		--epoch_size 1000 \
		--save_dir /data/$(AP_RES_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(AP_RES_BASENAME)/ 

.PHONY: ap-res-train-file
ap-res-train: $(ap-res-train-file)

ap-res-prep-seeds = \
	$(shell for ii in $$(seq 0 $$((0 + $(AP_RES_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/evaluation_no_$${ii}.png"; done)
$(ap-res-prep-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(ap-res-prep-seeds):
	@echo "Debugging Data [$(AP_RES_BASENAME) | seed: $(seed) | Debug"]
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/test_no_prep_myopic
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/test_prep_myopic
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan.scripts.evaluate_preparation \
		--save_dir /data/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/ \
		--current_seed $(seed)\
		--image_filename evaluation_$(seed).png \
		--logfile_name map_$(seed).txt \
		--network_file /data/restaurant/logs/v0/ap_beta-v0.pt

.PHONY: ap-res-prepare
ap-res-prepare: $(ap-res-prep-seeds)


 # Target for downloading sbert
.PHONY: download-sbert
download-sbert:
	@mkdir -p $(DATA_BASE_DIR)/sentence_transformers/
	@$(DOCKER_PYTHON) -m taskplan.scripts.sentence_bert

# Target for a demo
.PHONY: taskplan-demo
taskplan-demo:
	@$(DOCKER_PYTHON) -m taskplan.scripts.demo_pddl