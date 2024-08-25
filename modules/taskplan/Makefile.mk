help::
	@echo "Anticipatory Taskplanning in restaurant (ap-res):"
	@echo "  ap-res-gen-data	  Generate graph data from restaurants."
	@echo "  ap-res-eval-learned  Evaluates Anticipatory planner."
	@echo "  ap-res-eval-naive	  Evaluates Myopic planner."

AP_RES_BASENAME ?= restaurant
AP_RES_NUM_TRAINING_SEEDS ?= 20
AP_RES_NUM_TESTING_SEEDS ?= 0
AP_RES_NUM_EVAL_SEEDS ?= 10
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
	@-rm -f $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/data_$(traintest)_$(seed).csv
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
		--num_steps 10000 \
		--learning_rate 0.02 \
		--learning_rate_decay_factor 0.5 \
		--epoch_size 2500 \
		--save_dir /data/$(AP_RES_BASENAME)/logs/$(EXPERIMENT_NAME) \
		--data_csv_dir /data/$(AP_RES_BASENAME)/ 

.PHONY: ap-res-train-file
ap-res-train: $(ap-res-train-file)

ap-res-prep-seeds = \
	$(shell for ii in $$(seq 1 $$((1 + $(AP_RES_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/evaluation_no_$${ii}.png"; done)
$(ap-res-prep-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(ap-res-prep-seeds):
	@echo "Debugging Data [$(AP_RES_BASENAME) | seed: $(seed) | Debug"]
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/no_prep_myopic
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/prep_myopic
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan.scripts.generate_prepared_state \
		--save_dir /data/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/ \
		--current_seed $(seed)\
		--image_filename evaluation_$(seed).png \
		--logfile_name $(seed).txt \
		--network_file /data/restaurant/logs/v0/ap_beta-v2.pt

.PHONY: ap-res-prepare
ap-res-prepare: $(ap-res-prep-seeds)


ap-res-ant-seeds = \
	$(shell for ii in $$(seq 0 $$((0 + $(AP_RES_NUM_EVAL_SEEDS) - 1))); \
		do echo "$(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/evaluation_no_$${ii}.png"; done)
$(ap-res-ant-seeds): seed = $(shell echo $@ | grep -Eo '[0-9]+' | tail -1)
$(ap-res-ant-seeds):
	@echo "Anticipatory Data [$(AP_RES_BASENAME) | seed: $(seed) | Debug"]
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/no_prep_myopic
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/prep_myopic
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/no_prep_ap
	@mkdir -p $(DATA_BASE_DIR)/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/prep_ap
	@$(call xhost_activate)
	@$(DOCKER_PYTHON) -m taskplan.scripts.evaluate_anticipation \
		--save_dir /data/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/ \
		--current_seed $(seed)\
		--image_filename evaluation_$(seed).png \
		--logfile_name $(seed).txt \
		--network_file /data/restaurant/logs/v0/ap_beta-v2.pt

.PHONY: ap-res-anticipation
ap-res-anticipation: $(ap-res-ant-seeds)


.PHONY: ap-res-result
ap-res-result:
	@$(DOCKER_PYTHON) -m taskplan.scripts.result \
	--save_dir /data/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/ \


.PHONY: ap-res-compare-expected-cost
ap-res-compare-expected-cost:
	@$(DOCKER_PYTHON) -m taskplan.scripts.compare_expected_cost \
	--save_dir /data/$(AP_RES_BASENAME)/results/$(EXPERIMENT_NAME)/ \

 # Target for downloading sbert
.PHONY: download-sbert
download-sbert:
	@mkdir -p $(DATA_BASE_DIR)/sentence_transformers/
	@$(DOCKER_PYTHON) -m taskplan.scripts.sentence_bert

# Target for a demo
.PHONY: taskplan-demo
taskplan-demo:
	@$(DOCKER_PYTHON) -m taskplan.scripts.demo_pddl