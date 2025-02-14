#!/bin/bash

python -m verl.trainer.main_ppo \
multi_processing=ray \
data.dataset_id=HuggingFaceH4/MATH-500 \
data.train_batch_size=4 \
data.val_batch_size=4 \
data.max_prompt_length=13000 \
data.max_response_length=13000 \
data.max_start_length=256 \
data.max_obs_length=200 \
data.shuffle_train_dataloader=True \
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
trainer.project_name=math_code \
trainer.experiment_name=run \
env.name=math_code \

# algorithm.adv_estimator={config['optimization']['adv_estimator']},
# actor_rollout_ref.model.path={config['model']['base_model']},
# actor_rollout_ref.model.enable_gradient_checkpointing={str(config['model']['gradient_checkpointing']).lower()},
# actor_rollout_ref.actor.optim.lr={config['optimization']['actor_lr']},
# actor_rollout_ref.actor.use_kl_loss={config['training']['use_kl_loss']},
# actor_rollout_ref.actor.ppo_mini_batch_size={config['training']['ppo_batch_size']},
# actor_rollout_ref.actor.ppo_micro_batch_size={config['training']['micro_batch_size']},
# actor_rollout_ref.rollout.log_prob_micro_batch_size={config['training']['micro_batch_size']},
# actor_rollout_ref.rollout.tensor_model_parallel_size={config['training']['rollout_tp_size']},
# actor_rollout_ref.rollout.gpu_memory_utilization={config['optimization']['gpu_memory_utilization']},
# actor_rollout_ref.ref.log_prob_micro_batch_size={config['training']['micro_batch_size']},
# critic.optim.lr={config['optimization']['critic_lr']},
# critic.model.path={config['model']['base_model']},
# critic.ppo_micro_batch_size={config['training']['micro_batch_size']},
# algorithm.kl_ctrl.kl_coef={config['optimization']['kl_coef']},
# algorithm.no_think_rl={config['training']['no_think_rl']},
# actor_rollout_ref.rollout.n_agent={config['training']['n_rollout']},
# trainer.logger={config['logging']['mode']},
# +trainer.val_before_train={str(config['trainer']['val_before_train']).lower()},
# trainer.default_hdfs_dir={config['trainer']['default_hdfs_dir'] or 'null'},
# trainer.save_freq={config['trainer']['save_freq']},
# trainer.test_freq={config['trainer']['test_freq']},
# trainer.total_epochs={config['training']['total_epochs']},
# env.name={config['env']['name']},
