# train a character-level GPT for tianlong
out_dir = 'out-tianlong'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'tianlong-char'
wandb_run_name = 'mini-gpt-tianlong'

dataset = 'tianlong'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# model parameters
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

compile = False