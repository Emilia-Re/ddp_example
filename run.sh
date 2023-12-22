
# 改变：使用torch.distributed.launch启动DDP模式，
#   其会给main.py一个local_rank的参数。这就是之前需要"新增:从外面得到local_rank参数"的原因
#现在torch.distributed.launch已经不建议继续使用
#过去可以使用python -m torch.distributed.launch --nproc_per_node 4 main.py


#torchrun
#    --standalone
#    --nnodes=1
#    --nproc-per-node=$NUM_TRAINERS
#    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

#参考https://pytorch.org/docs/stable/elastic/run.html#launcher-api

torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py


