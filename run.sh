
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

#从torch.distributed.launch 改为torchrun
#python -m torch.distributed.launch --use-env train_script.py 改为
# torchrun train_script.py

#代码内部修改：
#原先
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("--local-rank", type=int)
#args = parser.parse_args()
#local_rank = args.local_rank

#后来
#import os
#local_rank = int(os.environ["LOCAL_RANK"])


torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py


