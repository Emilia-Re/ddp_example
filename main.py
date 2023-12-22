## main.py文件
import os

import torch
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn, optim

local_rank=int(os.environ["LOCAL_RANK"])

print(f"local rank {local_rank}")
# 新增：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 构造模型
device = torch.device("cuda", local_rank)
model = nn.Linear(10, 10).to(device)
# 新增：构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 前向传播
outputs = model(torch.randn(20, 10).to(local_rank))
labels = torch.randn(20, 10).to(local_rank)
loss_fn = nn.MSELoss()
loss_fn(outputs, labels).backward()
# 后向传播
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()
