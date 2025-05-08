import torch.nn as nn

def get_layer_shards(model: nn.Module, num_shards: int, pattern: str = "sequential"):
    """
    按层将 model.parameters() 划分成 num_shards 份：
      - sequential: 连续切分
      - stride: 跳跃切分
    返回 List[List[nn.Parameter]]，长度 = num_shards
    """
    # 1. 把每一“层”按顺序收集成一个大列表
    layers = []
    # embed
    layers.append(list(model.model.embed_tokens.parameters()))
    # transformer 层
    for lyr in model.model.layers:
        layers.append(list(lyr.parameters()))
    # norm + lm_head
    layers.append(list(model.model.norm.parameters()))
    layers.append(list(model.lm_head.parameters()))
    # 2. 根据 pattern 计算每个 shard 包含哪些 layer idx
    L = len(layers)
    shards = [[] for _ in range(num_shards)]
    if pattern == "sequential":
        per = (L + num_shards - 1) // num_shards
        for i, layer_params in enumerate(layers):
            shard_id = min(i // per, num_shards - 1)
            shards[shard_id].extend(layer_params)
    elif pattern == "stride":
        for i, layer_params in enumerate(layers):
            shard_id = i % num_shards
            shards[shard_id].extend(layer_params)
    else:
        raise ValueError(f"Unknown pattern {pattern}")
    return shards, L