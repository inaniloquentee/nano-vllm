### 一、 核心功能与架构定位

`Scheduler` 在大模型推理框架中的定位是**“中央调度室”**。它的主要职责是回答一个核心问题：**在下一次模型前向传播（Forward Pass）时，我应该把哪些请求（Sequences）送进 GPU？**

它需要在这个过程中平衡两个目标：
1. **最大化 GPU 利用率：** 尽可能把 `batch_size` 塞满，提高吞吐量。
2. **保证显存安全：** 不能导致 Out of Memory (OOM)，因此必须实时与 `BlockManager` 交互，检查 KV Cache 的容量。

### 二、 关键数据结构

```python
self.waiting: deque[Sequence] = deque()
self.running: deque[Sequence] = deque()
```
* **状态机:** 调度器维护了两个核心队列。
    * `waiting`: 存放刚刚提交进来的新请求（处于 `WAITING` 状态），或者被**抢占 (Preempted)** 退回来的老请求。
    * `running`: 存放正在生成 Token 的请求（处于 `RUNNING` 状态）。
* 为什么用 `deque`？ 因为调度涉及频繁的头部出队（`popleft`）和尾部追加，`deque` 提供 $O(1)$ 的时间复杂度。

### 三、 核心工作流解析 (`schedule` 方法)

`schedule` 方法是整个调度器的心脏，它会在每次 Engine `step()` 时被调用。其逻辑明确地划分为两个阶段：**Prefill（预填充阶段）** 和 **Decode（解码阶段）**。在一次 `schedule` 调用中，**这两种阶段互斥，只能执行其一**。

#### 1. 优先尝试 Prefill（预填充）
这是处理新用户请求的阶段。

```python
# prefill
while self.waiting and num_seqs < self.max_num_seqs:
    seq = self.waiting[0]
    # 两个硬性限制：1. Batch 内总 Token 数不能超上限。 2. BlockManager 必须有足够的物理块分配给它。
    if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
        break
    
    # 满足条件，开始分配并转移状态
    num_seqs += 1
    self.block_manager.allocate(seq) 
    
    # 注意这里扣除了 cached_tokens，如果命中了前缀缓存，实际参与计算的 token 数会减少！
    num_batched_tokens += len(seq) - seq.num_cached_tokens 
    
    seq.status = SequenceStatus.RUNNING
    self.waiting.popleft()
    self.running.append(seq)
    scheduled_seqs.append(seq)

# 如果成功调度了哪怕一个处于 Waiting 的请求，本次 step 就是 Prefill 阶段，直接返回！
if scheduled_seqs:
    return scheduled_seqs, True
```

**为什么 Prefill 优先级这么高且独立？**
在多数推理框架中，Prefill 计算是**计算密集型**的（大矩阵乘法），而 Decode 是**显存受限型**的（频繁读取 KV Cache）。将它们混合在一个 Batch 中处理不仅底层算子（如 FlashAttention）难以支持，而且会导致性能严重退化（Decode 序列会被迫等待极慢的 Prefill 序列）。

#### 2. 如果没有 Prefill，则执行 Decode（解码）
如果 `waiting` 队列为空，或者由于显存/Token 数限制无法调入任何新请求，调度器就会转而处理 `running` 队列中的老请求，为它们生成下一个 Token。

```python
# decode
while self.running and num_seqs < self.max_num_seqs:
    seq = self.running.popleft()
    
    # 核心难点：检查能否为当前序列追加一个 Token 的 KV Cache
    while not self.block_manager.can_append(seq):
        # 如果显存不够了，触发【抢占 (Preemption)】机制
        if self.running:
            # 牺牲策略：把 running 队列最后面的请求（通常是最晚进来的）强行终止并退回 waiting 队列
            self.preempt(self.running.pop())
        else:
            # 极端情况：哪怕只剩当前这一个请求了，显存还是不够，只能抢占自己（这通常意味着 max_tokens 设置得太大了）
            self.preempt(seq)
            break
    else: # 注意这是 while...else 结构。如果上面的 while 正常结束（显存够了），执行这里
        num_seqs += 1
        self.block_manager.may_append(seq) # 通知 BlockManager：我要追加了，可能需要新块，也可能要算 Hash 了
        scheduled_seqs.append(seq)

# 恢复 running 队列（因为前面被 popleft 清空了，且去除了被抢占的请求）
# extendleft 配合 reversed 保持了请求在队列中的相对优先级顺序
self.running.extendleft(reversed(scheduled_seqs))
return scheduled_seqs, False
```

### 四、 关键机制：抢占 (Preemption)

这是为了防止 OOM 保护系统的最后一道防线。

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    # 核心动作：彻底释放该序列当前占用的所有 KV Cache 物理块！
    self.block_manager.deallocate(seq) 
    # 把请求塞回 waiting 队列的头部，保证下次有显存时优先处理它
    self.waiting.appendleft(seq)
```

**深入思考：为什么释放了 KV Cache，还能把它放回 Waiting 队列？**
因为当它下次再被调度时，它会重新走一次 **Prefill 流程**！
这就是著名的 **Recomputation（重计算）** 策略。虽然扔掉了之前算好的 KV Cache 很浪费算力，但这在极端显存压力下是必要的妥协（空间换时间）。

> *注：vLLM 中抢占除了 Recomputation 外，还有一种 Swapping (换出到 CPU 内存) 策略。这个轻量级框架为了简洁，只实现了 Recomputation。*

### 五、 后处理 (`postprocess`)

在引擎 (`LLMEngine`) 拿到模型吐出的新 token IDs 后，需要通知调度器更新状态。

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id) # 将新 Token 加入序列记录
        
        # 结束条件判断：
        # 1. 模型输出了 EOS (End Of Sentence) token。
        # 2. 或者生成的 Token 数量达到了用户设置的上限。
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            # 重点：一旦结束，立刻通知 BlockManager 归还所有的显存块，造福其他请求！
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

### 六、 总结与面试考察点

这份代码非常精炼地展示了 Continuous Batching 的核心逻辑。如果你在准备相关面试，可以关注以下几点：

1. **Prefill 和 Decode 为什么要分开调度？** (算子特性、计算密集 vs 显存受限)
2. **什么是 Continuous Batching (连续批处理)？** 在 `postprocess` 中，一旦有序列完成，它的显存就会立刻被释放。在下一个 `step` 时，`waiting` 队列里的新请求马上就能见缝插针地加入进来，而不需要像传统批处理那样等待整个 batch 所有请求都跑完。这极大地提高了吞吐量。
3. **当显存不足时，调度器是如何处理的？** (解释 Preemption 机制，并对比 Recomputation 和 Swapping 的优劣)。
4. **`extendleft(reversed(scheduled_seqs))` 这一句有什么妙用？** (保持队列的公平性和优先级)。
