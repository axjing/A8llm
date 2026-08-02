import os
from datetime import timedelta
import torch
import torch.nn as nn
import torch.distributed as dist

 
def init_dist():
    dist.init_process_group(backend='nccl',timeout=timedelta(minutes=30))
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    # torch.cuda.manual_seed(0)
   
def is_master():
    return dist.get_rank()==0 if is_ddp_initialized() else True

def get_world_size():
    return dist.get_world_size() if is_ddp_initialized() else 1

def get_rank():
    return dist.get_rank() if is_ddp_initialized() else 0

_CPU_GROUP = None


def _get_cpu_group():
    """Lazily create a host-RAM gloo group used for object gather.

    Must be called collectively by all ranks (it is only invoked from the
    collective ``distributed_gather``).
    """
    global _CPU_GROUP
    if _CPU_GROUP is None:
        _CPU_GROUP = dist.new_group(backend="gloo")
    return _CPU_GROUP


def distributed_gather(obj):
    """
    无需分配临时CUDA缓冲区，从所有进程编号中收集**任意**可序列化对象。返回列表格式为[编号0对象、编号1对象……]。

    若分布式训练模块（torch.distributed）未完成初始化，则仅返回当前单个进程编号对应的对象列表。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    
    result=[None] * dist.get_world_size()
    dist.all_gather_object(result,obj,group=_get_cpu_group()) # CPU path
    
    return result

def distributed_mean_scalar(x:float|int)->float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)
    
    t=torch.tensor(x,device=torch.cuda.current_device(),dtype=torch.float32)
    dist.all_reduce(t,op=dist.ReduceOp.SUM)
    
    t/=dist.get_world_size()
    return t.item()

def wrap_model(model):
    local_rank=int(os.environ['LOCAL_RANK'])
    return nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
	env = os.environ.get("NANOCHAT_DTYPE")
	if env is not None:
		return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
	if torch.cuda.is_available():
		# bf16 requires SM 80+ (Ampere: A100, A10, etc.)
		# Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
		capability = torch.cuda.get_device_capability()
		if capability >= (8, 0):
			return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
		# fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
		# Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
		return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
	return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

def is_ddp_requested() -> bool:
	"""
	True if launched by torchrun (env present), even before init.
	Used to decide whether we *should* initialize a PG.
	"""
	return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
	"""
	True if torch.distributed is available and the process group is initialized.
	Used at cleanup to avoid destroying a non-existent PG.
	"""
	return dist.is_available() and dist.is_initialized()

def get_dist_info():
	if is_ddp_requested():
		# We rely on torchrun's env to decide if we SHOULD init.
		# (Initialization itself happens in compute init.)
		assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
		ddp_rank = int(os.environ['RANK'])
		ddp_local_rank = int(os.environ['LOCAL_RANK'])
		ddp_world_size = int(os.environ['WORLD_SIZE'])
		return True, ddp_rank, ddp_local_rank, ddp_world_size
	else:
		return False, 0, 0, 1

def autodetect_device_type():
	# prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
	if torch.cuda.is_available():
		device_type = "cuda"
	elif torch.backends.mps.is_available():
		device_type = "mps"
	else:
		device_type = "cpu"
	print(f"Autodetected device type: {device_type}")
	return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
	"""Basic initialization that we keep doing over and over, so make common."""

	assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
	if device_type == "cuda":
		assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
	if device_type == "mps":
		assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

	# Reproducibility
	# Note that we set the global seeds here, but most of the code uses explicit rng objects.
	# The only place where global rng might be used is nn.Module initialization of the model weights.
	torch.manual_seed(42)
	if device_type == "cuda":
		torch.cuda.manual_seed(42)
	# skipping full reproducibility for now, possibly investigate slowdown later
	# torch.use_deterministic_algorithms(True)

	# Precision
	if device_type == "cuda":
		torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls, see https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

	# Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
	is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
	if is_ddp_requested and device_type == "cuda":
		device = torch.device("cuda", ddp_local_rank)
		torch.cuda.set_device(device)  # make "cuda" default to this device
		dist.init_process_group(backend="nccl", device_id=device)
		dist.barrier()
	else:
		device = torch.device(device_type) # mps|cpu

	if ddp_rank == 0:
		print(f"Distributed world size: {ddp_world_size}")

	return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def destory_ddp_process_group():
	"""
	清理分布式训练环境，在脚本退出前销毁进程组
	
	该函数是 compute_init 的配套函数，用于在脚本退出前清理资源。
	如果分布式数据并行（DDP）已初始化，则销毁进程组。
	
	Returns:
		None
	"""
	if is_ddp_initialized():
		dist.destroy_process_group()
  

# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    print(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')
