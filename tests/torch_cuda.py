from typing import Any

import torch


class PyTorchGPUChecker:
    """
    PyTorch GPU 显卡支持检测工具类
    功能：CUDA可用性检测、显卡信息查询、显存查看、GPU运算测试、设备获取
    """

    def __init__(self) -> None:
        # 初始化基础信息
        self.pytorch_version = torch.__version__
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.cuda_version = torch.version.cuda if self.cuda_available else "未支持"

    def get_single_gpu_info(self, device_id: int = 0) -> dict[str, Any]:
        """
        获取单张显卡的详细信息
        :param device_id: 显卡索引，默认0
        :return: 显卡信息字典
        """
        if not self.cuda_available or device_id >= self.device_count:
            return {"错误": "显卡不存在或CUDA不可用"}

        # 显卡核心信息
        info = {
            "显卡名称": torch.cuda.get_device_name(device_id),
            "计算能力": torch.cuda.get_device_capability(device_id),
            "总显存(GB)": round(
                torch.cuda.get_device_properties(device_id).total_memory / 1024**3, 2
            ),
            "当前已用显存(GB)": round(
                torch.cuda.memory_allocated(device_id) / 1024**3, 2
            ),
            "当前空闲显存(GB)": round(
                torch.cuda.memory_reserved(device_id) / 1024**3, 2
            ),
        }
        return info

    def get_all_gpus_info(self) -> list[dict[str, Any]]:
        """获取所有显卡的信息列表"""
        return [self.get_single_gpu_info(i) for i in range(self.device_count)]

    def test_gpu_compute(self, device_id: int = 0) -> str:
        """
        测试GPU是否能正常运算（最真实的可用性验证）
        :return: 测试结果
        """
        if not self.cuda_available:
            return "❌ CUDA 不可用，无法测试"

        try:
            # 创建张量并迁移到GPU运算
            device = torch.device(f"cuda:{device_id}")
            a = torch.tensor([1.0, 2.0, 3.0], device=device)
            b = torch.tensor([4.0, 5.0, 6.0], device=device)
            c = a + b  # GPU运算
            return f"✅ GPU运算测试成功 | 运算结果: {c} | 运行设备: {c.device}"
        except (RuntimeError, OSError) as e:
            return f"❌ GPU运算失败: {e!s}"

    def get_default_device(self) -> torch.device:
        """获取默认训练设备（优先GPU，否则CPU），直接用于模型训练"""
        return torch.device("cuda:0" if self.cuda_available else "cpu")

    def print_full_report(self) -> None:
        """打印完整的GPU检测报告（格式化输出，直接看结果）"""
        print("=" * 60)
        print("📊 PyTorch GPU 显卡支持检测报告")
        print("=" * 60)
        print(f"🔹 PyTorch 版本: {self.pytorch_version}")
        print(f"🔹 CUDA 支持状态: {'✅ 可用' if self.cuda_available else '❌ 不可用'}")
        print(f"🔹 CUDA 版本: {self.cuda_version}")
        print(f"🔹 检测到 NVIDIA 显卡数量: {self.device_count} 张")
        print("-" * 60)

        # 打印所有显卡详情
        if self.cuda_available and self.device_count > 0:
            for i, info in enumerate(self.get_all_gpus_info()):
                print(f"🖥️  显卡 {i} 信息:")
                for key, value in info.items():
                    print(f"   {key}: {value}")
                print("-" * 40)

            # 运算测试结果
            print(self.test_gpu_compute())
            print(f"🔌 默认训练设备: {self.get_default_device()}")
        else:
            print("❌ 无可用GPU，将使用CPU运行")

        print("=" * 60)


# ------------------- 测试使用 -------------------
if __name__ == "__main__":
    # 1. 创建检测实例
    gpu_checker = PyTorchGPUChecker()

    # 2. 一键打印完整检测报告（核心用法）
    gpu_checker.print_full_report()

    # 3. 单独调用方法（按需使用）
    # print("\n单独获取默认设备:", gpu_checker.get_default_device())
    # print("单独获取显卡0信息:", gpu_checker.get_single_gpu_info(0))
