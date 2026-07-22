import torch
from ptflops import get_model_complexity_info

def count_model_complexity(model: torch.nn.Module, 
                            input_shape: tuple,
                            print_layer_detail: bool = False,
                            device: str = "cpu") -> dict:
    """
    统计模型参数量、MACs计算量工具函数
    Args:
        model: 待评估torch模型
        input_shape: 模型输入维度 (Channel, Height, Width)
        print_layer_detail: 是否打印每层计算量明细，定位计算密集层
        device: 计算设备，ptflops仅做前向推演统计，cpu/gpu均可
    Returns:
        dict: 结构化统计结果，包含格式化字符串与原始数值
    """
    # 1. 模型预处理：推理模式，关闭梯度节省内存
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # 2. 调用ptflops核心统计接口
        # 返回值: (macs_str, params_str)
        macs_str, params_str = get_model_complexity_info(
            model=model,
            input_res=input_shape,
            print_per_layer_stat=print_layer_detail,
            as_strings=True,
            verbose=False
        )
    
    # 3. 控制台打印汇总指标
    print("=" * 50)
    print(f"【模型复杂度统计汇总】")
    print(f"参数量 Parameters: {params_str}")
    print(f"计算量 MACs: {macs_str}")
    print("=" * 50)

    # 4. 封装结构化结果返回
    output_dict = {
        "format_info": {
            "params": params_str,
            "macs": macs_str,
        },
    }
    return output_dict
