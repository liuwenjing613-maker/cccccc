from torchvision import models
from Model_evaluation import count_model_complexity

# 1. 初始化模型
model = models.resnet18()
# 2. 输入尺寸：3通道224*224图像
input_size = (3, 224, 224)
# 3. 执行统计，开启分层打印查看各层计算分布
stats = count_model_complexity(model, input_size, print_layer_detail=True)