# 从 ultralytics 库里导入 YOLO 这个类
# 相当于 C 的 #include "ultralytics.h"，只取其中的 YOLO
from ultralytics import YOLO

# 创建 YOLO 对象，加载 yolov8n.pt 预训练权重
# 相当于 C 里调用一个初始化函数，把模型文件读进内存
# "yolov8n.pt" 是模型文件名，n 表示 nano 最小版本；首次运行会自动下载约6MB
model = YOLO("yolov8n.pt")

# 用模型对一张图片做推理（检测）
# 输入：一个图片URL（ultralytics官方示例图 bus.jpg），会自动下载
# 输出：results，是一个列表，里面存着检测结果
# 相当于 C 里调用 model_predict(model, image)，返回结果数组
results = model("test.jpg")

# 取第一个结果（results[0]），把画了检测框的图片保存为 result.jpg
# 相当于 C 里把结果图写文件：save_image(result, "result.jpg")
results[0].save("result.jpg")

# 打印检测框信息：每个框的坐标、类别、置信度
# results[0].boxes 是一个对象，里面装着所有检测到的框
# 相当于 C 里 printf 遍历结果数组，输出每个框的数据
print(results[0].boxes)

# 打印推理耗时：预处理、推理、后处理各花了多少毫秒
# 相当于 C 里打印计时器统计的各阶段耗时
print(results[0].speed)