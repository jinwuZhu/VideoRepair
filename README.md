### VideoRepair

基于AI的视频修复工具
该工具由一个三层卷积的和全连接层组成的判别模型和RIFE补帧模型组成，由判别模型鉴别视频帧并RIFE进行修复
目前已经通过训练，在200次左右达到最好效果，300批次后准确率开始下降

- 准确率展示 （测试样本）

![准确率](https://github.com/jinwuZhu/VideoRepair/blob/3b05e2530ee35ec7de7ea32bce3664fb8deaadc2/test/accuracy.png)

- 损失值展示（训练样本）

![损失](https://github.com/jinwuZhu/VideoRepair/blob/3b05e2530ee35ec7de7ea32bce3664fb8deaadc2/test/loss.png)

### 支持的功能
- 花屏帧修复补全

### 效果展示
- 下面是两个图片，展示了修复前后的效果
![修复前](https://github.com/jinwuZhu/VideoRepair/blob/00d7c18d634ed9781b7f359cafefaf73e0443d7e/view_bad.gif)
![修复后](https://github.com/jinwuZhu/VideoRepair/blob/bb2c3d517d0b758c395eae9d855b0fe32c440c40/view_repair.gif)

### 如何使用
- 工程提供了一个命令行工具，可以直接执行命令
```shell
python .\video_repair.py --i bad.mp4  --o o.avi
```

- 完整命令参数：
```shell
python video_repair.py 
--input        # 输入视频文件
--output      # 输出视频文件
--batch        # 每次处理的视频帧批次，默认30帧，如果您的内存不够，建议减少
--cache        # 临时文件路径，默认 ./cache
--model       # 模型文件路径， 默认 ./noise_image.plt
```

### 安装
1. 你需要安装python3.9 + 以及pytroch,numpy,cv2等依赖库
```
pip install torch torchvision
pip install numpy
pip install opencv-python
```

2. 确保设备上有FFMPEG命令行工具，并配置到全局环境变量中
3. 获取训练好的模型，并放到工程目录下。（注意：你也可以另外放到指定目录下，但是你需要指定 --model 参数来确定加载地址）
```
https://pan.baidu.com/s/18BerOUC_V_cliOawXy2NCw?pwd=bjub
```

### 引用
1. 补帧模型：https://github.com/megvii-research/ECCV2022-RIFE
