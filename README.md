# Self_Driver-KITTI

这是一个与BDD100K项目完全分离的KITTI专用自动驾驶视觉项目，面向Windows和RTX 4070 Ti
SUPER 16GB。项目只处理经典KITTI 2D Object Detection数据：

- KITTI原始标签转换为YOLO格式；
- 固定随机种子的可配置数据集；当前配置从已转换的全部带标签数据生成10%轻量子集；
- YOLO目标检测训练与评估；
- ByteTrack多目标跟踪和轨迹可视化；
- ONNX模型导出。

本项目不包含BDD100K代码、可行驶区域Mask、车道线Mask或LiteTwinSeg。KITTI目标检测集没有
与每张检测图片配套的上述分割标注，因此不伪造或混用其他数据集标签。

## 快速开始

请先自行准备满足 `requirements.txt` 的 Python 环境；运行本项目无需安装本地包。

```powershell
cd "D:\自动驾驶算法学习\Self_Driver-KITTI"
conda activate zzb310
python tools\check_environment.py
```

当前已存在 `data/kitti_100pct` 时，从中抽取10%并训练：

```powershell
python tools\prepare_subset_from_prepared.py --config configs\kitti.yaml
python tools\check_dataset.py --config configs\kitti.yaml
python tools\train_detector.py --config configs\kitti.yaml
```

如果重新从 KITTI 原始标签转换，请先把原始数据放到：

```text
Datas/KITTI/training/image_2/*.png
Datas/KITTI/training/label_2/*.txt
```

训练和评估：

```powershell
python tools\train_detector.py --config configs\kitti.yaml
python tools\evaluate_detector.py --config configs\kitti.yaml
```

视频推理和ONNX导出：

```powershell
python tools\infer_video.py --config configs\kitti.yaml --source assets\demo.mp4 --output runs\demo_result.mp4
python tools\export_onnx.py --config configs\kitti.yaml
```

## 当前数据规模

`configs/kitti.yaml` 当前从全部7,481张带标签图片中抽取748张（`fraction: 0.1`），再按80%/20%划分：

- 训练集：598张；
- 验证集：150张。

从原始 KITTI 标签准备数据时使用 `python tools\prepare_dataset.py --config configs\kitti.yaml`；
当前目录没有原始标签，因此采用上面的已转换数据入口。调整 `fraction` 时需同步使用独立的
`output_root`、`dataset_yaml`、实验名和权重路径。固定 `seed: 42` 可复现同一划分；
本地划分结果不能与KITTI官方测试榜单直接比较。

