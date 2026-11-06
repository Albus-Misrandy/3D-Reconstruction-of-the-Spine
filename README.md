# 脊柱 CT 深度学习辅助分割与手术规划 STL 三角网格生成系统

本项目严格按照文献与临床手术规划技术路径实现：
> **“深度学习辅助分割采用 nnU-Net，模型在 CTSpine1K 数据集上预训练，并结合临床小样本资料进行微调。模型提供的椎骨初始轮廓经逐例人工核查和修正；终板、椎弓根及椎管邻近区域需结合原始 CT 切片观察，尤其注意骨折后缘、薄层骨皮质和既往内固定物附近的边界。确定分割范围后，采用移动立方体（Marching Cubes）算法提取等值面，输出 STL 三角网格。手术规划以核查后的表面模型为依据。”**

---

## 🌟 核心特性与技术亮点

1. **nnU-Net 预训练模型即开即用**：
   - 自动加载 `CTSpine1K/model_final_checkpoint.model` 官方权重；
   - 兼容 PyTorch 2.6+ 与现代 CUDA 硬件（如 RTX 50 系列），支持快速推理模式与高精度 TTA 镜像模式。
2. **临床金标准人工核查与修正工作流**：
   - **ITK-SNAP 一键联动**：自动检测本地 `D:\ITK-SNAP 4.4`，一键载入 CT 灰度像与多颜色解剖标签覆盖层；
   - **正交多平面 (MPR) 质检报告生成器**：自动截取矢状位（终板/椎体高度）、轴位（椎弓根/椎管占位/骨折后缘）、冠状位（侧弯对称性）骨窗高分辨率质检图。
3. **手术级高精度移动立方体（Marching Cubes）表面重构**：
   - **患者物理世界坐标系 1:1 对齐**：严格根据 NIfTI Affine 仿射矩阵进行体素到物理毫米空间的转换，保证导出的 STL 在尺寸、空间姿态与原点上与患者 CT 绝对重合；
   - **高保真抗阶梯平滑（Volume-Preserving Anti-Aliasing）**：消除离散体素带来的阶梯伪影（Staircase Artifacts），同时防止传统平滑算法造成的椎体体积萎缩；
   - **拓扑修复与闭合流形（Watertight Manifold）**：滤除悬浮杂质噪点，规范法向量一致向外，输出完美闭合水密 STL；
   - **灵活分割范围（Scope）**：支持独立输出每个椎体 STL（如 `L1.stl`, `L2.stl`，便于椎弓根螺钉置钉仿真）、合并输出全段 STL（`spine_combined.stl`），或指定病变节段导出。
4. **小样本临床微调（Fine-Tuning）模块**：
   - 提供标准迁移学习训练脚本，支持冻结编码器或全网络在科室专有病例数据上微调。

---

## 📁 项目工程架构

```
Medical_Image_Segmentation/
├── CTSpine1K/                       # 数据集与预训练模型目录
│   ├── model_final_checkpoint.model
│   ├── model_final_checkpoint.model.pkl
│   ├── plans.pkl
│   ├── data/                        # 原始 CT 序列 (如 colon, COVID-19, liver 等)
│   └── label/                       # 金标准标注
├── spine_segmentation/              # 核心模块代码包
│   ├── __init__.py
│   ├── config.py                    # 解剖标签对照表、ITK-SNAP 路径与默认参数
│   ├── predictor.py                 # nnU-Net 推理引擎与滑窗预测器
│   ├── marching_cubes_stl.py        # 移动立方体等值面提取与高质量 STL 导出
│   ├── visualizer_qa.py             # 终板/椎弓根/椎管正交质检切片可视化器
│   ├── itksnap_helper.py            # ITK-SNAP 自动化联动与核查指引
│   └── finetuner.py                 # 临床小样本微调训练器 (Dice + CE 联合损失)
├── main_pipeline.py                 # 端到端一键运行流水线 (预测 -> 质检 -> 核查 -> STL)
├── export_stl_only.py               # 独立工具: 直接对已有/核查修正后的 Mask 提取 STL
├── launch_itksnap_check.py          # 独立工具: 一键打开 ITK-SNAP 进行人工核查与修正
├── run_qa_visualization.py          # 独立工具: 快速输出临床多平面质检图 (PNG)
├── train_finetune.py                # 独立工具: 临床小样本微调训练
└── README.md                        # 使用文档
```

---

## 🦴 脊柱解剖标签映射对照表 (VerSe / CTSpine1K 标准)

| 标签 ID | 解剖名称 | 中文解剖学名称 | 临床手术规划重点关注特征 |
| :---: | :---: | :---: | :--- |
| **1 ~ 7** | C1 ~ C7 | 寰椎、枢椎至隆椎 (颈椎) | 齿突完整性、椎动脉孔、横突孔 |
| **8 ~ 19** | T1 ~ T12 | 胸椎 1 ~ 12 节 | 肋骨小头关节面、胸椎椎弓根极细径 |
| **20** | **L1** | 第一腰椎 | 흉腰段交界处，最常见骨折部位、爆裂骨折后缘后突 |
| **21** | **L2** | 第二腰椎 | 上下软骨终板形态、椎管前后径 |
| **22** | **L3** | 第三腰椎 | 侧弯顶点椎、椎弓根螺钉关键定位节段 |
| **23** | **L4** | 第四腰椎 | 峡部裂、椎体滑脱 (Spondylolisthesis) |
| **24** | **L5** | 第五腰椎 | 腰骶交界处，椎弓根斜角最大 |
| **25** | **Sacrum** | 骶骨 (S1) | 骶骨翼、S1 螺钉置入锚定区 |

---

## 🚀 快速使用指南

运行环境位于 Conda 环境 `D:\anaconda3\envs\pytorch`：

### 1. 一键端到端运行 (全自动推理 -> 质检 -> 提取 STL)

直接指定任意输入的 CT 扫描文件（如 CTSpine1K 中的真实病例）：

```bash
D:\anaconda3\envs\pytorch\python.exe main_pipeline.py --input "CTSpine1K/data/colon/1.3.6.1.4.1.9328.50.4.0001.nii.gz" --output_dir output_results --scope both
```

#### 常用参数说明：
- `--input` (`-i`): 输入原始 CT 文件（`.nii` 或 `.nii.gz`）。
- `--output_dir` (`-o`): 结果输出目录，默认保存在 `output_results/`。
- `--scope` (`-s`): 
  - `individual`: 分别为每个椎骨输出独立的 STL（如 `patient_L1.stl`, `patient_L2.stl`）；
  - `all`: 输出整体合并后的完整脊柱 STL (`patient_combined_spine.stl`)；
  - `both`: **推荐**，同时输出各个独立椎骨与整体脊柱 STL。
- `--labels` (`-l`): 指定需要导出的特定节段，支持名称或编号，逗号隔开，例如 `--labels L1,L2,L3`。
- `--interactive_check`: 自动唤起 ITK-SNAP 供临床医生核查修正。
- `--clinical_mode`: 开启 8x 测试时数据增强 (TTA)，追求极致分割精度。

---

### 2. 独立提取 STL (对已有或人工修正后的 Mask 提取网格)

如果您已经拥有分割掩模（例如模型预测生成的掩模，或者经过 ITK-SNAP 人工修正保存的 `refined.nii.gz`，或是数据集自带的 ground truth）：

```bash
# 提取全部椎体，并分别导出各独立 STL 和整体 STL
D:\anaconda3\envs\pytorch\python.exe export_stl_only.py --mask "CTSpine1K/label/conlon/1.3.6.1.4.1.9328.50.4.0001_seg.nii.gz" --output_dir exported_stl --scope both

# 仅提取骨折相关的腰椎节段 (L1, L2, L3)
D:\anaconda3\envs\pytorch\python.exe export_stl_only.py --mask "CTSpine1K/label/conlon/1.3.6.1.4.1.9328.50.4.0001_seg.nii.gz" --output_dir exported_stl --labels L1,L2,L3
```

---

### 3. 临床逐例人工核查与修正 (ITK-SNAP 交互)

依据文献中特别强调的：
> **“终板、椎弓根及椎管邻近区域需结合原始 CT 切片观察，尤其注意骨折后缘、薄层骨皮质和既往内固定物附近的边界。”**

运行以下命令，系统将自动读取本机安装的 `ITK-SNAP 4.4`，并自动挂载解剖着色映射：

```bash
D:\anaconda3\envs\pytorch\python.exe launch_itksnap_check.py --ct "CTSpine1K/data/colon/1.3.6.1.4.1.9328.50.4.0001.nii.gz" --mask "output_results/1.3.6.1.4.1.9328.50.4.0001_pred_init.nii.gz"
```

#### 临床人工核查 SOP (Standard Operating Procedure):
1. **上下终板 (Endplates)**：在矢状位观察终板是否凹陷或受骨折挤压，修正软骨下骨缺损处过度凹陷或膨出；
2. **椎弓根通道 (Pedicles)**：在轴位观察双侧椎弓根皮质内外壁，确保螺钉通道边界精确；
3. **椎管临近区域与骨折后缘 (Spinal Canal & Fracture Edges)**：观察骨折碎片突入椎管的界限，用多边形工具精准勾勒；
4. **金属内固定伪影 (Metal Artifacts)**：如已有钛钉/钛板产生的放射状条纹伪影，手动去除放射条纹处的假阳性体素；
5. **保存修正结果**：在 ITK-SNAP 中按 `Ctrl + S`，另存为 `_pred_refined.nii.gz`，随后直接用 `export_stl_only.py` 导出最终手术规划 STL！

---

### 4. 生成多平面正交 (MPR) 质检报告

如果希望无需打开 GUI 即可批量快速审查分割质量，可生成高分辨率 MPR 质检图：

```bash
D:\anaconda3\envs\pytorch\python.exe run_qa_visualization.py --ct "CTSpine1K/data/colon/1.3.6.1.4.1.9328.50.4.0001.nii.gz" --mask "CTSpine1K/label/conlon/1.3.6.1.4.1.9328.50.4.0001_seg.nii.gz" --output "qa_reports/patient_0001_qa.png" --focus_label 20
```

---

### 5. 临床小样本微调训练 (Fine-Tuning)

如果科室有自采的特色病例（如重度骨质疏松性压缩骨折、脊柱畸形侧弯病例）：

将小样本整理为标准结构：
```
clinical_cohort/
├── imagesTr/    # 原始 CT (.nii.gz)
└── labelsTr/    # 精确标注掩模 (.nii.gz)
```
启动微调训练：
```bash
D:\anaconda3\envs\pytorch\python.exe train_finetune.py --data_dir "clinical_cohort" --output_dir "finetuned_models" --epochs 30 --lr 1e-4 --freeze_encoder
```

---

## 🛠️ 下游手术规划与 3D 打印软件兼容性

导出的 STL 三角网格符合工业标准二进制格式，已测试完美无缝导入以下软件：
- **CAD & 手术规划**：SolidWorks (SW2023 / SW2024)、Autodesk Inventor、Materialise Mimics、3D Slicer；
- **3D 打印切片**：Bambu Studio (拓竹)、JGcreat、UltiMaker Cura、PrusaSlicer。
