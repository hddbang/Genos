# Genos NPU 版本使用说明

Genos是一个用于DNA序列分析的深度学习模型，本版本专为华为昇腾NPU优化，提供高性能的DNA序列embedding提取和碱基预测服务。

## 目录结构

```
Genos_npu/
├── Dockerfile          # 用于构建NPU版本的Docker镜像
├── genos_server.py     # 主服务程序，提供API接口
└── README.md           # 本说明文档
```

## 功能特性

- **DNA序列Embedding提取**：支持多种池化方法（mean、max、last、none）
- **DNA碱基预测**：预测DNA序列的下游碱基
- **多设备支持**：优先使用NPU，同时支持GPU和CPU
- **多模型支持**：默认支持1.2B和10B两种模型
- **RESTful API**：提供简洁易用的HTTP接口

## 环境要求

- 华为昇腾NPU设备（如Atlas系列）
- Docker环境
- NPU驱动已正确安装

## 快速开始

### 1. 构建Docker镜像

进入包含Dockerfile的目录，执行以下命令构建镜像：

```bash
cd Genos/Genos_npu
docker build --network=host -t genos-npu-image .
```

### 2. 运行Docker容器

构建完成后，使用以下命令运行容器：

```bash
docker run --rm -d \
--name genos-npu-server \
--network=host \
--device=/dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--shm-size=2g \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /本地/模型/目录/:/AI_models/BGI-HangzhouAI/ \
-p 8000:8000 \
-it genos-npu-image
```

**参数说明**：
- `--device=/dev/davinci0`：指定使用的NPU设备（根据实际情况调整）
- `-v /本地/模型/目录/:/AI_models/BGI-HangzhouAI/`：挂载模型文件目录到容器内
- `-p 8000:8000`：端口映射，这个看实际情况，宿主机的8000端口被占用了就可以调整，比如-p 8888:8000
- `-d`：后台运行容器

### 3. 调用API接口

容器启动后，可以通过HTTP API调用服务：

#### 提取DNA序列Embedding

```bash
curl -X POST http://localhost:8000/extract \
-H "Content-Type: application/json" \
-d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "10B", "pooling_method": "max"}'
```

#### 预测DNA序列下游碱基

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "10B", "predict_length": 25}'
```

## API接口详细说明

### 1. Embedding提取接口

**URL**: `/extract`
**方法**: `POST`
**Content-Type**: `application/json`

**请求参数**:
- `sequence` (必需): 输入的DNA序列
- `model_name` (必需): 模型名称（"1.2B"或"10B"）
- `pooling_method` (可选): 池化方法（"mean"、"max"、"last"、"none"，默认"mean"）

**响应示例**:
```json
{
  "success": true,
  "message": "客户端序列embedding提取成功",
  "result": {
    "sequence": "GGATCCGGATCCGGATCCGGATCC",
    "sequence_length": 24,
    "token_count": 5,
    "embedding_shape": [1, 1024],
    "embedding_dim": 1024,
    "pooling_method": "max",
    "model_type": "flash",
    "device": "npu:0",
    "embedding": [0.123, 0.456, ...]
  }
}
```

### 2. 碱基预测接口

**URL**: `/predict`
**方法**: `POST`
**Content-Type**: `application/json`

**请求参数**:
- `sequence` (必需): 输入的DNA序列
- `model_name` (必需): 模型名称（"1.2B"或"10B"）
- `predict_length` (可选): 预测的碱基数量（默认10，最大1000）

**响应示例**:
```json
{
  "success": true,
  "message": "碱基预测成功",
  "result": {
    "original_sequence": "GGATCCGGATCCGGATCCGGATCC",
    "predicted_sequence": "GGATCCGGATCCGGATCCGGATCCATCGATCGATCGATCGAT",
    "predicted_bases": "ATCGATCGATCGATCGAT",
    "predict_length": 16,
    "total_length": 40
  }
}
```

## 自定义配置

### 命令行参数

`genos_server.py`支持以下命令行参数：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--host` | 服务器监听地址 | 0.0.0.0 |
| `--port` | 服务器监听端口 | 8000 |
| `--force_cpu` | 是否强制使用CPU | False |
| `--device` | 指定运行设备（单设备: npu:0, cuda:0, cpu; 多设备: 用逗号分隔，如 "cuda:0,cuda:1" 或 "npu:0,npu:1"） | None |
| `--device_map` | 设备映射方式（auto, balanced, sequential） | None |
| `--memory_ratio` | 内存分配比例 | 0.9 |
| `--model_path_prefix` | 模型存储路径前缀 | /AI_models/BGI-HangzhouAI/ |
| `--log_level` | 日志级别 | INFO |

### 示例：自定义路径和设备

```bash
docker run --rm -d \
--name genos-npu-server \
--network=host \
--device=/dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--shm-size=2g \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /DW/AI_models/modelscope/hub/models/BGI-HangzhouAI/:/AI_models/BGI-HangzhouAI/ \
-it genos-npu-image \
python genos_server.py --device npu:0
```

## 常见问题

### 1. NPU设备无法识别

- 确保NPU驱动已正确安装
- 检查设备文件权限是否正确
- 确认Docker容器已正确挂载NPU设备

### 2. 模型加载失败

- 检查模型文件路径是否正确
- 确认模型文件已完整下载
- 如内存不足，尝试调整`--memory_ratio`参数

### 3. API调用超时

- 对于长序列或复杂任务，可能需要更长的处理时间
- 检查NPU设备负载情况
- 考虑使用更小的模型（如1.2B替代10B）

## 日志查看

可以通过以下命令查看容器日志：

```bash
docker logs genos-npu-server
```

## 停止服务

使用以下命令停止并删除容器：

```bash
docker stop genos-npu-server
```

## 技术支持

如有问题或建议，请联系开发团队。

---

**版本信息**：v1.0.0  
**发布日期**：2025-12-15  
