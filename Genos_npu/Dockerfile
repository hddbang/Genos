FROM quay.io/ascend/vllm-ascend:v0.11.0rc2

# 安装必要的依赖
RUN pip install --upgrade pip && \
    pip install sanic

COPY genos_server.py /workspace/genos_server.py
WORKDIR /workspace/

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "genos_server.py"]

# 运行容器时需要添加的参数（用户根据实际情况调整）：
# docker build -t genos-npu-image .
#
# docker run --rm -d \
# --name genos-npu-server \
# --device=/dev/davinci0 \
# --device /dev/davinci_manager \
# --device /dev/devmm_svm \
# --shm-size=2g \
# --device /dev/hisi_hdc \
# -v /usr/local/dcmi:/usr/local/dcmi \
# -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
# -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
# -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
# -v /etc/ascend_install.info:/etc/ascend_install.info \
# -v /storeData/AI_models/modelscope/hub/models/BGI-HangzhouAI/:/AI_models/BGI-HangzhouAI/ \
# -p 8888:8000 \
# -it genos-npu-image 
#
# curl -X POST http://localhost:8888/predict -H "Content-Type: application/json" -d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "10B", "predict_length": 25}'
# curl -X POST http://localhost:8888/extract -H "Content-Type: application/json" -d '{"sequence": "GGATCCGGATCCGGATCCGGATCC", "model_name": "10B", "pooling_method": "max"}' 

