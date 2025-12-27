# 使用你本地已有的镜像
FROM ubuntu:24.04

# 设置环境变量，避免安装过程中的交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 更新系统并安装基础工具（包括常用开发库和显卡支持库）
RUN apt update && apt apt install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    vim \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 默认进入 bash
CMD ["/bin/bash"]
