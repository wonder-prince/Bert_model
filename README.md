## 运行模型

git clone https://github.com/wonder-prince/Bert_model.git

## 安装依赖

推荐使用虚拟环境进行安装（venv/conda）
python =3.12
pip install -r requirements.txt

## 安装torch

pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

## 训练效果
TrainOutput(global_step=32940, training_loss=0.951408569669463, metrics={'train_runtime': 47886.8206, 'train_samples_per_second': 5.503, 'train_steps_per_second': 0.688, 'total_flos': 5.164033933049242e+16, 'train_loss': 0.951408569669463, 'epoch': 2.0})

