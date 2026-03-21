# 开发文档

## 环境
```bash
uv venv --python=3.14
uv pip install -r requirements.txt
```

## 测试数据
```bash
hf download TAAC2026/data_sample_1000 --cache-dir ./data --type dataset
```