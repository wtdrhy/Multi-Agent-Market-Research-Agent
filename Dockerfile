FROM python:3.12-slim

WORKDIR /app

# 先拷依赖清单再装——层缓存，改代码不用重装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 密钥通过 .env 文件或 -e 环境变量注入，不要写死进镜像
CMD ["python", "main.py"]
