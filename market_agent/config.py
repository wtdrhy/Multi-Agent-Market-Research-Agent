"""项目配置：集中管理外部服务接口信息。

密钥放在项目根目录的 .env 文件里，代码中不出现任何密钥。
"""
import os

from dotenv import load_dotenv

load_dotenv()  # 自动读取根目录下的 .env

# DeepSeek 采用 OpenAI 兼容接口；以后换通义千问 / 智谱等，只改这三行
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# 联网搜索 Key（Tavily，https://tavily.com 免费申请）；不配则工人退化为纯模型知识
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
