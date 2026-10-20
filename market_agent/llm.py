"""统一的大模型调用入口。

本项目所有 Agent（主管、调研、分析、总结）最终都通过这里调大模型，
换模型 / 换厂商只需要改 config.py。

三个入口：
- chat：普通文本对话
- chat_json：强制 JSON 输出（接口级约束），供主管拆解任务、质检使用
- chat_with_tools：带工具调用的底层调用，返回原始 message 对象

所有调用都带指数退避重试，应对限流和网络抖动。
"""
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import API_KEY, BASE_URL, MODEL

_client: OpenAI | None = None

# 值得重试的异常：限流、网络中断、服务端 5xx。
# 鉴权失败、参数错误不在其列——重试多少次结果都一样。
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def _get_client() -> OpenAI:
    """惰性创建客户端。

    模块导入时不校验 Key，等真正调用时才检查——这样没配 Key 的
    环境（如 CI）也能导入模块、运行纯逻辑单元测试。
    """
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "还没配置 DEEPSEEK_API_KEY。\n"
                "请打开项目根目录的 .env 文件，把 sk- 开头的 Key 填进去再运行。"
            )
        # DeepSeek 是 OpenAI 兼容接口，直接用 openai 官方库
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


@retry(
    stop=stop_after_attempt(3),  # 最多试 3 次
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 2s → 4s → 8s，封顶 30s
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    reraise=True,  # 重试耗尽后抛原始异常，而不是包一层的 TenacityError
)
def _create(**kwargs):
    return _get_client().chat.completions.create(**kwargs)


def chat(system_prompt: str, user_message: str) -> str:
    """调一次大模型，返回文本回复。

    参数:
        system_prompt: 系统提示词，定义 Agent 的角色和规则
        user_message: 用户消息，本次要解决的具体任务
    """
    response = _create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # 调研类任务要稳定可靠，温度调低
    )
    return response.choices[0].message.content


def chat_json(system_prompt: str, user_message: str) -> str:
    """调一次大模型，强制返回 JSON 文本。

    response_format={"type": "json_object"} 是接口级硬约束：模型只能
    吐出合法 JSON。相比之下，在提示词里写「只输出 JSON」只是软约束，
    模型偶尔仍会附带解释文字，下游还要多写一层清洗。
    """
    response = _create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def chat_with_tools(messages: list, tools: list | None = None):
    """带工具调用的底层调用，返回原始 message 对象。

    调用方自行处理工具循环：把返回的 message 和工具执行结果依次
    append 回 messages，直到 message 不再包含 tool_calls。
    """
    kwargs = {"model": MODEL, "messages": messages, "temperature": 0.3}
    if tools:
        kwargs["tools"] = tools
    response = _create(**kwargs)
    return response.choices[0].message
