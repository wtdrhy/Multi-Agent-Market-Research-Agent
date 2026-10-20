"""联网搜索工具（Tavily）。

不配置 TAVILY_API_KEY 时自动降级：工人不带工具运行，用模型知识回答，
保证没配搜索 Key 的人也能跑通整个项目。
"""
from .config import TAVILY_API_KEY

# 交给大模型的工具描述（OpenAI 函数调用格式）
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索互联网获取最新信息。任务需要最新数据、事实或不确定的信息时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，尽量具体，如「2026年中国新能源汽车销量 中汽协」",
                }
            },
            "required": ["query"],
        },
    },
}

_client = None


def search_available() -> bool:
    """是否配置了搜索 Key。"""
    return bool(TAVILY_API_KEY)


def web_search(query: str, max_results: int = 3) -> str:
    """执行搜索并格式化为文本，作为工具结果返回给大模型。"""
    if not search_available():
        return "未配置搜索 API Key，跳过联网检索。"
    global _client
    try:
        if _client is None:
            from tavily import TavilyClient

            _client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = _client.search(query, max_results=max_results)
    except Exception as e:
        return f"搜索失败：{e}"

    items = []
    for r in resp.get("results", []):
        content = (r.get("content") or "")[:300]
        items.append(f"- {r.get('title', '无标题')}\n  {content}\n  来源：{r.get('url', '')}")
    return "\n".join(items) if items else "未找到相关结果。"
