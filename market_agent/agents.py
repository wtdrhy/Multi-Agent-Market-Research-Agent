"""工人 Agent：调研、分析与总结。

调研、分析工人带联网搜索能力（工具调用循环）；总结工人纯提示词驱动。
未配置搜索 Key 时自动降级为普通调用。
"""
import json

from .llm import chat, chat_with_tools
from .tools import SEARCH_TOOL, search_available, web_search

MAX_TOOL_ROUNDS = 3  # 单个工人内最多工具调用轮数，防止失控

RESEARCHER_PROMPT = """你是一名行业研究员，负责行业现状调研。
你可以使用 web_search 工具搜索互联网获取最新数据和事实。
输出要求：
1. 按四节输出，每节一两句话：市场规模与增速 / 技术趋势 / 政策环境 / 竞争格局
2. 优先使用搜索到的数据和事实，并注明数据年份
3. 300 字以内，Markdown 格式
"""

ANALYST_PROMPT = """你是一名行业分析师，负责竞品分析或风险分析。
你可以使用 web_search 工具搜索互联网获取最新数据和事实。
先根据任务描述判断类型，再按对应框架输出：
1. 竞品分析：主要玩家 / 大致份额 / 各自优劣势
2. 风险分析：按政策、技术、市场、供应链分类，按严重程度从高到低排列
3. 优先使用搜索到的最新数据，并注明数据年份
4. 300 字以内，Markdown 格式
"""

SUMMARIZER_PROMPT = """你是一名资深行业研究员，负责把多份调研素材整合成一份完整的行业调研报告。
报告结构（严格遵守，章节名一字不改）：
# {主题}调研报告
## 一、行业现状
## 二、竞争格局
## 三、风险分析
## 四、结论与建议

整合要求：
1. 每个章节综合所有相关素材，保留关键数据，删除重复与空话
2. 素材结论冲突时，以更具体、有数据支撑的为准
3. 结论与建议给出 3-5 条可执行判断
4. 全文 800-1200 字，Markdown 格式
"""


def _run_with_search(system_prompt: str, task: str) -> str:
    """带联网搜索的工人执行体。

    工具调用循环：大模型决定搜索什么 → 执行搜索 → 结果回传给大模型
    → 继续生成，直到不再请求工具（或达到轮次上限）。
    """
    tools = [SEARCH_TOOL] if search_available() else None
    if tools:
        print("[工具] 联网搜索已启用")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    for _ in range(MAX_TOOL_ROUNDS):
        msg = chat_with_tools(messages, tools)
        if not msg.tool_calls:
            return msg.content or ""
        # 把带工具调用的回复放回上下文，再补上每个工具的执行结果
        messages.append(msg)
        for tc in msg.tool_calls:
            query = json.loads(tc.function.arguments).get("query", task)
            print(f"[工具] 搜索：{query}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": web_search(query),
            })
    return msg.content or "（工具调用轮次已达上限，未能生成完整回答）"


def research_agent(task: str) -> str:
    """调研工人：行业现状，带联网搜索。"""
    print(f"[调研 Agent] 收到任务：{task}")
    return _run_with_search(RESEARCHER_PROMPT, task)


def analysis_agent(task: str) -> str:
    """分析工人：竞品分析 / 风险分析，带联网搜索。"""
    print(f"[分析 Agent] 收到任务：{task}")
    return _run_with_search(ANALYST_PROMPT, task)


def summary_agent(topic: str, materials: str) -> str:
    """总结工人：把全部素材整合成结构化报告（不需要实时数据）。"""
    print("[总结 Agent] 整合素材，生成报告")
    return chat(
        system_prompt=SUMMARIZER_PROMPT,
        user_message=f"调研主题：{topic}\n\n调研素材：\n{materials}",
    )
