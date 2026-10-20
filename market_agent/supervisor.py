"""主管业务逻辑：任务拆解、质检提示词与 JSON 解析。"""
import json

SUPERVISOR_PROMPT = """你是一名调研主管，负责把调研主题拆解成子任务，分发给下面的工人：
- research：行业研究员，负责行业现状调研
- analysis：行业分析师，负责竞品分析、风险分析

拆解要求：
1. 拆成 3 个子任务，必须覆盖：行业现状、竞争格局、潜在风险
2. 只输出 JSON，不要输出任何其他文字
3. 格式严格如下：
[{"agent": "research", "task": "调研某行业的行业现状"},
 {"agent": "analysis", "task": "分析某行业的竞争格局与主要玩家"}]

agent 只能取 research 或 analysis，task 要写清楚具体做什么。"""

CHECK_PROMPT = """你是一名严格的调研主管，负责审查工人交回的调研结果。
审查标准：
1. 行业现状、竞争格局、潜在风险三个方向都必须覆盖
2. 每个方向的内容要具体，有数据或事实支撑，不能是空话
3. 各方向之间不能互相矛盾

输出要求：只输出 JSON，不要其他文字。
- 完整时：{"complete": true, "missing": []}
- 不完整时：{"complete": false, "missing": [{"agent": "research", "task": "补充调研XX的YY"}]}

missing 里只列需要补充的任务，已合格的方向不要重复派。"""


def parse_subtasks(text: str, topic: str) -> list[dict]:
    """解析主管输出的 JSON 子任务清单。

    调用方走 JSON mode，输出一定是合法 JSON，但外层结构不完全固定
    （实测 DeepSeek 有时套 {"tasks": [...]}，有时套 {"content": [...]}），
    这里做三层容错：
    1. 剥掉 markdown 代码块包裹
    2. 兼容各种包装结构：优先找 subtasks / tasks / content 键，
       找不到就取第一个列表值
    3. 解析失败时退回默认拆解，保证流程不中断
    """
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("subtasks", "tasks", "content"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                # 外壳键名不固定，退而求其次：取第一个列表值
                lists = [v for v in data.values() if isinstance(v, list)]
                if lists:
                    data = lists[0]
        return [{"agent": item["agent"], "task": item["task"]} for item in data]
    except (json.JSONDecodeError, KeyError, TypeError):
        print("[主管] JSON 解析失败，使用默认拆解")
        return [
            {"agent": "research", "task": f"调研{topic}的行业现状"},
            {"agent": "analysis", "task": f"分析{topic}的竞争格局与主要玩家"},
            {"agent": "analysis", "task": f"分析{topic}的潜在风险"},
        ]


def parse_check(text: str) -> dict:
    """解析质检结论。

    解析失败时按完整放行——不让一次判断失误堵死整个流程，
    报告质量靠上限轮次和人工抽查兜底。
    """
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        data = json.loads(text)
        missing = [
            {"agent": item["agent"], "task": item["task"]}
            for item in data.get("missing", [])
            if isinstance(item, dict) and "agent" in item and "task" in item
        ]
        return {"complete": bool(data["complete"]), "missing": missing}
    except (json.JSONDecodeError, KeyError, TypeError):
        print("[质检] 输出解析失败，按完整放行")
        return {"complete": True, "missing": []}
