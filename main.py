"""多智能体市场调研 Agent 入口。

图结构：plan → dispatch → check ─┬─ 缺信息 → dispatch（补派，最多 2 轮）
                                  └─ 完整或满 2 轮 → summarize → END

主管（plan / check）走 JSON mode 拿结构化输出；工人并行执行，
可联网搜索（配置 TAVILY_API_KEY 后启用）。
"""
import asyncio
import operator
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from market_agent.agents import analysis_agent, research_agent, summary_agent
from market_agent.llm import chat_json
from market_agent.supervisor import (
    CHECK_PROMPT,
    SUPERVISOR_PROMPT,
    parse_check,
    parse_subtasks,
)

WORKERS = {"research": research_agent, "analysis": analysis_agent}
MAX_ROUNDS = 2  # 最多调度 2 轮，防止无限循环
OUTPUT_DIR = Path("output")


class ResearchState(TypedDict):
    """共享 State：所有节点都在读写它。"""
    topic: str                              # 调研主题
    subtasks: list                          # 本轮要执行的子任务
    results: dict                           # 累计的工人产出 {task: result}
    round: int                              # 第几轮调度
    complete: bool                          # 质检结论
    report: str                             # 最终报告
    messages: Annotated[list, operator.add]  # 执行记录，只增不减


def plan_node(state: ResearchState) -> dict:
    """主管：把主题拆成子任务清单。"""
    print(f"[主管] 拆解任务：{state['topic']}")
    plan = chat_json(
        system_prompt=SUPERVISOR_PROMPT,
        user_message=f"调研主题：{state['topic']}",
    )
    subtasks = parse_subtasks(plan, state["topic"])
    for i, st in enumerate(subtasks, 1):
        print(f"  子任务 {i}：[{st['agent']}] {st['task']}")
    return {
        "subtasks": subtasks,
        "messages": [f"[主管] 第 1 轮拆解出 {len(subtasks)} 个子任务"],
    }


async def run_workers(subtasks: list[dict], parallel: bool = True) -> dict:
    """执行子任务。

    parallel=True 时所有工人同时开工；同步的工人函数通过 asyncio.to_thread
    丢进线程池，避免阻塞事件循环。结果由这里统一收集——
    工人之间不共享任何可变状态，这是并发安全的关键。
    """
    if not parallel:
        results = {}
        for st in subtasks:
            worker = WORKERS.get(st["agent"], analysis_agent)
            results[st["task"]] = worker(st["task"])
        return results

    async def run_one(st: dict) -> str:
        worker = WORKERS.get(st["agent"], analysis_agent)
        return await asyncio.to_thread(worker, st["task"])

    outputs = await asyncio.gather(*(run_one(st) for st in subtasks))
    return {st["task"]: out for st, out in zip(subtasks, outputs)}


async def dispatch_node(state: ResearchState) -> dict:
    """并行分发：所有工人同时开工，到齐后统一收集。"""
    round_no = state.get("round", 0) + 1  # round 由分发轮次自身计数
    t0 = time.perf_counter()
    new_results = await run_workers(state["subtasks"])
    cost = time.perf_counter() - t0
    # 新结果合并进累计结果，不是替换——补派回来的内容要保留
    return {
        "round": round_no,
        "results": {**state.get("results", {}), **new_results},
        "messages": [
            f"[调度] 第 {round_no} 轮：{len(new_results)} 个任务并行完成，耗时 {cost:.1f}s"
        ],
    }


def check_node(state: ResearchState) -> dict:
    """质检：审查已有结果，决定放行还是补派。"""
    digest = "\n\n".join(f"【{task}】\n{text}" for task, text in state["results"].items())
    verdict = chat_json(
        system_prompt=CHECK_PROMPT,
        user_message=f"调研主题：{state['topic']}\n\n当前调研结果：\n{digest}",
    )
    result = parse_check(verdict)
    current = state["round"]  # 已完成的调度轮数

    if result["complete"] or not result["missing"]:
        print(f"[质检] 第 {current} 轮：信息完整，同意汇总")
        return {"complete": True, "messages": [f"[质检] 第 {current} 轮：完整"]}

    if current >= MAX_ROUNDS:
        print(f"[质检] 第 {current} 轮：仍不完整，已达 {MAX_ROUNDS} 轮上限，强制汇总")
        return {
            "complete": True,
            "messages": [f"[质检] 第 {current} 轮：不完整但达到上限，强制汇总"],
        }

    print(f"[质检] 第 {current} 轮：缺少 {len(result['missing'])} 项，进入第 {current + 1} 轮补派")
    for st in result["missing"]:
        print(f"    补派：[{st['agent']}] {st['task']}")
    return {
        "complete": False,
        "subtasks": result["missing"],
        "messages": [
            f"[质检] 第 {current} 轮：缺少 {len(result['missing'])} 项，补派第 {current + 1} 轮"
        ],
    }


def route_after_check(state: ResearchState):
    """条件边：完整或到上限就去汇总，否则回 dispatch 补派。"""
    if state["complete"] or state["round"] >= MAX_ROUNDS:
        return "summarize"
    return "dispatch"


def summarize_node(state: ResearchState) -> dict:
    """总结工人：把全部素材整合成结构化报告并保存。"""
    materials = "\n\n".join(f"【{task}】\n{text}" for task, text in state["results"].items())
    report = summary_agent(state["topic"], materials)

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "report.md"
    footer = (
        f"\n\n---\n\n- 调研主题：{state['topic']}\n"
        f"- 调度轮次：{state['round']}\n"
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}"
    )
    path.write_text(report + footer, encoding="utf-8")
    print(f"[总结] 报告已保存：{path}")
    return {"report": report, "messages": ["[总结] 报告生成并保存至 output/report.md"]}


def build_graph():
    """搭图：计划 → 并行执行 → 质检循环 → 汇总成文。"""
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("check", check_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "dispatch")
    graph.add_edge("dispatch", "check")
    graph.add_conditional_edges("check", route_after_check)
    graph.add_edge("summarize", END)
    return graph.compile()


DEFAULT_TOPIC = "2026年中国新能源汽车行业"  # 不传主题时的演示主题


def main():
    """命令行入口。

    用法：
        python main.py                        # 启动后交互输入主题
        python main.py 2026年中国人工智能行业    # 直接把主题作为参数传入
    """
    if len(sys.argv) > 1:
        # 参数没加引号时 shell 会拆成多个词，拼回去
        topic = " ".join(sys.argv[1:]).strip() or DEFAULT_TOPIC
    else:
        topic = input(f"请输入调研主题（直接回车用演示主题「{DEFAULT_TOPIC}」）：").strip()
        topic = topic or DEFAULT_TOPIC

    app = build_graph()
    final = asyncio.run(app.ainvoke({"topic": topic}))

    print("\n=== 执行记录 ===")
    for m in final["messages"]:
        print(" ", m)

    print("\n=== 最终报告 ===")
    print(final["report"])


if __name__ == "__main__":
    main()
