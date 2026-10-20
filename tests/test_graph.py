"""全图集成测试：mock 掉大模型和工人，验证 LangGraph 图的结构正确性。

覆盖fan-out/fan-in 并行、跨轮结果合并、轮次计数、质检循环路由、
上限强制放行。全程不碰真实 API——CI 里没有密钥也能跑。

运行：.venv/Scripts/python.exe -m unittest tests.test_graph -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


def fake_worker(name: str):
    """造一个假工人：只回一句话，证明自己被调度过。"""
    def worker(task: str) -> str:
        return f"[{name}] 关于「{task}」的调研结论"
    return worker


class TestGraph(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 报告改写到临时目录，不污染项目 output/
        self.tmp = tempfile.TemporaryDirectory()
        patcher = patch.object(main, "OUTPUT_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def _patch_workers(self):
        patcher = patch.object(
            main,
            "WORKERS",
            {"research": fake_worker("research"), "analysis": fake_worker("analysis")},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_happy_path_single_round(self):
        """质检一次通过：1 轮调度，3 个工人全部产出，报告落盘。"""
        self._patch_workers()
        plan = (
            '[{"agent": "research", "task": "调研现状"},'
            ' {"agent": "analysis", "task": "分析竞争"},'
            ' {"agent": "analysis", "task": "分析风险"}]'
        )
        check = '{"complete": true, "missing": []}'
        with patch.object(main, "chat_json", side_effect=[plan, check]):
            final = await main.build_graph().ainvoke({"topic": "测试行业"})

        self.assertEqual(final["round"], 1, "一轮就齐活")
        self.assertEqual(len(final["results"]), 3, "三个子任务都有产出")
        self.assertTrue(final["report"], "必须有最终报告")
        self.assertTrue((Path(self.tmp.name) / "report.md").exists(), "报告应保存到 output 目录")

    async def test_reflection_loop(self):
        """质检打回补派：第 1 轮缺 1 项，第 2 轮补齐后放行。"""
        self._patch_workers()
        plan = (
            '[{"agent": "research", "task": "调研现状"},'
            ' {"agent": "analysis", "task": "分析竞争"}]'
        )
        check_1 = '{"complete": false, "missing": [{"agent": "analysis", "task": "补充风险分析"}]}'
        check_2 = '{"complete": true, "missing": []}'
        with patch.object(main, "chat_json", side_effect=[plan, check_1, check_2]):
            final = await main.build_graph().ainvoke({"topic": "测试行业"})

        self.assertEqual(final["round"], 2, "补派后应跑第 2 轮")
        self.assertEqual(len(final["results"]), 3, "第 1 轮 2 项 + 补派 1 项")
        self.assertIn("补充风险分析", final["results"], "补派结果要合并进累计结果")
        self.assertTrue(
            any("第 2 轮" in m for m in final["messages"]),
            "执行记录应体现第二轮",
        )

    async def test_force_finish_at_max_rounds(self):
        """始终不完整：达到轮次上限后强制汇总，不死循环。"""
        self._patch_workers()
        plan = '[{"agent": "research", "task": "调研现状"}]'
        check_1 = '{"complete": false, "missing": [{"agent": "research", "task": "补充数据"}]}'
        check_2 = '{"complete": false, "missing": [{"agent": "research", "task": "还要更多"}]}'
        with patch.object(main, "chat_json", side_effect=[plan, check_1, check_2]):
            final = await main.build_graph().ainvoke({"topic": "测试行业"})

        self.assertEqual(final["round"], main.MAX_ROUNDS, "不能超过上限轮次")
        self.assertTrue(
            any("强制汇总" in m for m in final["messages"]),
            "应有强制汇总的执行记录",
        )
        self.assertTrue(final["report"], "即使不完整也必须产出报告")

    async def test_run_workers_parallel(self):
        """run_workers 并行路径：任务到产出的映射不能错位。"""
        self._patch_workers()
        subtasks = [
            {"agent": "research", "task": "任务A"},
            {"agent": "analysis", "task": "任务B"},
        ]
        results = await main.run_workers(subtasks)
        self.assertEqual(set(results), {"任务A", "任务B"})
        self.assertIn("research", results["任务A"])
        self.assertIn("analysis", results["任务B"])


if __name__ == "__main__":
    unittest.main()
