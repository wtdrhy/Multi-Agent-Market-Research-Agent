"""主管纯逻辑测试：JSON 解析容错与质检结论解析。不调大模型。

这些函数是 Agent 间协议的「海关」——上游模型偶尔不按格式输出时，
靠它们保证流程不中断，所以每一种畸形输入都要有用例钉住。
"""
import unittest

from market_agent.supervisor import parse_check, parse_subtasks


class TestParseSubtasks(unittest.TestCase):
    def test_plain_array(self):
        """标准 JSON 数组直接解析。"""
        text = (
            '[{"agent": "research", "task": "调研行业现状"},'
            ' {"agent": "analysis", "task": "分析竞争格局"}]'
        )
        self.assertEqual(
            parse_subtasks(text, "某行业"),
            [
                {"agent": "research", "task": "调研行业现状"},
                {"agent": "analysis", "task": "分析竞争格局"},
            ],
        )

    def test_code_block_wrapped(self):
        """模型把 JSON 包进 markdown 代码块时也要能剥出来。"""
        text = '```json\n[{"agent": "research", "task": "调研行业现状"}]\n```'
        self.assertEqual(
            parse_subtasks(text, "某行业"),
            [{"agent": "research", "task": "调研行业现状"}],
        )

    def test_dict_wrapper_subtasks(self):
        """兼容 {"subtasks": [...]} 包装结构。"""
        text = '{"subtasks": [{"agent": "research", "task": "调研行业现状"}]}'
        self.assertEqual(
            parse_subtasks(text, "某行业"),
            [{"agent": "research", "task": "调研行业现状"}],
        )

    def test_dict_wrapper_tasks(self):
        """DeepSeek JSON mode 可能返回 {"type": "json_object", "tasks": [...]}。"""
        text = '{"type": "json_object", "tasks": [{"agent": "research", "task": "调研行业现状"}]}'
        self.assertEqual(
            parse_subtasks(text, "某行业"),
            [{"agent": "research", "task": "调研行业现状"}],
        )

    def test_dict_wrapper_content(self):
        """实测 DeepSeek 也返回过 {"type": "json_object", "content": [...]}。"""
        text = '{"type": "json_object", "content": [{"agent": "research", "task": "调研行业现状"}]}'
        self.assertEqual(
            parse_subtasks(text, "某行业"),
            [{"agent": "research", "task": "调研行业现状"}],
        )

    def test_dict_wrapper_unknown_key(self):
        """外壳键名连 content 都不是时，取第一个列表值。"""
        text = '{"foo": "bar", "items": [{"agent": "research", "task": "调研行业现状"}]}'
        self.assertEqual(
            parse_subtasks(text, "某行业"),
            [{"agent": "research", "task": "调研行业现状"}],
        )

    def test_garbage_falls_back_to_default(self):
        """完全无法解析时退回默认拆解，流程不中断。"""
        result = parse_subtasks("完全不是JSON", "新能源汽车")
        self.assertEqual(len(result), 3, "默认拆解固定 3 项")
        self.assertEqual(result[0]["task"], "调研新能源汽车的行业现状")
        self.assertTrue(all(set(item) == {"agent", "task"} for item in result))


class TestParseCheck(unittest.TestCase):
    def test_complete(self):
        result = parse_check('{"complete": true, "missing": []}')
        self.assertTrue(result["complete"])
        self.assertEqual(result["missing"], [])

    def test_incomplete_with_missing(self):
        text = '{"complete": false, "missing": [{"agent": "research", "task": "补充政策数据"}]}'
        result = parse_check(text)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing"], [{"agent": "research", "task": "补充政策数据"}])

    def test_malformed_missing_items_filtered(self):
        """missing 里混进缺字段的条目时，只保留合法的。"""
        text = (
            '{"complete": false, "missing": ['
            '{"agent": "research"}, "不是字典",'
            ' {"agent": "analysis", "task": "补风险"}]}'
        )
        result = parse_check(text)
        self.assertEqual(result["missing"], [{"agent": "analysis", "task": "补风险"}])

    def test_garbage_passes(self):
        """解析失败按完整放行——不让一次判断失误堵死整个流程。"""
        result = parse_check("模型今天不想输出JSON")
        self.assertTrue(result["complete"])
        self.assertEqual(result["missing"], [])


if __name__ == "__main__":
    unittest.main()
