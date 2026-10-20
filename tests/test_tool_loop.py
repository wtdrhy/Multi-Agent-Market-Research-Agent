"""调研工人工具调用循环测试（全 mock，不需要任何 API Key）。

mock 掉两个外部依赖——大模型调用（chat_with_tools）和搜索（web_search），
用「剧本」驱动 fake 模型按脚本返回，验证三件事：
1. 配置了搜索 Key 时，工具被正确调用、结果回传、循环收敛出回答
2. 未配置时自动降级为普通调用，不会碰搜索
3. 模型一直要工具时，轮次上限能强制收口，不会死循环
"""
import unittest
from unittest.mock import patch

from market_agent.agents import research_agent


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    """模拟 openai 返回的 message 对象，按需携带 tool_calls。"""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class TestResearchAgentToolLoop(unittest.TestCase):
    @patch("market_agent.agents.search_available", return_value=True)
    @patch("market_agent.agents.web_search", return_value="- 2026年销量1600万辆\n  来源：mock")
    @patch("market_agent.agents.chat_with_tools")
    def test_tool_loop(self, mock_chat, mock_search, mock_available):
        """启用搜索时：两个搜索词都被执行，最终产出模型回答。"""
        mock_chat.side_effect = [
            # 第 1 轮：模型一次请求两个并行搜索
            FakeMessage(tool_calls=[
                FakeToolCall("call_1", "web_search", '{"query": "2026中国新能源汽车销量"}'),
                FakeToolCall("call_2", "web_search", '{"query": "新能源汽车渗透率 2026"}'),
            ]),
            # 第 2 轮：拿到搜索结果后生成回答
            FakeMessage(content="2026年中国新能源汽车销量预计1600万辆，渗透率约55%。"),
        ]
        answer = research_agent("调研2026年中国新能源汽车行业销量与渗透率")

        self.assertEqual(mock_chat.call_count, 2, "应调两次大模型：一次要工具，一次出回答")
        self.assertEqual(mock_search.call_count, 2, "两个 tool_calls 都应触发搜索")
        mock_search.assert_any_call("2026中国新能源汽车销量")
        self.assertEqual(answer, "2026年中国新能源汽车销量预计1600万辆，渗透率约55%。")

    @patch("market_agent.agents.search_available", return_value=False)
    @patch("market_agent.agents.web_search")
    @patch("market_agent.agents.chat_with_tools")
    def test_degraded_without_key(self, mock_chat, mock_search, mock_available):
        """无搜索 Key 时：退化为普通调用，不碰搜索。"""
        mock_chat.return_value = FakeMessage(content="新能源汽车是采用电力驱动的汽车。")
        answer = research_agent("用一句话介绍什么是新能源汽车")

        mock_search.assert_not_called()
        self.assertEqual(answer, "新能源汽车是采用电力驱动的汽车。")

    @patch("market_agent.agents.search_available", return_value=True)
    @patch("market_agent.agents.web_search", return_value="- 结果\n  来源：mock")
    @patch("market_agent.agents.chat_with_tools")
    def test_tool_round_cap(self, mock_chat, mock_search, mock_available):
        """模型一直要工具时：3 轮后强制收口，不会死循环。"""
        mock_chat.side_effect = [
            FakeMessage(tool_calls=[FakeToolCall("c1", "web_search", '{"query": "q"}')]),
            FakeMessage(tool_calls=[FakeToolCall("c2", "web_search", '{"query": "q"}')]),
            FakeMessage(tool_calls=[FakeToolCall("c3", "web_search", '{"query": "q"}')]),
        ]
        answer = research_agent("调研一个永远搜不到的问题")

        self.assertEqual(mock_chat.call_count, 3, "达到 MAX_TOOL_ROUNDS 后必须停止")
        self.assertIn("上限", answer, "应收口并给出提示，而不是无限循环")


if __name__ == "__main__":
    unittest.main()
