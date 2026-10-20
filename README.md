# 多智能体市场调研 Agent

基于 LangGraph 的 Supervisor（主管-工人）架构多智能体系统。输入一个行业调研主题，自动完成：任务拆解 → 多工人并行调研 → 质量复查与补派 → 汇总生成结构化 Markdown 报告。

## 功能特性

- **主管-工人分工**：主管 Agent 负责任务拆解、调度与质检；调研、分析工人各司其职，提示词即岗位说明书
- **并行执行**：多个工人通过 **`asyncio`** 同时开工，实测分发耗时约为串行的 1/3
- **质检循环**：主管审查结果完整性，自动补派缺口任务，最多 2 轮，防止无限循环
- **联网搜索**：调研/分析工人可调用 Tavily 搜索获取最新数据，搜什么由模型自主决定；未配置 Key 时自动降级为纯模型知识
- **结构化输出**：主管走 JSON mode（接口级硬约束）拿拆解清单和质检结论；报告按固定大纲生成，保存至 `output/report.md`
- **可靠性设计**：LLM 调用带指数退避重试（限流 / 网络抖动 / 5xx）；JSON 解析三层容错，任何一环失败都有 fallback，流程不中断

## 架构

```
START ──▶ plan ──▶ dispatch ──▶ check ──┬── 缺信息 ──▶ dispatch（补派，≤2 轮）
         主管拆解    工人并行执行  主管质检  │
                                        └── 完整/满 2 轮 ──▶ summarize ──▶ END
                                                            总结成文
```

所有节点通过共享 State 协作，节点之间不直接调用：

| State 字段                       | 说明                | 合并策略         |
| ------------------------------ | ----------------- | ------------ |
| `topic` / `subtasks` / `round` | 主题 / 本轮子任务 / 调度轮次 | 覆盖式          |
| `results`                      | 累计的工人产出           | 节点内合并保留      |
| `messages`                     | 执行记录              | 追加式（reducer） |

## 快速开始

```bash
# 1. 创建虚拟环境（Python 3.10+）
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥
cp .env.example .env          # 然后编辑 .env 填入 DEEPSEEK_API_KEY

# 4. 运行（二选一）
python main.py                      # 启动后按提示输入调研主题
python main.py 2026年中国人工智能行业  # 或直接把主题作为参数传入
```

运行后报告生成在 `output/report.md`，控制台会打印完整的调度日志（谁拆解、谁执行、质检结论、补派内容）。

## 配置说明

| 环境变量               | 必填 | 说明                                                     |
| ------------------ | -- | ------------------------------------------------------ |
| `DEEPSEEK_API_KEY` | 是  | 大模型接口，OpenAI 兼容格式，换其他厂商只需改 `market_agent/config.py` 三行 |
| `TAVILY_API_KEY`   | 否  | 联网搜索；留空则工人退化为纯模型知识                                     |
| `LANGCHAIN_API_KEY` | 否 | LangSmith 链路追踪；填入后（并设 `LANGCHAIN_TRACING_V2=true`）可在 LangSmith 控制台查看每次图执行的节点调用与耗时 |

## Docker 运行

```bash
docker build -t market-agent .
docker run --rm --env-file .env market-agent
```

## CI

推送到 GitHub 后自动执行（`.github/workflows/ci.yml`）：Python 3.11 / 3.12 双版本安装依赖 → 运行全部单元测试 → ruff 风格检查。测试全 mock，无需任何密钥。

## 项目结构

```
market_agent/
├── config.py       # 接口配置（从 .env 读取）
├── llm.py          # 大模型调用封装：文本 / JSON mode / 工具调用，带重试
├── agents.py       # 工人 Agent：调研 / 分析 / 总结
├── supervisor.py   # 主管提示词与 JSON 解析（拆解、质检）
└── tools.py        # 联网搜索工具（Tavily）
tests/
├── test_tool_loop.py    # 工具调用循环：mock 大模型 + 搜索，验证收敛与降级
├── test_supervisor.py   # JSON 解析容错：包装结构、畸形输入、兜底
└── test_graph.py        # 全图集成：mock 工人 + 主管，验证并行 / 补派 / 轮次上限
main.py             # 图组装与入口
```

## 运行测试

```bash
pip install -r requirements-dev.txt   # 第一次先装 ruff
python -m unittest discover -s tests -v
ruff check .
```

测试用 mock 替代大模型和搜索 API，覆盖：工具调用循环的收敛与轮次上限、无 Key 降级、JSON 解析三层容错、质检循环的补派与强制放行、并行 fan-out/fan-in 的结果合并。CI 无需密钥即可运行。

## 实现要点

- **JSON 作为 Agent 间协议**：主管输出结构化子任务清单（`[{"agent": "research", "task": "..."}]`），程序按 `agent` 字段路由；调用侧走 JSON mode 做接口级约束，解析侧三层容错（剥代码块、兼容包装结构、失败兜底）
- **并发安全靠统一收口**：工人只返回字符串，结果由 dispatch 节点统一合并，工人之间不共享可变状态
- **循环变量由执行方维护**：轮次计数放在 dispatch 节点，避免质检节点预增导致路由条件误触
- **优雅降级**：缺搜索 Key、搜索 API 异常、LLM 不守 JSON 格式，均有 fallback，流程不中断
- **LLM 客户端惰性初始化**：没配 Key 时导入不报错，纯逻辑测试和 CI 都能跑

## 技术栈

LangGraph · DeepSeek（OpenAI 兼容接口）· Tavily（可选）· LangSmith（可选）· tenacity · Python 3.10+ · Docker · GitHub Actions · ruff
