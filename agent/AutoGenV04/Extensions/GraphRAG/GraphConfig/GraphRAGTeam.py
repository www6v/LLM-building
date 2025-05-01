import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_ext.tools.graphrag import GlobalSearchTool,LocalSearchTool





os.environ["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
os.environ["OPENAI_API_KEY"] = "sk-Rj3IWhZU9DZjoyy8AQ2Q3Fo82qlKApSFtYsu40EOtH9GCytV"


# 定义LLM
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini-2024-07-18"
)

# 定义工具
local_tool = LocalSearchTool.from_settings(settings_path="./settings.yaml")
global_tool = GlobalSearchTool.from_settings(settings_path="./settings.yaml")


async def main() -> None:
    # 定义Agent
    agent = AssistantAgent(
        name="agent",
        model_client=model_client,
        description="你是一个乐于助人的AI智能助手。",
        system_message="你是一个乐于助人的AI智能助手。擅长使用工具解决任务。任务完成后，回复南哥AGI研习社",
        tools=[local_tool, global_tool],
    )

    # 定义终止条件  如果提到特定文本则终止对话
    termination = TextMentionTermination("南哥AGI研习社")

    # 定义Team Team的类型选择为RoundRobinGroupChat
    agent_team = RoundRobinGroupChat(participants=[agent], termination_condition=termination, max_turns=None)

    # run_stream()方法运行team并使用官方提供的Console工具以适当的格式输出
    stream = agent_team.run_stream(task="张三九的基本信息?")
    await Console(stream)




if __name__ == '__main__':
    # 运行main
    asyncio.run(main())

