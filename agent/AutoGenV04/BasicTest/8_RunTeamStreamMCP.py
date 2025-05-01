import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools



# os.environ["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
# os.environ["OPENAI_API_KEY"] = "sk-nq393KS4lvCCNjscfFJDNjkwOiTc12tICQU4QZi89a6aGG0F"


# 定义LLM
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini-2024-07-18"
)


async def main() -> None:

    fetch_mcp_server = StdioServerParams(command="python", args=["utils/mcp_server_fetch.py"])
    tools = await mcp_server_tools(fetch_mcp_server)

    # 定义Agent
    agent = AssistantAgent(
        name="agent",
        model_client=model_client,
        system_message="你是一个乐于助人的AI智能助手。记住问题完成后回复“南哥AGI研习社”。",
        model_client_stream=True,
        tools=tools,
        reflect_on_tool_use=True
    )


    # 定义终止条件  如果提到特定文本则终止对话
    text_termination = TextMentionTermination("南哥AGI研习社")

    # 定义Team Team的类型选择为RoundRobinGroupChat
    reflection_team = RoundRobinGroupChat(participants=[agent], termination_condition=text_termination, max_turns=None)

    # 1、运行team并使用官方提供的Console工具以适当的格式输出
    stream = reflection_team.run_stream(task="概述 https://www.emqx.com/zh 的内容")
    await Console(stream)

    # # 2、运行team并使用流式输出,自己处理消息并将其流到前端
    # async for message in reflection_team.run_stream(task="概述 https://en.wikipedia.org/wiki/Seattle 的内容"):
    #     print(message)



if __name__ == '__main__':
    # 运行main
    asyncio.run(main())

