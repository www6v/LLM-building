import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_ext.tools.http import HttpTool




# os.environ["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
# os.environ["OPENAI_API_KEY"] = "sk-nq393KS4lvCCNjscfFJDNjkwOiTc12tICQU4QZi89a6aGG0F"


# 定义LLM
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini-2024-07-18"
)


async def main() -> None:
    # 为 base64 解码工具定义 JSON 模式
    base64_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "The base64 value to decode"},
        },
        "required": ["value"],
    }

    # 为 httpbin 应用程序接口创建 HTTP 工具
    base64_tool = HttpTool(
        name="base64_decode",
        description="base64 decode a value",
        scheme="https",
        host="httpbin.org",
        port=443,
        path="/base64/{value}",
        method="GET",
        json_schema=base64_schema,
    )

    # 定义Agent
    agent = AssistantAgent(
        name="agent",
        model_client=model_client,
        system_message="你是一个乐于助人的AI智能助手。记住问题完成后回复“南哥AGI研习社”。",
        model_client_stream=True,
        tools=[base64_tool],
        reflect_on_tool_use=True
    )


    # 定义终止条件  如果提到特定文本则终止对话
    text_termination = TextMentionTermination("南哥AGI研习社")

    # 定义Team Team的类型选择为RoundRobinGroupChat
    reflection_team = RoundRobinGroupChat(participants=[agent], termination_condition=text_termination, max_turns=None)

    # 1、运行team并使用官方提供的Console工具以适当的格式输出
    stream = reflection_team.run_stream(task="能否对“YWJjZGU=”进行 base64 解码?")
    await Console(stream)

    # # 2、运行team并使用流式输出,自己处理消息并将其流到前端
    # async for message in reflection_team.run_stream(task="能否对“YWJjZGU=”进行 base64 解码?"):
    #     print(message)



if __name__ == '__main__':
    # 运行main
    asyncio.run(main())

