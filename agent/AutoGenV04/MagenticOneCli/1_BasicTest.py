import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console



os.environ["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
os.environ["OPENAI_API_KEY"] = "sk-COUbOl2EH7IYaz6SqFADUp7Ie4WA4rzFvNr3PzIAZjVyHQcJ"


# 定义LLM
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini-2024-07-18"
)


async def main() -> None:
    # 定义Agent
    assistant = AssistantAgent(
        name="Assistant",
        model_client=model_client,
        system_message="你是一个乐于助人的AI智能助手。"
    )


    # 定义终止条件  如果提到特定文本则终止对话
    text_termination = TextMentionTermination("南哥AGI研习社")
    # 定义终止条件，在5条信息后停止任务
    max_message_termination = MaxMessageTermination(5)
    # 使用`|` 运算符组合终止条件，在满足任一条件时停止任务
    termination = text_termination | max_message_termination

    # 定义team
    team = MagenticOneGroupChat(participants=[assistant], model_client=model_client, termination_condition=termination, max_turns=None)

    # 运行team并使用官方提供的Console工具以适当的格式输出
    stream = team.run_stream(task="写一首关于秋季的短诗")
    await Console(stream)



if __name__ == '__main__':
    asyncio.run(main())

