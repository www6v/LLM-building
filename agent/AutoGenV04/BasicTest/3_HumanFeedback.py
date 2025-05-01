import os
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console




# os.environ["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
# os.environ["OPENAI_API_KEY"] = "sk-g3xus7CKnnGTKKzL7DD26BjQXRUAwkzDwRk2gu9oLfGTKO9a"


# 定义LLM
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini-2024-07-18"
)


# 定义函数  模拟用户反馈
def custom_input(task):
    return "上海正在下雨,天气很糟糕。"


async def main() -> None:
    # 定义Agent
    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        description="一个和用户交互的智能体。"
    )

    # 定义Agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        input_func=custom_input
    )

    # 定义终止条件  如果提到特定文本则终止对话
    text_termination = TextMentionTermination("南哥AGI研习社")
    # 定义终止条件，在5条信息后停止任务
    max_message_termination = MaxMessageTermination(4)
    # 使用`|` 运算符组合终止条件，在满足任一条件时停止任务
    termination = text_termination | max_message_termination

    # 定义Team Team的类型选择为RoundRobinGroupChat
    agent_team = RoundRobinGroupChat(participants=[weather_agent,user_proxy], termination_condition=termination, max_turns=None)

    # 运行team并使用官方提供的Console工具以适当的格式输出
    stream = agent_team.run_stream(task="上海的天气如何?")
    await Console(stream)


if __name__ == '__main__':
    # 运行main
    asyncio.run(main())