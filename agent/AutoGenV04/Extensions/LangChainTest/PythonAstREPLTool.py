import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from autogen_ext.tools.langchain import LangChainToolAdapter



os.environ["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
os.environ["OPENAI_API_KEY"] = "sk-ImgTvNymbf3cV0mMOyuJlL1SsCj60Gmi4qcrAmG1yXodtqeS"



# 定义Tool
df = pd.read_csv("titanic.csv")
tool = LangChainToolAdapter(PythonAstREPLTool(locals={"df": df}))


async def main() -> None:
    # 定义Agent
    agent = AssistantAgent(
        name="agent",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-mini-2024-07-18"
        ),
        description="一个通过使用工具为用户提供帮助的智能体。使用`df`变量访问数据集。",
        system_message="你是一个乐于助人的AI智能助手。擅长使用工具解决任务，使用`df`变量访问数据集。任务完成后，回复南哥AGI研习社。",
        tools=[tool],
    )

    # 定义终止条件  如果提到特定文本则终止对话
    termination = TextMentionTermination("南哥AGI研习社")

    # 定义Team Team的类型选择为RoundRobinGroupChat
    agent_team = RoundRobinGroupChat(participants=[agent], termination_condition=termination, max_turns=None)

    # run_stream()方法运行team并使用官方提供的Console工具以适当的格式输出
    stream = agent_team.run_stream(task="乘客的平均年龄?")
    # stream = agent_team.run_stream(task="乘客的性别占比分别为多少?")
    await Console(stream)




if __name__ == '__main__':
    # 运行main
    asyncio.run(main())

