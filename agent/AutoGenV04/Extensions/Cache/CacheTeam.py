import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache





os.environ["OPENAI_BASE_URL"] = "https://yunwu.ai/v1"
os.environ["OPENAI_API_KEY"] = "sk-Rj3IWhZU9DZjoyy8AQ2Q3Fo82qlKApSFtYsu40EOtH9GCytV"


# 定义LLM
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini-2024-07-18"
)

# 将 diskcache.Cache 封装为 DiskCacheStore，后者是一个特定类型的缓存存储，这里专门用于缓存聊天响应
# Cache(tmpdirname)将数据存储在 tmpdirname 目录中
cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache("tmpdirname"))

# 创建 ChatCompletionCache 对象，它将 openai_model_client 和 cache_store 作为参数
# 这样做的目的是，当请求模型生成内容时，缓存客户端会先检查是否已有缓存的响应。如果有，则直接返回缓存的响应，否则发起 API 请求
cache_client = ChatCompletionCache(model_client, cache_store)


async def main() -> None:
    # 定义Agent
    agent = AssistantAgent(
        name="agent",
        model_client=cache_client,
        description="你是一个乐于助人的AI智能助手。",
        system_message="你是一个乐于助人的AI智能助手。任务完成后，回复南哥AGI研习社"
    )

    # 定义终止条件  如果提到特定文本则终止对话
    termination = TextMentionTermination("南哥AGI研习社")

    # 定义Team Team的类型选择为RoundRobinGroupChat
    agent_team = RoundRobinGroupChat(participants=[agent], termination_condition=termination, max_turns=None)

    # run_stream()方法运行team并使用官方提供的Console工具以适当的格式输出
    stream = agent_team.run_stream(task="Hello, how are you?")
    await Console(stream)

    # stream = agent_team.run_stream(task="Hello, how are you?")
    # await Console(stream)



if __name__ == '__main__':
    # 运行main
    asyncio.run(main())

