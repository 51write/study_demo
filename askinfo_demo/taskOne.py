##实现对话式表单(和正常的人类提问，变为AI提问，人类回答)
"""
    核心任务有两个：
    1、 需要让大语言模型只负责提问，而不进行回答，同时限制问题的问题(这个工具的职责)
    2、 程序需要根据用户的回答来更新数据库和下一个问题
"""

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
from tagchain import getQianFanModel
from config import set_environment
set_environment()
"""
    实现让大模型只提问不回答
    模板的核心是:
    1) 大语言模型应该扮演前台的角色，并询问用户的个人信息
    2) 大模型不应该和用户打招呼，只需要解决需要哪些信息
    3) 所有大语言模型的输出都应该是问题
    4) 大语言模型应该从ask_for列表中随机选择一个项目进行提问
"""
def ask_for_info(ask_for=["name","city","email"]):
    #定义一个提问词模板
    first_prompt = ChatPromptTemplate.from_template(
        """
        假设你现在是一个前台，你现在需要对用户进行询问他个人的具体信息。
        不要跟用户打招呼！你可以解释你需要什么信息。不要说"你好！"！
        接下来你和用户之间的对话都是你来提问，凡是你说的都是问句。
        你每次随机选择{ask_for}列表中的一个项目，向用户提问。
        比如["name","city"]列表，你可以随机选择一个"name",
        你的问题就是"请问你的名字是?"
        """
    )
    
    info_gathering_chain = LLMChain(llm=getQianFanModel(),prompt = first_prompt)
    chat_chain = info_gathering_chain.run(ask_for = ask_for)

    return chat_chain

#def getQianFanModel():
#    qianfan_api_key = os.environ['QIANFAN_API_KEY']
#    qianfan_secret_key= os.environ['QIANFAN_SECRET_KEY']
#    llm = QianfanChatEndpoint(api_key=qianfan_api_key, secret_key=qianfan_secret_key,model = 'ERNIE-4.0-Turbo-8K-Latest')##ERNIE-4.0-Turbo-8K-Latestd这要注意选择Turbo模型，不同的模型回复模型是不一样的。
#    return llm

#print(ask_for_info(ask_for=["name","city","email"]))