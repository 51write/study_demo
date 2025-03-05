from langchain_community.chat_models import QianfanChatEndpoint
import os
from config import set_environment
set_environment()
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate

qianfan_api_key = os.environ['QIANFAN_API_KEY']
qianfan_secret_key= os.environ['QIANFAN_SECRET_KEY']

class PersonalDetails(BaseModel):
    name:  str = Field(
        description="这是用户输入的名字"
    )
    city: str = Field(
        description="这是用户输入的居住城市"
    )
    email:str = Field(
        description="这是用户输入的邮箱地址"
    )

# 初始化千帆大模型（例如对文心 ERNIE-Speed-8K 测试）
qianfan = QianfanChatEndpoint(
    qianfan_ak=qianfan_api_key,
    qianfan_sk=qianfan_secret_key,
    model="ERNIE-Speed-8K"
)

test_prompt = """返回如下 JSON 格式：
{
    "name": "黄国华", 
    "city": "深圳",
    "email": "330526235@qq.com"
}
文本内容：{input}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个数据分析工具，必须严格按照以下 JSON 格式输出结果：""{{\"name\": \"...\", \"city\": \"...\",\"email\": \"...\"}}"),
    ("human", "{input}")
    ])

response = qianfan.invoke(prompt_template.format(input="你好，我是黄国华，我住在深圳福田，我的邮箱是jeremy_h_shenzhen@163.com"))
print("千帆模型原始输出:", response.content) 

# 使用更鲁棒的 JSON 解析器
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser(pydantic_object=PersonalDetails)
# 创建链路
chain = prompt_template | qianfan | parser  # 使用管道语法简化