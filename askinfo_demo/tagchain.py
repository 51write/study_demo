##假设我们录入的信息有姓名、地址和邮件
##通过标记链实现对客户输入的标识处理
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from enum import Enum
#from langchain.chains.openai_functions import(
#    create_tagging_chain,
#    create_tagging_chain_pydantic,
#)
import os
from langchain_core.exceptions import OutputParserException
from config import set_environment
set_environment()

## 千帆模型去兼容OepnAI
# 设置千帆大模型平台的参数，调试不通，丢弃
def getOpenAI():
    api_key = os.environ['QIANFAN_API_KEY']  # 替换为您的Bearer Token
    base_url = "https://qianfan.baidubce.com/v2"
    default_headers = {
        "appid": "b7c9a832-61ed-480d-818e-11464efbedd8"  # 替换为您的应用ID，非必传
    }
    os.environ["OPENAI_API_KEY"] = "FAKE_KEY"
    # 需要将 OPENAI 的 API 请求重定向到本地的服务
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8001/v1"  # 修改为本地 OpenAI API 地址

    # 创建OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
    # 调用模型服务
    completion = client.chat.completions.create(
    model="ernie-4.0-turbo-8k",  # 选择您要使用的模型
    messages=[{"role": "system","content": "You are a helpful assistant."},{"role": "user", "content": "Hello!"}]
    )
    print(completion.choices[0].message)

## schema文件
## function的注释要写上，不然有出现以下错误code: 336003, msg: functions description can't be blank
class PersonalDetails(BaseModel):
    """
    PersonalDetails 模型用于存储个人的详细信息。

    包含以下字段：
    - name: 姓名
    - city: 城市
    - email: 电子邮件地址
    """
    #定义数据的类型
    name:  str = Field(
        description="这是用户输入的名字"
    )
    city: str = Field(
        description="这是用户输入的居住城市"
    )
    email:str = Field(
        description="这是用户输入的邮箱地址"
    )

def getQianFanModel():
    qianfan_api_key = os.environ['QIANFAN_API_KEY']
    qianfan_secret_key= os.environ['QIANFAN_SECRET_KEY']
    llm = QianfanChatEndpoint(api_key=qianfan_api_key, secret_key=qianfan_secret_key,model = 'ERNIE-4.0-Turbo-8K-Latest')##ERNIE-4.0-Turbo-8K-Latestd这要注意选择Turbo模型，不同的模型回复模型是不一样的。
    return llm

#因为使用的是国内的模型，这里有格式转换的问题，需要用如下代码解决
#prompt_template = ChatPromptTemplate.from_messages([
#    ("system", "你是一个数据分析工具，必须严格按照以下 JSON 格式输出结果：""{{\"name\": \"...\", \"city\": \"...\",\"email\": \"...\"}}"),
#    ("human", "{input}")
#    ])

#以下方法被丢弃了
#chain = create_tagging_chain_pydantic(PersonalDetails,llm = getQianFanModel())

#from langchain_core.utils.function_calling import convert_to_openai_tool
#dict_schema = convert_to_openai_tool(PersonalDetails)

llm = getQianFanModel()
structured_llm = llm.with_structured_output(PersonalDetails)#调试的时候建议,include_raw=True

test_str1 = "你好，我是黄国华，我住在深圳福田，我的邮箱是jeremy_h_shenzhen@163.com,请告诉我我的个人信息"
#try:
#test_res1 = chain.invoke(input = test_str1)
test_res1=structured_llm.invoke(test_str1)
print(test_res1)
#except OutputParserException as e:
#    print(f"OutputParserException: {e}")