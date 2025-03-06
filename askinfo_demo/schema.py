from pydantic import BaseModel, Field
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