"""
    核心任务有两个：
    1、 需要让大语言模型只负责提问，而不进行回答，同时限制问题的问题
    2、 程序需要根据用户的回答来更新数据库和下一个问题(这个工具的职责)
"""
from schema import PersonalDetails
from taskOne import ask_for_info
def check_what_is_empty(user_personal_details:PersonalDetails):
    ask_for=[]
    #检查是否为空
    for field,value in user_personal_details.dict().items():
        if value in [None,"",0]:
            print(f"Field '{field}' 为空")
            ask_for.append(f'{field}')
    return ask_for

def add_non_empty_details(currentDetails:PersonalDetails, new_details:PersonalDetails):
    non_empty_details = {k:v for k,v in new_details.dict().items() if v not in [None,"",0]}
    update_details = currentDetails.model_copy(update=non_empty_details)
    return update_details

#user_007_personal_details = PersonalDetails(name="",city="",email="")
#ask_for = check_what_is_empty(user_007_personal_details)
#print(ask_for)

#user_999_personal_details = PersonalDetails(name="",city="",email="")
#decide_ask(ask_for)

def decide_ask(func:ask_for_info, ask_for=["name","city","email"]):
    if ask_for:
        ai_res = func(ask_for=ask_for)
        print(ai_res)
    else:
        print("填充完毕")