from tagchain import get_structured_llm
from taskOne import ask_for_info
from taskTwo import decide_ask
from schema import PersonalDetails

#decide_ask(ask_for_info,)

user_999_personal_details = PersonalDetails(name="",city="",email="")
decide_ask(ask_for_info, ask_for=["name","city","email"])

str999 = "我的名字是999"
#user_999_personal_details,ask_for_999 = filter_response()
