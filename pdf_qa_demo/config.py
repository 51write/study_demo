import os 

QIANFAN_API_KEY = 'n5yZqaRArd2DK888PFJfRWrm'
QIANFAN_SECRET_KEY='YWqjMteSoR4KIJBtnvPzia8WkrA4gszo'

def set_environment():
    for key,value in globals().items():
        if "API" in key or "ID" in key or "KEY" in key:
            os.environ[key] = value