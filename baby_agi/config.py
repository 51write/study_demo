import os 

QIANFAN_API_KEY = 'n5yZqaRArd2DK888PFJfRWrm'
QIANFAN_SECRET_KEY='YWqjMteSoR4KIJBtnvPzia8WkrA4gszo'
#QIANFAN_API_KEY = "bce-v3/ALTAK-Q54Ct4grc4DfZerkTSTdi/dd62cd34a0db932239736c93c3e5ac63fe2d7435"
#QIANFAN_API_KEY = 'bce-v3/ALTAK-QS8XlDs7j5BalVUQNCBYR/8bc31d1640192fcc019065e091ef79dc8f174007'
SERPER_API_KEY="ea5479904d76080654a0196fa6e479b36cc9fd4dec8a812e34b1f8794cd8b05f"
def set_environment():
    for key,value in globals().items():
        if "API" in key or "ID" in key or "KEY" in key:
            os.environ[key] = value