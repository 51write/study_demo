from PyPDF2 import PdfReader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils import getCurrentDir
from config import set_environment
import os
from langchain.chains.question_answering import load_qa_chain
from qianfan import Qianfan
from langchain.chains import RetrievalQA
set_environment()

##这个案例是学习langchain 的LEDVR工作流的例子
##======================================

def getText()->str:
    doc_reader = PdfReader(getCurrentDir() + '\\' + 'impromptu-rh.pdf')
    raw_text=''
    for i,page in enumerate(doc_reader.pages):
        #if i < 5:#只拿前面8页        
        text = page.extract_text()
        if text:
            raw_text += text

    return raw_text

#这个方法实际上不能产生作用
def generate_embeddings(texts, endpoint):
    embeddings = []
    for text in texts:
        try:
            embedding = endpoint(text)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing text: {text[:50]}... - {e}")
    return embeddings

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
)

texts = text_splitter.split_text(getText())

qianfan_api_key = os.environ['QIANFAN_API_KEY']
qianfan_secret_key= os.environ['QIANFAN_SECRET_KEY']

#必须使用model='bge-large-zh',否则会提示prompt提示词长度过长的失败
embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=qianfan_api_key, qianfan_sk=qianfan_secret_key,model='bge-large-zh')
docsearch = FAISS.from_texts(texts,embeddings)

query = "Humanity"
docs = docsearch.similarity_search(query)
print(str(docs))

##以上是处理PDF,以下是开始创建问答链
#chain = load_qa_chain(Qianfan(access_key=qianfan_api_key, secret_key=qianfan_secret_key),hain_type='map_rerank',return_intermediate_steps=True) #return_intermediate_steps=True是对文档进行打分
#query ='openAi的创始人是谁?'
#docs = docsearch.similarity_search(query, k=10)
#results = chain({"input_documents":docs, "question":query},return_only_outputs=True)
#print(results['output_text'])

llm = QianfanChatEndpoint(api_key=qianfan_api_key, secret_key=qianfan_secret_key)
retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":4})
rqa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=True)
print(rqa('OpenAI是什么'))