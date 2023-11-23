from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv 
load_dotenv()

index= FAISS.load_local("faiss-jshs-history", OpenAIEmbeddings())

llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  

chain = load_qa_chain(llm_model, verbose=False)

query = "현재 교장은 ? "
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)#현재 교장에 대한 답 저장
print( query,res)

query = "1회 졸업 인원수 ? "
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)


query = "초대 교장은? "
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)

query = "제주과고 20년사 발간일은?"
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)
 