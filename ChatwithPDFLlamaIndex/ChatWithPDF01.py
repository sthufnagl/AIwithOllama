
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex

# https://github.com/run-llama/llama_index/issues/10670
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# https://docs.llamaindex.ai/en/latest/module_guides/supporting_modules/service_context_migration.html
from llama_index.core import Settings

Settings.llm = Ollama(model="mistral:7b-text-fp16", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# llm = Ollama(model="mistral:7b-text-fp16", request_timeout=30.0)
# resp = llm.complete("Who is Paul Graham?")
# print(resp)

#from llama_index.tools import QueryEngineTool, ToolMetadata
#from llama_index.query_engine import SubQuestionQueryEngine
#from langchain.schema.cache import BaseCache

#llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=-1)
#service_context = ServiceContext.from_defaults(llm=llm)

lyft_docs = SimpleDirectoryReader(input_files=["lyft_2021.pdf"]).load_data()
uber_docs = SimpleDirectoryReader(input_files=["uber_2021.pdf"]).load_data()

print(lyft_docs[0].get_content())
print(uber_docs[0].get_content())

# setup baseline index
base_index = VectorStoreIndex.from_documents(lyft_docs + uber_docs)
base_engine = base_index.as_query_engine(similarity_top_k=4)

response = base_engine.query("What are some risk factors for Uber?")
print(str(response))