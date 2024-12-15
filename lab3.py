import boto3
import json
import copy
import pandas as pd
from termcolor import colored
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
bedrock_embedding = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0',region_name = "us-east-1")
test_embedding = bedrock_embedding.embed_documents(['I love programing'])
print(f"The embedding dimension is {len(test_embedding[0])}, first 10 elements are: {test_embedding[0][:10]}")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
category_definition = "data/categories.csv"
categories = pd.read_csv(category_definition)

# 加载客户评论数据
comments_filepath = "data/comments.csv"
comments = pd.read_csv(comments_filepath)
print(comments)

# 加载示例数据
examples_filepath = "data/examples_with_label.csv"
examples_df = pd.read_csv(examples_filepath) 

from langchain_chroma import Chroma
# You can uncomment below code to reset the vector db, if you want to retry more times
# vector_store.reset_collection()  
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=bedrock_embedding,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)

from uuid import uuid4
import hashlib
from langchain_core.documents import Document

# 构建 langchain 文档
documents = []
for comment,groundtruth in examples_df.values:
    documents.append(
        Document(
        page_content=comment,
        metadata={"groundtruth":groundtruth}
        )
    )

# 将文档添加到矢量存储
hash_ids = [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in documents]
vector_store.add_documents(documents=documents, ids=hash_ids)

query = comments['comment'].sample(1).values[0]
print(colored(f"******query*****:\n{query}","blue"))

results = vector_store.similarity_search_with_relevance_scores(query, k=4)
print(colored("\n\n******results*****","green"))
for res, score in results:
    print(colored(f"* [SIM={score:3f}] \n{res.page_content}\n{res.metadata}","green"))

import boto3
import json
from botocore.exceptions import ClientError
import dotenv
import os
dotenv.load_dotenv()

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage,AIMessage,SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser,XMLOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder,HumanMessagePromptTemplate


class ChatModelNova(BaseChatModel):

    model_name: str
    br_runtime : Any = None
    ak: str = None
    sk: str = None
    region:str = "us-east-1"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

        if not self.br_runtime:
            if self.ak and self.sk:
                self.br_runtime = boto3.client(service_name = 'bedrock-runtime',
                                               region_name = self.region,
                                              aws_access_key_id = self.ak,
                                               aws_secret_access_key = self.sk
            
                                              )

            else:
                self.br_runtime = boto3.client(region_name = self.region,service_name = 'bedrock-runtime')
            
        
        new_messages = []
        system_message = ''
        for msg in messages:
            if isinstance(msg,SystemMessage):
                system_message = msg.content
            elif isinstance(msg,HumanMessage):
                new_messages.append({
                        "role": "user",
                        "content": [ {"text": msg.content}]
                    })
            elif isinstance(msg,AIMessage):
                new_messages.append({
                        "role": "assistant",
                        "content": [ {"text": msg.content}]
                    })

        
        temperature = kwargs.get('temperature',0.5)
        maxTokens = kwargs.get('max_tokens',3000)

        #Base inference parameters to use.
        inference_config = {"temperature": temperature,"maxTokens":maxTokens}


        # Send the message.
        response = self.br_runtime.converse(
            modelId=self.model_name,
            messages=new_messages,
            system=[{"text" : system_message}] if system_message else [],
            inferenceConfig=inference_config
        )
        output_message = response['output']['message']

        message = AIMessage(
            content=output_message['content'][0]['text'],
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                **response['usage']
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if not self.br_runtime:
            if self.ak and self.sk:
                self.br_runtime = boto3.client(service_name = 'bedrock-runtime',
                                               region_name = self.region,
                                              aws_access_key_id = self.ak,
                                               aws_secret_access_key = self.sk
            
                                              )

            else:
                self.br_runtime = boto3.client(service_name = 'bedrock-runtime')
            
        
        new_messages = []
        system_message = ''
        for msg in messages:
            if isinstance(msg,SystemMessage):
                system_message = msg.content
            elif isinstance(msg,HumanMessage):
                new_messages.append({
                        "role": "user",
                        "content": [ {"text": msg.content}]
                    })
            elif isinstance(msg,AIMessage):
                new_messages.append({
                        "role": "assistant",
                        "content": [ {"text": msg.content}]
                    })

        
        temperature = kwargs.get('temperature',0.5)
        maxTokens = kwargs.get('max_tokens',3000)

        #Base inference parameters to use.
        inference_config = {"temperature": temperature,"maxTokens":maxTokens}

        # Send the message.
        streaming_response = self.br_runtime.converse_stream(
            modelId=self.model_name,
            messages=new_messages,
            system=[{"text" : system_message}] if system_message else [],
            inferenceConfig=inference_config
        )
        # Extract and print the streamed response text in real-time.
        for event in streaming_response["stream"]:
            if "contentBlockDelta" in event:
                text = event["contentBlockDelta"]["delta"]["text"]
                # print(text, end="")
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=[{"type":"text","text":text}]))

                if run_manager:
                    # This is optional in newer versions of LangChain
                    # The on_llm_new_token will be called automatically
                    run_manager.on_llm_new_token(text, chunk=chunk)

                yield chunk
            if 'metadata' in event:
                metadata = event['metadata']
                # Let's add some other information (e.g., response metadata)
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content="", response_metadata={**metadata})
                )
                if run_manager:
                    run_manager.on_llm_new_token('', chunk=chunk)
                yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model_name": self.model_name,
        }

llm = ChatModelNova(region_name="us-east-1", model_name="amazon.nova-pro-v1:0")

# 定义提示模板
system = """You are a professional  customer feedback analyst. Your daily task is to categorize user feedback.
You will be given an input in the form of a JSON array. Each object in the array contains a comment ID and a 'c' field representing the user's comment content.
Your role is to analyze these comments and categorize them appropriately.
Please note:
1. Only output valid XML format data.
2. Do not include any explanations or additional text outside the XML structure.
3. Ensure your categorization is accurate and consistent.
4. If you encounter any ambiguous cases, use your best judgment based on the context provided.
5. Maintain a professional and neutral tone in your categorizations.
"""

user = """
Please categorize user comments according to the following category tags library:
<categories>
{tags}
</categories>

Here are examples for your to categorize:
<examples>
{examples}
<examples>

Please follow these instructions for categorization:
<instruction>
1. Categorize each comment using the tags above. If no tags apply, output "Invalid Data".
2. Summarize the comment content in no more than 50 words. Replace any double quotation marks with single quotation marks.
</instruction>

Below are the customer comments records to be categorized. The input is an array, where each element has an 'id' field representing the complaint ID and a 'c' field summarizing the complaint content.
<comments>
{input}
</comments>

For each record, summarize the comment, categorize according to the category explainations, and return the  ID, summary , reasons for tag matches, and category.

Output format example:
<output>
  <item>
    <id>xxx</id>
    <summary>xxx</summary>
    <reason>xxx</reason>
    <category>xxx</category>
  </item>
</output>

Skip the preamble and output only valid XML format data. Remember:
- Avoid double quotation marks within quotation marks. Use single quotation marks instead.
- Replace any double quotation marks in the content with single quotation marks.
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
def format_docs(docs):
    formatted = "\n".join([f"<content>{doc.page_content}</content>\n<cateogry>{doc.metadata['groundtruth']}</cateogry>" for doc in docs])
    # print(colored(f"*****retrived examples******:\n{formatted}","yellow"))
    return formatted

retrieved_docs = (retriever|format_docs).invoke("my phone always freeze")

prompt = ChatPromptTemplate([
    ('system',system),
    ('user',user)],
partial_variables={'tags':categories['mappings'].values}
)

chain = (
    {"examples":retriever|format_docs,"input":RunnablePassthrough()}
    | prompt
    | llm
    | XMLOutputParser()
)

sample_data = [str({"id":'s_'+str(i),"comment":x[0]}) for i,x in enumerate(comments.values)]
print(sample_data[:3])

import math,json
from json import JSONDecodeError

resps = []


def retry_call(chain,args: Dict[str,Any],times:int=3):
    """
      Retry mechanism to ensure the success rate of final structure output 
    """
    try:
        content = chain.invoke(args)
        if 'output' not in content:
            raise (JSONDecodeError('KeyError: output'))
        return content
    except Exception as e:
        if times:
            print(f'Exception, return [{times}]')
            return retry_call(chain,args,times=times-1)
        else:
            raise(JSONDecodeError(e))

for i in range(len(sample_data)):
    data = sample_data[i]
    # print(colored(f"*****input[{i}]*****:\n{data}","blue"))

    # resp = chain.invoke(data)
    resp = retry_call(chain,data)
    print(colored(f"*****response*****\n{resp}","green"))
    # resps += json.loads(resp)
    for item in resp['output']:
        row={}
        for it in item['item']:
            row[list(it.keys())[0]]=list(it.values())[0]
        resps.append(row)

# 检查所有数据是否已处理
assert len(resps) == len(sample_data), "Due to the uncertainly of LLM generation, the JSON output might fail occasionally, if you happen to experience error, please re-run above cell again"

# 转换为pandas数据框
prediction_df = pd.DataFrame(resps).rename(columns={"category":"predict_label"}).drop_duplicates(['id']).reset_index(drop='index')
# convert the label value to lowercase
prediction_df['predict_label'] = prediction_df['predict_label'].apply(lambda x: x.strip().lower().replace("'",""))

# 合并数据
ground_truth = comments.copy()
# convert the label value to lowercase
ground_truth['groundtruth'] = ground_truth['groundtruth'].apply(lambda x: x.strip().lower())
merge_df=pd.concat([ground_truth,prediction_df],axis=1)
print(merge_df)

# 计算准确率
def check_contains(row):
    return row['groundtruth'] in row['predict_label']
matches = merge_df.apply(check_contains, axis=1)
count = matches.sum()
print(f"correct: {count}")
print(f"accuracy: {count/len(merge_df)*100:.2f}%")

# 保存结果
# 保存的csv文件将作为lab_4的实验文件
merge_df.to_csv('result_lab_3.csv',index=False)