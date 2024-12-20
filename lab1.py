import boto3
import json
import copy
import pandas as pd
from termcolor import colored
# create clients of bedrock
bedrock = boto3.client(service_name='bedrock', region_name = 'us-east-1')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name = 'us-east-1') 
pd.set_option('display.max_rows', None)
results = []

available_models = bedrock.list_foundation_models()
for model in available_models['modelSummaries']:
    if 'Amazon' in model['providerName'] and 'TEXT' in model['outputModalities']: 
        results.append({
            'Model Name': model['modelName'],
            'Model ID': model['modelId'],  # Add Model ID column
            'Provider': model['providerName'],
            'Input Modalities': ', '.join(model['inputModalities']),
            'Output Modalities': ', '.join(model['outputModalities']),
            'Streaming': model.get('responseStreamingSupported', 'N/A'),
            'Status': model['modelLifecycle']['status']
        })

df = pd.DataFrame(results)

pd.reset_option('display.max_rows')
print(df)

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
                self.br_runtime = boto3.client(region_name = self.region, service_name = 'bedrock-runtime')
            
        
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

        
        temperature = kwargs.get('temperature',0.1)
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
                self.br_runtime = boto3.client(service_name = 'bedrock-runtime', region_name = self.region)
            
        
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

        
        temperature = kwargs.get('temperature',0.1)
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

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
llm.invoke(messages)
print(llm.invoke(messages))

import pandas as pd
comments_filepath = "data/comments.csv"
comments = pd.read_csv(comments_filepath)

category_definition = "data/categories.csv"
categories = pd.read_csv(category_definition)
print(categories)

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

Please follow these instructions for categorization:
<instruction>
1. Categorize each comment using the tags above. If no tags apply, output "Others".
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

prompt = ChatPromptTemplate([
    ('system',system),
    ('user',user),
    ],
    partial_variables={'tags':categories['mappings'].values}
)
chain = prompt | llm | XMLOutputParser()
sample_data = [str({"id":'s_'+str(i),"comment":x[0]}) for i,x in enumerate(comments.values)]
print("\n".join(sample_data[:3]))

import math,json,time
from termcolor import colored
batch_size = 20
batch = math.ceil(comments.shape[0]/batch_size)
i = 0
resps = []
for i in range(batch):
    print(colored(f"****[{i}]*****\n","blue"))
    data = sample_data[i*batch_size:(i+1)*batch_size]
    resp = chain.invoke(data)
    print(colored(f"****response*****\n{resp}","green"))
    for item in resp['output']:
        row={}
        for it in item['item']:
            row[list(it.keys())[0]]=list(it.values())[0]
        resps.append(row)
    time.sleep(10)

prediction_df = pd.DataFrame(resps).rename(columns={"category":"predict_label"}).drop_duplicates(['id']).reset_index(drop='index')
# convert the label value to lowercase
prediction_df['predict_label'] = prediction_df['predict_label'].apply(lambda x: x.strip().lower().replace("'",""))

ground_truth = comments.copy()
# convert the label value to lowercase
ground_truth['groundtruth'] = ground_truth['groundtruth'].apply(lambda x: x.strip().lower())
merge_df=pd.concat([ground_truth,prediction_df],axis=1)

# 计算准确率
def check_contains(row):
    return str(row['groundtruth']) in str(row['predict_label'])
matches = merge_df.apply(check_contains, axis=1)
count = matches.sum()
print(colored(f"accuracy: {count/len(merge_df)*100:.2f}%","green"))

# 列出所有错误分类的记录
def check_not_contains(row):
    return str(row['groundtruth']) not in str(row['predict_label'])
merge_df[merge_df.apply(check_not_contains, axis=1)]

# 保存结果
merge_df.to_csv('result_lab_1.csv',index=False)

