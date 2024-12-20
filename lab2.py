import boto3
import json
import copy
import pandas as pd
from termcolor import colored
# create clients of bedrock
bedrock = boto3.client(service_name='bedrock', region_name = "us-east-1")
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name = "us-east-1") 
pd.set_option('display.max_rows', None)
results = []

available_models = bedrock.list_foundation_models()
for model in available_models['modelSummaries']:
    if 'Amazon' in model['providerName'] and 'EMBEDDING' in model['outputModalities']: 
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
print(df)
pd.reset_option('display.max_rows')

from langchain_aws.embeddings.bedrock import BedrockEmbeddings
bedrock_embedding = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0', region_name="us-east-1")
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
comments[:3]

# 为类别创建嵌入
category_embeddings = bedrock_embedding.embed_documents(categories['mappings'].values)
print(f'Rows of category_embeddings:{len(category_embeddings)}')

# 为客户评论创建嵌入
comments_embeddings = bedrock_embedding.embed_documents(comments['comment'].values)
print(f'Rows of comments_embeddings:{len(comments_embeddings)}')

# 计算相似度
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_embeddings_batch(query_matrix, embedding_matrix, top_k=2):
    """
    Find the top k similar embeddings in the embedding matrix for multiple query vectors.
    
    :param query_matrix: Matrix of query embedding vectors (2D numpy array)
    :param embedding_matrix: Matrix of embeddings to search in (2D numpy array)
    :param top_k: Number of top similar embeddings to return for each query
    :return: Indices of top k similar embeddings and their similarity scores for each query
    """
    # Calculate cosine similarity for all queries at once
    similarities = cosine_similarity(query_matrix, embedding_matrix)
    
    # Get the indices of top k similar embeddings for each query
    top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :top_k]
    
    # Get the similarity scores for top k for each query
    top_scores = np.take_along_axis(similarities, top_indices, axis=1)
    
    return top_indices, top_scores

top_indices, top_scores = find_similar_embeddings_batch(query_matrix=comments_embeddings,
                                                        embedding_matrix=category_embeddings,
                                                       top_k=1)

predicts = [ categories.loc[i]['mappings'].values[0] for i in top_indices]
predict_labels = pd.DataFrame(predicts,columns=['predict_label'])

# 连接到 Ground Truth 
ground_truth = comments.copy()
print(ground_truth)
merge_df=pd.concat([ground_truth,predict_labels],axis=1)

# 计算准确率
def check_contains(row):
    return row['groundtruth'] in row['predict_label']
matches = merge_df.apply(check_contains, axis=1)
count = matches.sum()
print(colored(f"accuracy: {count/len(merge_df)*100:.2f}%","green"))

# 保存结果
merge_df.to_csv('result_lab_2.csv',index=False)
