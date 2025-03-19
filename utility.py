# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:50:52 2025

@author: 18523
"""
import base64
import requests
import cv2 
import time 
import numpy as np
import json
import httpx
from config import APIConfig, RAGConfig,VideoConfig



def frames_to_base64(frames,fps,timestamps):
    print(len(frames))
    print(fps)
    width = frames[0].shape[1]
    height = frames[0].shape[0]    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    
    #filename = ":".join(timestamps).replace("-","")
    #video_writer = cv2.VideoWriter(f'./{filename}.mp4', fourcc, fps, (width, height))  
    video_writer = cv2.VideoWriter('./video_warning/output.mp4', fourcc, fps, (width, height))  
    # 遍历所有帧，并将其写入视频文件
    for frame in frames:
        # 确保帧是正确的数据类型和形状
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 2:
            # 如果帧是灰度的，转换为BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame)
    
    # 释放VideoWriter对象
    video_writer.release()

    
    with open('./video_warning/output.mp4', 'rb') as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
    
    return video_base64


#强制抽取关键帧帧，每秒一帧率
async def video_chat_async_limit_frame(text, frames,timestamps,fps=20):

    video_base64 = frames_to_base64(frames,fps,timestamps)


    #url = "http://172.16.10.44:8085/v1/chat/completions"
    url = APIConfig.QWEN_API_URL
    headers = {
        "Content-Type": "application/json",
        "authorization": APIConfig.QWEN_API_KEY
    }
    model = APIConfig.QWEN_MODEL

    data_image = []
    frame_count = int(VideoConfig.BUFFER_DURATION)
    for i in range(frame_count):
        frame = frames[(len(frames)//frame_count)*i]
        image_path = 'output_frame.jpg'
        cv2.imwrite(image_path, frame)
        with open(image_path,'rb') as file:
            image_base64 = "data:image/jpeg;base64,"+ base64.b64encode(file.read()).decode('utf-8')
        data_image.append(image_base64)
        
    content =  [{"type": "text", "text": text}] + [{"type": "image_url","image_url": {"url":i}} for  i in data_image]
      

    # 构建API请求的URL和Headers

    # 构建请求体
    data = {
        "model": model,  # 模型名称
        "vl_high_resolution_images":False,
        "messages": [
            {
                "role": "user",

                "content": content,
            }
        ],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(url, headers=headers, json=data)
        response_data = response.json()
        #print(response_data)
        return response_data['choices'][0]['message']['content']



async def video_chat_async(text, frames, timestamps, fps=20):
    video_base64 = frames_to_base64(frames, fps, timestamps)

    url = APIConfig.QWEN_API_URL
    headers = {
        "Content-Type": "application/json",
        "authorization": APIConfig.QWEN_API_KEY
    }
    model = APIConfig.QWEN_MODEL
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        }
                    }
                ]
            }
        ],
        "stop_token_ids": [151645, 151643]
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(APIConfig.REQUEST_TIMEOUT)) as client:
        response = await client.post(url, headers=headers, json=data)
        response_data = response.json()
        return response_data['choices'][0]['message']['content']


async def chat_request(message,stream=False):
    url = APIConfig.MOONSHOT_API_URL
    model = APIConfig.MOONSHOT_MODEL

    messages =[{"role" : "user", "content" :message}]
    headers = {
        "content-Type" : "application/json",
        "authorization" : APIConfig.MOONSHOT_API_KEY
    }
    data ={
        "messages" : messages,
        "model" : model,
        "repetition_penalty" : APIConfig.REPETITION_PENALTY,
        "temperature" : APIConfig.TEMPERATURE,
        "top_p": APIConfig.TOP_P,
        "top_k": APIConfig.TOP_K,
        "stream" : stream
    }
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(APIConfig.REQUEST_TIMEOUT)) as client:
        response = await client.post(url, headers=headers, json=data)
        response = response.json()
        return response['choices'][0]['message']['content']

def insert_txt(docs,table_name):
    #插入文本，同时向量化
    url = RAGConfig.VECTOR_API_URL
    """docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956."
    ]"""
    data = {
        "docs": docs,
        "table_name": table_name
    }
    response = requests.post(url, json=data)
    return response.json()
