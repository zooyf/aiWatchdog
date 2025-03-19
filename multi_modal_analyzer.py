# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:03:32 2025

@author: 18523
"""

import base64
import requests
import cv2 
import time 
import numpy as np
import time
import json
import httpx
import os
import datetime
import asyncio

from utility import video_chat_async,chat_request,insert_txt,video_chat_async_limit_frame
from config import RAGConfig

# 从配置文件加载提示词
# with open('prompts.json', 'r', encoding='utf-8') as f:
#     prompts = json.load(f)

# prompt_detect = prompts['prompt_detect']
# prompt_summary = prompts['prompt_summary']
# prompt_vieo = prompts['prompt_video']

from prompt import prompt_detect,prompt_summary,prompt_vieo


class MultiModalAnalyzer:
    def __init__(self):
        self.message_queue = []
        self.time_step_story = []

    def trans_date(self,date_str):
        # Split the input string into components
        year, month, day, hour, minute, second = date_str.split('-')
        
        # Determine AM or PM
        am_pm = "上午" if int(hour) < 12 else "下午"
        
        # Convert 24-hour format to 12-hour format
        hour_12 = hour if hour == '12' else str(int(hour) % 12)
        
        # Return the formatted date and time string
        return f"{year}年{int(month)}月{int(day)}日{am_pm}{hour_12}点（{hour}时）{int(minute)}分{int(second)}秒"
    
    async def analyze(self, frames,fps=20,timestamps=None):
        start_time = time.time()
        
        histroy = "录像视频刚刚开始。"
        Recursive_summary = ""
        for i in self.message_queue:
            histroy = "历史视频内容总结:"+Recursive_summary+"\n\n当前时间段：" + i['start_time']+"  - " + i['end_time'] + "\n该时间段视频描述如下：" +i['description'] + "\n\n该时间段异常提醒:"+i['is_alert']
        
        time_temp = time.time()
        tasks = [chat_request(prompt_summary.format(histroy=histroy)), video_chat_async_limit_frame(prompt_vieo,frames,timestamps,fps=fps)]
        results = await asyncio.gather(*tasks)
       
        Recursive_summary =results[0]
        description = results[1]
        description_time = time.time()-time_temp

        if timestamps==None:
            return description
        
        date_flag = self.trans_date(timestamps[0])+"："
        #保存监控视频描述到数据库
        if RAGConfig.ENABLE_RAG:
            insert_txt([date_flag+description],'table_test_table')
        else:
            print("RAG未开启,准备保存到本地")
            # 本地文件保存
            with open(RAGConfig.HISTORY_FILE, 'a', encoding='utf-8') as file:
                print("开始保存历史消息")
                file.write(date_flag+description + '\n')          
                

        
        text = prompt_detect.format(Recursive_summary=Recursive_summary,current_time=timestamps[0]+"  - " + timestamps[-1],latest_description=description)
        
        time_temp = time.time()
        alert = await chat_request(text)
        alert_time = time.time() - time_temp
        
        print("警告内容：",alert)    

        print("\n\n下面是视频描述原文：")
        print(description)
        
        print("视频分析耗时",time.time() - start_time)
        print("视频描述用时,警告文本用时:",description_time,alert_time)
        
        if "无异常" not in alert:
            current_time = timestamps[0]
            file_str = f"waring_{current_time}"
            new_file_name = f"video_warning/{file_str}.mp4"
            # 重命名文件
            os.rename("./video_warning/output.mp4", new_file_name)            
            #with open("./video_warning/video_warning.txt", 'a', encoding='utf-8') as file:
            #    file.write(new_file_name+" "+alert + '\n')    
            
            frame = frames[0]
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) == 2:
                # 如果帧是灰度的，转换为BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) 
            cv2.imwrite(f"video_warning/{file_str}.jpg", frame)
            
            return {"alert":f"<span style=\"color:red;\">{alert}</span>",
                    "description":f' 当前10秒监控消息描述：\n{description}`\n\n 历史监控内容:\n{Recursive_summary}`',
                    "video_file_name":f"{file_str}.mp4",
                    "picture_file_name":f"{file_str}.jpg"}
        
        if timestamps:
            self.message_queue.append({ 
                'start_time': timestamps[0],
                'end_time': timestamps[1],
                'description': description, 
                'is_alert':alert
            })

            # 只保留最近15条消息用作历史信息总结
            self.message_queue = self.message_queue[-15:]
            
        return {"alert":"无异常"}
        