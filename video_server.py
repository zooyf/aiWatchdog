"""智能视频监控系统 (2025.02.26版)
核心功能：
1. 实时视频流采集与缓冲 
2. 智能多模态异常检测 
3. 视频分段存储与特征归档 
4. WebSocket实时警报推送 
"""
 
import cv2 
import asyncio 
import json 
import argparse
from datetime import datetime 
from concurrent.futures import ThreadPoolExecutor 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from collections import deque 
from typing import Optional, Dict, Any 
import numpy as np 
import logging 
from multi_modal_analyzer import MultiModalAnalyzer
import time
import uvicorn 
from multiprocessing import set_start_method 
from config import VideoConfig, ServerConfig, VIDEO_SOURCE, LOG_CONFIG, ARCHIVE_DIR, update_config

# 配置日志记录
logging.basicConfig(
    level=LOG_CONFIG['level'],
    format=LOG_CONFIG['format'],
    handlers=[logging.FileHandler(LOG_CONFIG['handlers'][0]['filename'], encoding='utf-8'), logging.StreamHandler()]
)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='智能视频监控系统')
    parser.add_argument('--video_source', type=str, help='视频源路径')
    parser.add_argument('--video_interval', type=int, help='视频分段时长(秒)')
    parser.add_argument('--analysis_interval', type=int, help='分析间隔(秒)')
    parser.add_argument('--buffer_duration', type=int, help='滑窗分析时长')
    parser.add_argument('--ws_retry_interval', type=int, help='WebSocket重连间隔(秒)')
    parser.add_argument('--max_ws_queue', type=int, help='消息队列最大容量')
    parser.add_argument('--jpeg_quality', type=int, help='JPEG压缩质量')
    parser.add_argument('--host', type=str, help='服务器主机地址')
    parser.add_argument('--port', type=int, help='服务器端口')
    parser.add_argument('--reload', type=bool, help='是否启用热重载')
    parser.add_argument('--workers', type=int, help='工作进程数')
    
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}

# 更新配置
args = parse_args()
update_config(args)




# 初始化视频源
video_source = VIDEO_SOURCE
cap = cv2.VideoCapture(video_source)     # 读取视频
for i in range(5):
    ret, frame = cap.read() 
    
width = frame.shape[1]
height = frame.shape[0] 
fps = cap.get(cv2.CAP_PROP_FPS)
cv2.destroyAllWindows()
cap.release()
print("fps",fps)

# 视频流处理器 
class VideoProcessor:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        ret, frame = self.cap.read()
        self.buffer = deque(maxlen=int(fps * VideoConfig.BUFFER_DURATION))
        self.executor = ThreadPoolExecutor()
        self.analyzer = MultiModalAnalyzer()
        self.last_analysis = datetime.now().timestamp() 
        self._running = False 
        self.lock = asyncio.Lock()
        self.frame_queue = asyncio.Queue()  # 添加一个异步队列用于缓存帧
        self.start_push_queue = 0
 
    @property 
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0 
 
    async def video_streamer(self, websocket: WebSocket):
        try:
            while True:
                #start_time = time.monotonic() 
                frame = await self.frame_queue.get()  # 从队列中获取帧
                # 压缩为JPEG格式（调整quality参数控制质量）
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), VideoConfig.JPEG_QUALITY])
                
                # 通过WebSocket发送二进制数据
                await websocket.send_bytes(buffer.tobytes())
                #elapsed = time.monotonic()  - start_time
                #await asyncio.sleep(1 / self.fps- elapsed-0.02)  # 发送的数度需要比生产的速度快，根据视频的fps来等待
                #if count%60==0:
                #    print("长度",self.frame_queue.qsize())
        except Exception as e:
            print(f"Error: {e}")
        finally:
            print("停止直播")
    
    async def frame_generator(self):
        """异步视频帧生成器"""
        count = 0
        while self._running:
            start_time = time.monotonic() 
            ret, frame = self.cap.read() 
            count = count + 1
            if not ret:
                #logging.error(" 视频流中断，尝试重新连接...")
                break 
            
            # 转换颜色空间并缓冲 
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self.buffer.append({ 
                "frame": frame,
                "timestamp": datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            })
            
            yield frame 
            
            
            if self.start_push_queue:
                await self.frame_queue.put(frame)  # 将帧放入队列

            # 控制帧生成速度

            elapsed = time.monotonic() - start_time
            await asyncio.sleep(max(0, 1/self.fps - elapsed))  # 控制帧生成速度

        #await self._reconnect()
 
    async def _reconnect(self):
        """视频流重连逻辑"""
        await asyncio.sleep(VideoConfig.WS_RETRY_INTERVAL) 
        self.cap.release() 
        self.cap = cv2.VideoCapture(self.video_source)
        ret, frame = self.cap.read()
 
    async def start_processing(self):
        """启动处理流水线"""
        self._running = True 
        count = 0
        start = time.time()
        async for frame in self.frame_generator(): 
            asyncio.create_task(archiver.write_frame(frame))
            count = count + 1
            
            # 定时触发分析 
            if (datetime.now().timestamp() - self.last_analysis) >= VideoConfig.ANALYSIS_INTERVAL and count >= fps * VideoConfig.ANALYSIS_INTERVAL:
                print("count", count)
                print("fps * interval",fps * VideoConfig.ANALYSIS_INTERVAL,fps)
                count = 0
                asyncio.create_task(self.trigger_analysis())
                self.last_analysis = datetime.now().timestamp() 
           
    async def trigger_analysis(self):
        """触发异步分析"""
        print("start")
        try: 
            async with self.lock:
                clip = list(self.buffer) 
                if not clip:
                    return 
                
                print("self.buffer:", len(clip))
                #print("clip[0]['timestamp']:", clip[0]['timestamp'])
                #print("clip[-1]['timestamp']:", clip[-1]['timestamp'])

                result = await self.analyzer.analyze([f["frame"] for f in clip], self.fps, (clip[0]['timestamp'], clip[-1]['timestamp']))
                
                if result["alert"] != "无异常":
                    await AlertService.notify(result) 
            
        except Exception as e:
                logging.error(f" 分析失败: {str(e)}")
        
# 警报服务 
class AlertService:
    _connections = set()

    @classmethod
    async def register(cls, websocket: WebSocket):
        await websocket.accept()
        cls._connections.add(websocket)

    @classmethod
    async def notify(cls, data: Dict):
        """广播警报信息"""
        message = json.dumps({
            "timestamp": datetime.now().isoformat(),
            **data
        })

        for conn in list(cls._connections):
            try:
                if conn.client_state == WebSocketState.CONNECTED:
                    await conn.send_text(message)
                else:
                    cls._connections.remove(conn)
            except Exception as e:
                logging.warning(f"推送失败: {str(e)}")
                cls._connections.remove(conn)

# 视频存储服务 
class VideoArchiver:
    def __init__(self):
        self.current_writer: Optional[cv2.VideoWriter] = None 
        self.last_split = datetime.now() 
 
    async def write_frame(self, frame: np.ndarray): 
        """异步写入视频帧"""
        if self._should_split():
            self._create_new_file()
 
        if self.current_writer is not None:
            self.current_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
    def _should_split(self) -> bool:
        return (datetime.now() - self.last_split).total_seconds() >= VideoConfig.VIDEO_INTERVAL 
 
    def _create_new_file(self):
        if self.current_writer is not None:
            self.current_writer.release() 
 
        filename = f"{ARCHIVE_DIR}/{datetime.now().strftime('%Y%m%d_%H%M')}.mp4" 
        self.current_writer = cv2.VideoWriter(
            filename, 
            cv2.VideoWriter_fourcc(*'avc1'), 
            fps, 
            (width, height)
        )
        self.last_split = datetime.now() 
 
# FastAPI应用配置 
app = FastAPI(title="智能视频监控系统")
processor = VideoProcessor(video_source)
archiver = VideoArchiver()
 
@app.on_event("startup") 
async def startup():
    asyncio.create_task(processor.start_processing()) 
 
@app.websocket("/alerts") 
async def alert_websocket(websocket: WebSocket):
    await AlertService.register(websocket) 
    try:
        while True:
            await websocket.receive_text()   # 维持连接 
    except Exception:
        pass 

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    try:
        await websocket.accept()
        processor.start_push_queue = 1
        await processor.video_streamer(websocket)
        
    except WebSocketDisconnect:
        print("Client disconnected from video feed")
        processor.start_push_queue = 0
        processor.frame_queue = asyncio.Queue()
    except Exception as e:
        print(f"An error occurred: {e}")
        processor.start_push_queue = 0
        processor.frame_queue = asyncio.Queue()
    finally:
        processor.start_push_queue = 0
        processor.frame_queue = asyncio.Queue()

if __name__ == "__main__":
    uvicorn.run( 
        app="video_server:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=ServerConfig.RELOAD,
        workers=ServerConfig.WORKERS
    )

# python video_server.py --video_source "./测试视频/小猫开门.mp4"