"""配置文件
包含视频监控系统的所有可配置参数
"""

from typing import Dict, Any
import logging

# 视频源配置
VIDEO_SOURCE = r'测试视频/捕鱼0.mp4'

# 视频处理配置
class VideoConfig:
    VIDEO_INTERVAL = 1800  # 视频分段时长(秒)
    ANALYSIS_INTERVAL = 10  # 分析间隔(秒)
    BUFFER_DURATION = 11  # 滑窗分析时长（实际模型分析视频时长）
    WS_RETRY_INTERVAL = 3  # WebSocket重连间隔(秒)
    MAX_WS_QUEUE = 100  # 消息队列最大容量
    JPEG_QUALITY = 70  # JPEG压缩质量



# API配置
class APIConfig:
    # 通义千问API配置
    QWEN_API_KEY = ""
    QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    QWEN_MODEL = "qwen-vl-max-2025-01-25"
    


    # Moonshot语言模型 API配置，这里可以换成其他语言模型或者本地部署的内容
    MOONSHOT_API_KEY = ""
    MOONSHOT_API_URL = "https://api.moonshot.cn/v1/chat/completions"
    MOONSHOT_MODEL = "moonshot-v1-8k"
    
    # API请求配置
    REQUEST_TIMEOUT = 60.0 # 请求超时时间（秒）
    TEMPERATURE = 0.5 # 温度
    TOP_P = 0.01 
    TOP_K = 20
    REPETITION_PENALTY = 1.05

# RAG系统配置
class RAGConfig:
    # 知识库配置
    ENABLE_RAG = False
    VECTOR_API_URL = "http://172.16.10.44:8085/add_text/" # 启用RAG需要自行搭建rag服务，并构造相应api接口
    HISTORY_FILE = "video_histroy_info.txt"  # 如果不启用RAG，历史记录将保存在该文件中

# 存档配置
ARCHIVE_DIR = "archive"

# 服务器配置
class ServerConfig:
    HOST = "0.0.0.0"
    PORT = 16532
    RELOAD = True
    WORKERS = 1

# 日志配置
LOG_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'handlers': [
        {'type': 'file', 'filename': 'code.log'},
        {'type': 'stream'}
    ]
}



def update_config(args: Dict[str, Any]) -> None:
    """使用命令行参数更新配置
    
    Args:
        args: 包含命令行参数的字典
    """
    global VIDEO_SOURCE
    
    # 更新视频源
    if args.get('video_source'):
        VIDEO_SOURCE = args['video_source']
    
    # 更新视频处理配置
    for key in ['video_interval', 'analysis_interval', 'buffer_duration',
               'ws_retry_interval', 'max_ws_queue', 'jpeg_quality']:
        if key in args:
            setattr(VideoConfig, key.upper(), args[key])
    
    # 更新服务器配置
    for key in ['host', 'port', 'reload', 'workers']:
        if key in args:
            setattr(ServerConfig, key.upper(), args[key])
            
    # 更新API配置
    for key in ['qwen_api_key', 'qwen_api_url', 'qwen_model',
               'moonshot_api_key', 'moonshot_api_url', 'moonshot_model',
               'request_timeout', 'temperature', 'top_p', 'top_k',
               'repetition_penalty']:
        if key in args:
            setattr(APIConfig, key.upper(), args[key])
            
    # 更新RAG配置
    for key in ['enable_rag', 'vector_api_url', 'history_file',
               'history_save_interval']:
        if key in args:
            setattr(RAGConfig, key.upper(), args[key])


