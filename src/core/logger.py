import logging

# 1. 创建 Logger 实例
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别

# 2. 定义日志格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 3. 创建控制台 Handler（输出到命令行）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 设置控制台日志级别
console_handler.setFormatter(formatter)  # 应用格式

# 4. 创建文件 Handler（输出到文件）
file_handler = logging.FileHandler('app.log')  # 日志文件路径
file_handler.setLevel(logging.INFO)  # 设置文件日志级别
file_handler.setFormatter(formatter)  # 应用格式

# 5. 将 Handler 添加到 Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 测试日志输出
logger.info("这是一条 INFO 级别的日志，会同时输出到控制台和文件！")
logger.error("这是一条 ERROR 级别的日志，同样会输出到两边。")
