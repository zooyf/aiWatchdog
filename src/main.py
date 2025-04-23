import base64
from io import BytesIO
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager

import aiofiles
import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter
from fastapi import UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 确保目录存在
os.makedirs("data/images", exist_ok=True)

from core.app import create_app
from core.logger import logger

load_dotenv()

# Initialize clients
model_name = os.getenv("MODEL_NAME", "qwen2.5-vl-72b-instruct")
base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
api_key = os.getenv("OPENAI_API_KEY")
max_count = int(os.getenv("MAX_COUNT", 100))
port = int(os.getenv('BACKEND_PORT', 8879))

if not api_key:
    print('api_key not found')
    exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化逻辑
    logger.info("Application startup: initializing resources")

    yield  # 应用运行期间

    # 关闭时清理逻辑
    logger.info("Application shutdown: cleaning up")


prefix = '/vl'
app = create_app(prefix, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境应指定具体域名
    allow_credentials=False,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

api_router = APIRouter(prefix=f'{prefix}/fish')

"""
POST: 首先判断 sqlite 是否已建表, 建表后判断 count <= max_count, 否则输出{"result": "次数用尽", "remain_count": 0}.  上传图片, 保存到 data/images/uuid4-文件名; 使用 httpx 异步发送给大模型, 判断请求是否成功, 将 结果(result)/uuid4-文件名(name)/自增主键(id)/是否调用成功(call-YN)/识别结果(rec-YN)/token消耗(usage json字符串) 存储到sqlite中, 返回示例: {"result": "Y", "remain_count": 99}
GET: ?id=uuid4, 从保存的路径下载图片, 图片名不带uuid4
GET: 返回列表[{`uuid4-文件名`:识别结果}]
"""


# SQLite数据库连接
def get_sqlite_connection():
    conn = sqlite3.connect("fish_recognition.db")
    conn.row_factory = sqlite3.Row
    return conn


# 初始化SQLite数据库
def init_sqlite_db():
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fish_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        result TEXT NOT NULL,
        call_success TEXT NOT NULL,
        recognition_success TEXT NOT NULL,
        usage TEXT NOT NULL,
        completion_tokens INTEGER,
        prompt_tokens INTEGER,
        total_tokens INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()


# 保存记录到SQLite
def save_record(name, result, call_success, recognition_success, usage_data):
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    usage_dict = json.loads(usage_data)
    cursor.execute(
        "INSERT INTO fish_records (name, result, call_success, recognition_success, usage, completion_tokens, prompt_tokens, total_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (name, result, call_success, recognition_success, usage_data,
         usage_dict.get('completion_tokens', 0),
         usage_dict.get('prompt_tokens', 0),
         usage_dict.get('total_tokens', 0))
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


# 初始化数据库
init_sqlite_db()


async def process_image(file_content, filename):
    # 生成唯一文件名
    unique_filename = f"{uuid.uuid4()}-{filename}"
    dir_path = "data/images"
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/{unique_filename}"

    # 保存文件
    async with aiofiles.open(file_path, 'wb') as out_file:
        await out_file.write(file_content)

    # --- 调用图片处理辅助函数 ---
    # 目标大小设置为 200KB
    processed_image_bytes, process_message, size_check_success = compress_image_bytes_to_size(file_content, target_kb=200)
    # 可以在日志中记录 process_message
    print(f"图片大小处理结果: {process_message}")

    # 准备发送给模型的数据
    base64_image = base64.b64encode(processed_image_bytes).decode('utf-8')

    messages = [
        {"role": "system", "content": """请仔细判断以下图片是否存在至少一条死鱼，存在回复Y，否则回复N，注意只回复Y或N不要包含任何其它字
1. 检查是否有翻白肚、侧翻或僵直的鱼体；  
2. 观察鱼群是否密集堆积或沿岸边分布；  
3. 注意鱼体颜色是否褪色或局部泛白；  
4. 排除塑料袋、泡沫等非生物漂浮物干扰； 
5. 注意表面是否出现黏液、蛆虫滋生、身体膨胀、内脏外露、鳍部破损或缺失等腐败迹象"""},
        {"role": "user", "content": [
            {"type": "text", "text": "请判断这张图片中的鱼是否是死鱼，只需回答'Y'或'N'，不需要其他解释。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ]

    # 调用模型API
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.5,
                    "max_tokens": 100
                }
            )

            response_data = response.json()

            if response.status_code == 200:
                result_text = response_data.get("choices", [{}])[0].get("message", {}).get("content",
                                                                                           "").strip().upper()
                # is_dead_fish = "Y" if "Y" in result_text else "N"
                is_dead_fish = result_text
                usage_data = json.dumps(response_data.get("usage", {}))

                # 保存记录
                record_id = save_record(
                    name=unique_filename,
                    result=result_text,
                    call_success="Y",
                    recognition_success=is_dead_fish,
                    usage_data=usage_data
                )

                data = {
                    "success": True,
                    "is_dead_fish": is_dead_fish,
                    "file_name": unique_filename,
                    "record_id": record_id
                }
                logger.info(data)
            else:
                # 保存失败记录
                resp_content = ""
                if hasattr(response, "content"):
                    resp_content += response.content
                if hasattr(response, "text"):
                    resp_content += response.text
                record_id = save_record(
                    name=unique_filename,
                    result=f"API错误: {response.status_code} {resp_content}",
                    call_success="N",
                    recognition_success="N",
                    usage_data="{}"
                )

                data = {
                    "success": False,
                    "error": f"API错误: {response.status_code}",
                    "file_name": unique_filename,
                    "record_id": record_id
                }
                logger.error(data)

    except Exception as e:
        logger.exception(e)
        # 保存异常记录
        record_id = save_record(
            name=unique_filename,
            result=f"处理异常: {str(e)}{type(e)}",
            call_success="N",
            recognition_success="N",
            usage_data="{}"
        )

        data = {
            "success": False,
            "error": f"处理异常: {str(e)}{type(e)}",
            "file_name": unique_filename,
            "record_id": record_id
        }
        logger.error(data)

    # 保存文件
    if data["success"]:
        result_dir = "data/success"
    else:
        result_dir = "data/failed"
    os.makedirs(result_dir, exist_ok=True)

    async with aiofiles.open(f"{result_dir}/{unique_filename}", 'wb') as out_file:
        await out_file.write(file_content)
    return data


@api_router.get("/image/{image_id}", summary="获取图片")
async def get_image(image_id: str):
    # 从SQLite获取图片名称
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM fish_records WHERE id = ?", (image_id,))
    record = cursor.fetchone()
    conn.close()

    if not record:
        raise HTTPException(status_code=404, detail="图片不存在")

    file_name = record["name"]
    original_name = file_name.split("-", 1)[36:]  # 去掉UUID前缀
    file_path = f"data/images/{file_name}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="图片文件不存在")

    return FileResponse(file_path, filename=original_name)


@api_router.get("/list", summary="获取所有记录")
async def get_records():
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, result, recognition_success FROM fish_records ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()

    result = []
    for record in records:
        result.append({
            "id": record["id"],
            "name": record["name"],
            "result": record["result"],
            "is_dead_fish": record["recognition_success"]
        })

    return {"records": result}


def compress_image_bytes_to_size(image_bytes: bytes, target_kb: int = 200):
    """
    检查图片字节数据大小，如果超过指定大小，则进行JPEG压缩，并返回处理后的字节数据。

    Args:
        image_bytes (bytes): 输入图片数据的字节。
        target_kb (int): 目标文件大小，单位KB。默认为200KB。

    Returns:
        bytes: 处理后的图片字节数据 (原始数据或压缩后的数据)。
        str: 处理结果消息。
        bool: 如果处理成功且文件大小符合要求 (或原始大小已符合)，返回True；
              如果在达到最低质量后仍无法达到目标大小，返回False。
    """
    target_bytes = target_kb * 1024
    min_quality = 5  # 设置一个最低JPEG质量
    quality_step = 5 # 每次降低的质量步长

    initial_size = len(image_bytes)
    initial_size_kb = initial_size / 1024.0
    print(f"原始图片数据大小：{initial_size_kb:.2f} KB")

    # 1. 判断是否需要压缩
    if initial_size <= target_bytes:
        print(f"图片大小 ({initial_size_kb:.2f} KB) 已小于或等于目标大小 ({target_kb} KB)，无需压缩。")
        return image_bytes, "文件已小于目标大小，无需压缩。", True

    print(f"图片大小 ({initial_size_kb:.2f} KB) 超过目标大小 ({target_kb} KB)，开始尝试压缩...")

    # 2. 将字节数据解码为OpenCV图片格式
    try:
        # 使用 np.frombuffer 将字节转换为 numpy 数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        # 使用 cv2.imdecode 解码图片
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
             return image_bytes, "错误：无法解码图片数据。", False # 解码失败，返回原始数据或错误

    except Exception as e:
        return image_bytes, f"错误：图片解码失败 - {e}", False # 解码失败

    # 3. 逐步降低质量进行压缩 (编码为JPEG)
    current_quality = 95 # 从较高的质量开始
    best_buffer = None
    best_size = initial_size

    while current_quality >= min_quality:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), current_quality]
        success, buffer = cv2.imencode('.jpg', img, encode_param) # 尝试编码为JPEG

        if not success:
            print(f"警告：使用质量 {current_quality} 进行JPEG编码失败。")
            current_quality -= quality_step
            continue

        current_size = len(buffer)
        current_size_kb = current_size / 1024.0
        print(f"尝试质量 {current_quality}: 数据大小 {current_size_kb:.2f} KB")

        if current_size <= target_bytes:
            # 找到符合大小的质量
            best_buffer = buffer
            best_size = current_size
            print(f"成功：找到质量 {current_quality} 使数据大小 ({current_size_kb:.2f} KB) 小于或等于目标大小 ({target_kb} KB)。")
            break # 退出循环

        # 如果当前大小仍大于目标，降低质量
        current_quality -= quality_step

        # 如果质量降到最低仍未达到目标，保存当前质量下的图片数据
        if current_quality < min_quality:
             print(f"警告：已达到最低质量 {min_quality}，但数据大小 ({current_size_kb:.2f} KB) 仍大于目标大小 ({target_kb} KB)。将使用此质量的数据。")
             best_buffer = buffer
             best_size = current_size
             break


    # 4. 返回处理后的数据
    if best_buffer is not None:
        processed_bytes = bytes(best_buffer) # 将 numpy array buffer 转换为 bytes
        print(f"最终处理后的数据大小：{best_size / 1024.0:.2f} KB")
        if best_size <= target_bytes:
             return processed_bytes, f"文件成功压缩到 {best_size / 1024.0:.2f} KB。", True
        else:
             return processed_bytes, f"已达到最低质量，文件仍大于目标大小，最终大小 {best_size / 1024.0:.2f} KB。", False
    else:
        # 如果因为某种原因没有生成 buffer，返回原始数据或错误
        return image_bytes, "错误：未能生成有效的图片数据进行保存。", False


@api_router.post("/upload", summary="上传图片并保存,使用qwen-vl-72B模型识别图片是否是死鱼")
async def upload_image(file: UploadFile = File(...)):
    # 检查API调用次数是否已达上限
    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM fish_records WHERE call_success = 'Y'")
    result = cursor.fetchone()
    conn.close()
    
    current_count = result["count"] if result else 0
    
    if current_count >= max_count:
        raise HTTPException(
            status_code=403, 
            detail={"result": "次数用尽", "remain_count": 0}
        )
    
    remain_count = max_count - current_count


    # 检查文件是否为图片
    content_type = file.content_type
    if not content_type or not content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="只接受图片文件")

    # 读取文件内容
    file_content = await file.read()

    data = await process_image(file_content, file.filename)
    data["remain_count"] = max(remain_count, 0)
    return data


@api_router.get("/test_concurrent", summary="测试并发请求")
async def test_concurrent():
    import asyncio
    import os
    import time
    import shutil
    from pathlib import Path

    # 创建目标目录
    finish_dir = Path("data/finish")
    if not finish_dir.exists():
        finish_dir.mkdir(parents=True, exist_ok=True)

    async def send_request(file_path):
        try:
            # 使用普通方式读取文件
            with open(file_path, "rb") as f:
                file_content = f.read()

            # 创建文件对象
            file_name = os.path.basename(file_path)

            # 使用HTTP请求调用API
            files = {"file": (file_name, file_content, f"image/{file_name.split('.')[-1].lower()}")}
            if file_name.lower().endswith('.jpg'):
                files = {"file": (file_name, file_content, "image/jpeg")}
                
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:{port}{prefix}/fish/upload", 
                    files=files
                )
                result = response.json()

            # 如果请求成功，移动文件到finish目录
            if result.get("success", False):
                shutil.move(str(file_path), f"{finish_dir}/{result.get('is_dead_fish')}-{file_name}")

            logger.info(result)

            return {"file": file_name, "result": "success", "data": result}
        except Exception as e:
            logger.exception(e)
            return {"file": file_name, "result": "error", "error": str(e)}

    # 获取图片文件列表
    pic_dir = Path("data/pic")
    if not pic_dir.exists():
        return {"status": "error", "message": "data/pic 目录不存在"}

    # 获取所有图片文件
    image_files = []
    for ext in ["jpg", "jpeg", "png", "gif"]:
        image_files.extend(list(pic_dir.glob(f"*.{ext}")))

    if not image_files:
        return {"status": "error", "message": "data/pic 目录中没有图片文件"}

    # 批量处理，每批60个
    batch_size = 60
    total_results = []
    total_start_time = time.time()

    for batch_start in range(0, len(image_files), batch_size):
        batch_files = image_files[batch_start:batch_start + batch_size]

        # 创建任务列表
        tasks = [send_request(file_path) for file_path in batch_files]

        # 记录批次开始时间
        batch_start_time = time.time()

        # 并发执行所有任务
        batch_results = await asyncio.gather(*tasks)

        # 计算批次执行时间
        batch_execution_time = time.time() - batch_start_time

        # 如果执行时间少于60秒，等待剩余时间
        if batch_execution_time < 60 and batch_start + batch_size < len(image_files):
            wait_time = 60 - batch_execution_time
            print(f"批次执行完成，等待 {wait_time:.2f} 秒后开始下一批...")
            await asyncio.sleep(wait_time)

        total_results.extend(batch_results)

    total_time = time.time() - total_start_time

    return {
        "status": "completed",
        "total_files_processed": len(total_results),
        "total_time_taken": total_time,
        "average_requests_per_minute": len(total_results) / (total_time / 60),
        "results": total_results
    }

app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
