import time
from datetime import datetime


def wait_until(target_time):
    """阻塞程序，直到到达目标时间"""
    while True:
        current_time = datetime.now().strftime("%H:%M")
        if current_time >= target_time:
            print(
                f"当前时间 {current_time}，达到或超过目标时间 {target_time}，继续执行..."
            )
            break
        else:
            print(f"当前时间 {current_time}，未到目标时间 {target_time}，等待...")
            time.sleep(60)  # 每分钟检查一次
