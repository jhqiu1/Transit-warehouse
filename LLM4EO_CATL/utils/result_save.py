import json
import os


def save_operators_results(json_file_path, all_data):
    # 检查文件是否存在，并读取现有数据或初始化
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:  # 处理文件存在但内容可能不是合法JSON的情况
            existing_data = []
    else:
        existing_data = []  # 文件不存在，初始化一个空列表

    # 确保现有数据是一个列表，以便我们可以向其中追加新记录。
    if not isinstance(existing_data, list):
        # 如果现有数据不是列表，为了追加新记录，我们将其转换为列表。
        existing_data = [existing_data]

    # 将新数据追加到现有数据列表中
    existing_data.append(all_data)

    # 将更新后的数据写回文件
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    # return all_data["operator"]
