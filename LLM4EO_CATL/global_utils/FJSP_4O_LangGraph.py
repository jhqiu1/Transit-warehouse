from typing import Dict, Any
from langgraph.graph import StateGraph


# 1. 定义节点函数
def hello_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """第一个节点：添加 Hello 到状态"""
    state["message"] = "Hello"
    return state


def world_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """第二个节点：添加 World 到状态"""
    state["message"] += " World!"
    return state


# 2. 创建图
builder = StateGraph(Dict[str, Any])

# 3. 添加节点
builder.add_node("hello", hello_node)
builder.add_node("world", world_node)

# 4. 设置边（连接节点）
builder.add_edge("hello", "world")
builder.set_entry_point("hello")

# 5. 编译图
graph = builder.compile()

# 6. 执行图
result = graph.invoke({})
print(result)  # 输出: {'message': 'Hello World!'}
