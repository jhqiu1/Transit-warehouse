# 将代码分割成行，逐行检查
lines = prompt.split('\n')
for i, line in enumerate(lines, 1):
    print(f"{i}: {repr(line)}")


