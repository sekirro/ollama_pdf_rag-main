## 项目简介

用本地ollama实现简单rag框架

## 运行方式

1. 进行本地ollama配置，[ollama配置方法](https://blog.csdn.net/2401_85375298/article/details/144883000)

2. 拉取两个模型`deepseek-r1`和`nomic-embed-text`
3. 运行`main.py`

## 文件结构

- `documents_for_analyse`是用来分析的文件夹，可以把待分析文件放入
- `notebooks`是jupyter版本（直接扒另一个github上的，找不到源地址了，项目也是模仿着做的）
- `utils`是工具文件，`main.py`调用其中的函数
- `main.py`是主要代码