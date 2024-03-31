from TTS.api import TTS
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)
tts.tts_to_file("这篇文章主要讨论了一个名为AutoDev的工具，它是一个人工智能驱动的软件开发助手，用于自动化测试生成、代码检索、构建与执行等软件工程任务。文章介绍了AutoDev在这些任务中的表现，如效率和测试成功率，并提到了它的人机交互设计，允许用户在必要时提供反馈。未来计划包括将AutoDev更深入地集成到开发环境（如IDE）和持续集成/持续部署（CI/CD）流程中，以提升软件开发的效率和协作性。同时，文章还提到了AI在软件工程中的应用和AutoDev在相关研究领域中的位置。", file_path="audio/test.wav")
