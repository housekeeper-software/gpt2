# gpt2
gpt2的c++最慢最详细的实现，没有使用任何优化，只使用原始的计算方法，以展示transformer推理的详细过程
# 依赖
PCRE2，一个正则表达式开源库，c++标准库中的正则表达式不能完全兼容gpt2的tokennizer表达式，所以引入这个开源库  
nlohmann::json：json库，用于解析config.json, vocab.json等  
# 模型
可以从 https://huggingface.co/openai-community/gpt2下载到本地，只支持safetensors模型格式。  
除此之外，还需要vocab.json,merges.txt,config.json  
config.json包含模型基础参数，嵌入维度，层和多头的一些重要参数  
vocab.json：词表文件  
merges.txt：合并规则  
# 源码介绍
1.tiktokenizer.h/cc:加载词库和合并规则，并且实现encode和decode  
2.sampling.h/cc:实现多种采样方法，代码来自于llama.cpp  
3.cache.h/cc:实现attention kv cache，加速推理过程  
4.layers.h/cc:实现gpt模型加载和推理，以及各层的具体实现  
5.safe_tensors.h/cc:实现加载 huggingface safetensors模型参数  
6.storage.h/cc:内存管理  
7.tensor.h/cc:一个极其简单的张量对象，保存张量的形状和数据类型  

# 功能实现
1.只实现单次推理  
2.完整的tokenizer 编码和解码实现  
3.比较丰富的sampling实现  

# 验证
gpt2的small,medium,large,xl都验证过。可以正常推理，加上合适的sampling，可以让模型有合理的输出。

# 模型框架图
https://colab.research.google.com/drive/1MR-i2ZxoMuE-SUkhM8iYwYngXhGYlAEB?usp=sharing

