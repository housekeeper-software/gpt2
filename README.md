# gpt2
gpt2推理和训练，c++版本的朴素实现，注重计算方法和过程，没有考虑算法效率。  
矩阵乘法可选支持 OpenBLAS，可以比原始的数值计算方法快很多。   

# 依赖
PCRE2，一个正则表达式开源库，c++标准库中的正则表达式不能完全兼容gpt2的tokennizer表达式，所以引入这个开源库  
nlohmann::json：json库，用于解析config.json, vocab.json等  
OpenBLAS: https://github.com/OpenMathLib/OpenBLAS. 可选  

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
8.data_loadh/cc: 加载训练数据  
9.training.h/cc: 训练模型

# 功能实现
1.只实现单次推理  
2.完整的 tokenizer 编码和解码实现  
3.比较丰富的 sampling 实现，此部分实现来源于 llama.cpp  
4.支持从头开始训练

# 验证
gpt2的small,medium,large,xl都验证过。可以正常推理，加上合适的 sampling 配置，可以让模型有合理的输出。  

# 训练  
因为没有太多的加速优化，一般只能用小数据集训练，为了简单，我们对数据集预先处理，将其转换为 token 再保存起来。  
在训练时，直接加载这个 token 序列文件，分成批次进行训练。  

# 模型框架图
https://colab.research.google.com/drive/1MR-i2ZxoMuE-SUkhM8iYwYngXhGYlAEB?usp=sharing
![上图](https://github.com/housekeeper-software/gpt2/blob/main/Full_GPT_architecture.png)

