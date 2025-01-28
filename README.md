# Hallucination-Suppression-Framework
本仓库是《**基于大模型微调的出院小结生成幻觉抑制方法**》的官方实现


# 依赖

要运行我们的代码，请安装相关软件包。
```
accelerate	  0.27.2
deepspeed	    0.14.2
fire	        0.5.0
flash-attn	  2.5.8
ninja	        1.11.1.1
sentencepiece	0.1.99
torch	        2.2.1
vllm	        0.4.1
peft	        0.10.0
trl	          0.8.1
datasets    	2.17.1	
transformers	4.40.0	
scipy	        1.12.0
tiktoken    	0.6.0	
protobuf	    3.20.3	
pydantic    	2.6.1	
matplotlib	  3.8.3	
sse-starlette	2.0.0	
packaging	    23.2	
pyyaml      	6.0.1
pandas	      1.5.3
numpy	        1.23.4
```

# 快速开始

如果您想与我们的构建过程保持一致，请进入 `/train/LLaMA-Factory/ours-script`目录，并按照目录中的说明进行操作。

## 预训练
```sh
# 前往相关目录
cd /train/LLaMA-Factory/ours-script/pretrain

# 获得数据集的缓存
bash 1_get_cache.sh

# 开始预训练
bash 2_start_pretrain.sh
```

## 指令微调
```sh
# 前往相关目录
cd /train/LLaMA-Factory/ours-script/sft

# 获得stage1到stage4的数据集缓存
bash 1_chatglm_cache_stage1.sh
bash 1_chatglm_cache_stage2.sh
bash 1_chatglm_cache_stage3.sh
bash 1_chatglm_cache_stage4.sh

# 开始迭代式指令微调
bash 2_chatglm_train_stage1_lora.sh
# 修改配置
bash /train/LLaMA-Factory/ours-script/export_lora_model.sh
bash 2_chatglm_train_stage2_lora.sh
# 修改配置
bash /train/LLaMA-Factory/ours-script/export_lora_model.sh
bash 2_chatglm_train_stage3_lora.sh
# 修改配置
bash /train/LLaMA-Factory/ours-script/export_lora_model.sh
bash 2_chatglm_train_stage4_lora.sh
# 修改配置
bash /train/LLaMA-Factory/ours-script/export_lora_model.sh
```


# 代码结构

`./train`:存放训练代码。

`./evaluate`:存放评测代码。

`dataset`:存放数据集样本。

# 致谢

特别感谢 [hiyouga](https://github.com/hiyouga/LLaMA-Factory) 提供模型的训练框架。

项目基于 [Chatglm3-6b](https://github.com/THUDM/ChatGLM3)。

感谢所有提供开源数据的创作者。
