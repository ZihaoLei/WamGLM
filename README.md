# WamGLM: A multimodal large-scale language model for wafer map defect information in-depth query through multi-turn dialogue based on prototypical supervised contrastive learning
> **Authors:** Shulong Gu, Zihao Lei, Guangrui Wen, Quanning Xu, Zhaojun Steven Li, Xuefeng Chen, Chunsheng Yang

> **Abstract:** To ensure production efficiency and process stability in semiconductor manufacturing, it is of critical importance to detect wafer map defects and perform information query for tracing and solving problems during the manufacturing process. Numerous vision models based on deep learning have been successfully applied to wafer map defect recognition (WMDR), yielding remarkable results. However, the dynamic and in-depth querying of wafer map defect information remains relatively underexplored. Leveraging the rapid advancements in multimodal large language models (MLLMs), this paper proposes a novel approach for wafer map defect information query (WMDIQ). First, following the paradigm of employing cross-modal alignment model to bridge vision and language models, an end-to-end response MLLM: general language model for wafer map (WamGLM), is constructed for WMDIQ. Concurrently, by designing an interactive dialogue framework between large language models (LLMs), the first large-scale multi-turn dialogue dataset: visual multi-turn question answering dataset for wafer map defects (WaferMapVMQA Dataset), is constructed for wafer map defect analysis. Subsequently, WamGLM is trained using a two-stage fine-tuning strategy. In the first stage, a visual fine-tuning method based on prototypical supervised contrastive learning (PSCL) is introduced to enhance the intra-class compactness and inter-class separability of defect features. In the second stage, language fine-tuning is conducted using the WaferMapVMQA Dataset to infuse specialized knowledge into WamGLM. To validate the effectiveness and superiority of the proposed method, experiments are conducted on a real wafer map dataset. The results demonstrate that the proposed method significantly outperforms other methods in both defect recognition performance and information query response performance.
<p align="center">
  <img src="./imgs/Network architecture.emf" alt="framework" width="700">
</p>

## Introduction
This repository provides the official PyTorch implementation of our paper:<br>
**"WamGLM: A multimodal large-scale language model for wafer map defect information in-depth query through multi-turn dialogue based on prototypical supervised contrastive learning"**.

> Our proposed WamGLM is based on [VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B) and employs LoRA fine-tuning.

## Dataset
The dataset we used was the WaferMapVMQA dataset that we constructed ourselves. For details, see the folder finetune_data/m.json.
