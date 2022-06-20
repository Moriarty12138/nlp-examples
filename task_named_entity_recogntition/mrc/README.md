# 脚本运行说明

## 脚本功能

    * squadv1_base_at.sh : SQuAD 1.1 任务上进行AT训练
    * squadv1_base_vat.sh : SQuAD 1.1 任务上进行VAT训练

## 脚本主要参数说明

    * BERT_DIR: BERT相关文件目录
    * SQUAD_DIR: 数据集目录
    * OUTPUT_DIR: 输出目录
    * model_name_or_path: 数据集缓存名称
    * model_type: 模型类别，bert或roberta
    * ckpt_frequency: 每个epoch的保存次数
    * schedule 和 s_opt1: 学习率调节器相关设定
    * epsilon: AT和VAT中扰动的强度，一般取值范围1e-4~1e-2
    * si: VAT中的初始扰动强度，一般取值范围1e-5~1e-4

请使用单卡运行；尚未在多卡环境下测试。