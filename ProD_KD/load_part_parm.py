import torch
import model
from model.models import BiEncoderNllLoss, BiBertEncoder, ColBERT, ColBERTNllLoss
# model = BiBertEncoder(args)
# pretrain_file = "/data/liuguandu/SimXNS/result/DE_6layer/checkpoint-9000"
# ctx_layer = q_layer = 2
# # cut_layer = cut_layer - 1
# pretrain_state = torch.load(pretrain_file)
# # print(pretrain_state["model_dict"]["question_model.encoder.layer.3.attention.output.dense.weight"])
# new_dict = {}
# str1 = ["question_model.encoder.layer.{}.attention.self.query.weight", "question_model.encoder.layer.{}.attention.self.query.bias", \
#         "question_model.encoder.layer.{}.attention.self.key.weight", "question_model.encoder.layer.{}.attention.self.key.bias", \
#         "question_model.encoder.layer.{}.attention.self.value.weight", "question_model.encoder.layer.{}.attention.self.value.bias", \
#         "question_model.encoder.layer.{}.attention.output.dense.weight", "question_model.encoder.layer.{}.attention.output.dense.bias", \
#         "question_model.encoder.layer.{}.attention.output.LayerNorm.weight", "question_model.encoder.layer.{}.attention.output.LayerNorm.bias", \
#         "question_model.encoder.layer.{}.intermediate.dense.weight", "question_model.encoder.layer.{}.intermediate.dense.bias", \
#         "question_model.encoder.layer.{}.output.dense.weight", "question_model.encoder.layer.{}.output.dense.bias", \
#         "question_model.encoder.layer.{}.output.LayerNorm.weight", "question_model.encoder.layer.{}.output.LayerNorm.bias"]
# str2 = ["ctx_model.encoder.layer.{}.attention.self.query.weight", "ctx_model.encoder.layer.{}.attention.self.query.bias", \
#         "ctx_model.encoder.layer.{}.attention.self.key.weight", "ctx_model.encoder.layer.{}.attention.self.key.bias", \
#         "ctx_model.encoder.layer.{}.attention.self.value.weight", "ctx_model.encoder.layer.{}.attention.self.value.bias", \
#         "ctx_model.encoder.layer.{}.attention.output.dense.weight", "ctx_model.encoder.layer.{}.attention.output.dense.bias", \
#         "ctx_model.encoder.layer.{}.attention.output.LayerNorm.weight", "ctx_model.encoder.layer.{}.attention.output.LayerNorm.bias", \
#         "ctx_model.encoder.layer.{}.intermediate.dense.weight", "ctx_model.encoder.layer.{}.intermediate.dense.bias", \
#         "ctx_model.encoder.layer.{}.output.dense.weight", "ctx_model.encoder.layer.{}.output.dense.bias", \
#         "ctx_model.encoder.layer.{}.output.LayerNorm.weight", "ctx_model.encoder.layer.{}.output.LayerNorm.bias"]

# for k, v in pretrain_state["model_dict"].items():
#     if "question_model.encoder.layer."+str(q_layer-1) in k or "ctx_model.encoder.layer."+str(ctx_layer-1) in k:
#         continue
#     if "question_model.encoder.layer."+str(q_layer) in k or "ctx_model.encoder.layer."+str(ctx_layer) in k:
#         if "question_model" in k:
#             for i in str1:
#                 new_dict[i.format(str(q_layer-1))] = pretrain_state["model_dict"][i.format(str(q_layer))]
#             q_layer += 1
#         elif "ctx_model" in k:
#             for i in str2:
#                 new_dict[i.format(str(ctx_layer-1))] = pretrain_state["model_dict"][i.format(str(ctx_layer))]
#             ctx_layer += 1
        
#         continue
#     new_dict[k] = v

# for k, v in new_dict.items():
#     print(k)



pretrain_file = "/data/liuguandu/SimXNS/result/DE_12layer/checkpoint-40000"
pretrain_state = torch.load(pretrain_file)
for k, v in pretrain_state["model_dict"].items():
    print(k)
