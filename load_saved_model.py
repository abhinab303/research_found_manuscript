import os
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
from torch.optim import lr_scheduler

from utils import load_data_train_names, load_data_test_names, evaluteTop5_names, evaluteTop1_names
from model.servenetlt import ServeNet

epochs = 100
SEED = 123
LEARNING_RATE = 0.004
WEIGHT_DECAY = 0.01
EPSILON = 1e-8
BATCH_SIZE = 512
CLASS_NUM = 50
cat_num = "50"

des_max_length = 110  # 110#160#200
name_max_length = 10

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

train_data = load_data_train_names(CLASS_NUM)
test_data = load_data_test_names(CLASS_NUM)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

# model = ServeNet(768, CLASS_NUM)
# model.bert_description.requires_grad_(False)
# model.bert_name.requires_grad_(False)
# model = torch.nn.DataParallel(model)
# model = model.cuda()
# model.train()

# pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# pytorch_total_params_all = sum(p.numel() for p in model.parameters())
# print("Trainable: ", pytorch_total_params_trainable)
# print("All: ", pytorch_total_params_all)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
# # optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

model = ServeNet(768, CLASS_NUM)
model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load("/home/aa7514/PycharmProjects/research_found_manuscript/saved_model"))
model.eval()

print("=======>top1 acc on the test:{}".format(str(evaluteTop1_names(model, test_dataloader, CLASS_NUM, True, p=1))))
pass