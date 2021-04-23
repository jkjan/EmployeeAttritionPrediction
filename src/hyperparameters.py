import data_retriever as dr
import torch
from LogisticRegression import LogisticRegression
import sys


if torch.cuda.is_available():
    sys.stdout.write("CUDA %s is available.\n\n" % torch.version.cuda)
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
    cur_device = torch.cuda.current_device()
    print('Device:      ')
    print("    Index:   ", cur_device)
    print("    Name:    ", torch.cuda.get_device_name(cur_device))
    print('METADATA:    ')
    print('    CUDA ver: ', torch.version.cuda)
    print('    Torch ver:', torch.__version__)
    print('    cuDNN ver:', torch.backends.cudnn.version())
    print('Memory Usage:')
    print('    Max Alloc:', round(torch.cuda.max_memory_allocated(cur_device) / 1024 ** 3, 1), 'GB')
    print('    Allocated:', round(torch.cuda.memory_allocated(cur_device) / 1024 ** 3, 1), 'GB')
    print('    Cached:   ', round(torch.cuda.memory_reserved(cur_device) / 1024 ** 3, 1), 'GB')
    print()
else:
    device = 'cpu'

data_path = "../data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
original_data = dr.get_original_data(data_path)
features_not_to_use = ["Over18"]
data, label = dr.get_data(original_data, features_not_to_use)
data = dr.split_data(data, label)
batch_size = 256
input_size = len(data["train"]["input"][0])
output_size = 2
n_iter = 10000
print_freq = n_iter // 10
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.005
model = LogisticRegression(input_size, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, n_iter // 3)