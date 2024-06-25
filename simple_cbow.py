import torch
import torch.nn as nn
import pickle 
from collections import Counter
import torch.optim as optim
import tool
import time

###################### Tegmier Standard Gpu Checking Processing ######################
# work_place lab:0 home:1 laptop:2
work_place = 1
gpu_setup_ascii_art_start = '''
Running Tegmier GPU Setup  
'''
print(gpu_setup_ascii_art_start)
if work_place == 0:
    torch.cuda.set_device(0)
elif work_place == 1:
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    torch.cuda.set_device(0)
else:
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.cuda.current_device()
if torch.cuda.is_available() and device != 'cpu':
    print(f"当前设备: CUDA")
    print(f"设备名称: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(device)}")
    print(f"总内存: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
else:
    print("当前设备: CPU")

gpu_setup_ascii_art_end = '''
GPU Setup Finished
'''
print(gpu_setup_ascii_art_end)
###################### Tegmier Standard Gpu Checking Processing END ######################

# data_loading
with open(r'data/corpus.pkl', 'rb') as f:
    data = pickle.load(f)

with open(r'data/voc.pkl', 'rb') as f:
    voc = pickle.load(f)

word_to_index = {word: idx for idx, word in enumerate(voc)}

word_to_one_hot = []
for word, index in word_to_index.items():
    word_to_one_hot.append(tool.word_to_one_hot(word, word_to_index))

train_data, valid_data = tool.shuffle_and_dataset_split(word_to_one_hot)

batch_size = 5
voc_size = len(word_to_index)
one_hot_length = len(word_to_one_hot[0])
embedding_length = 300
nepochs = 25

class Model(nn.Module):
    def __init__(self, batch_size, voc_size, one_hot_length, embedding_length) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.voc_size = voc_size
        self.one_hot_length = one_hot_length
        self.embedding_length = embedding_length
        self.hiddenlayer = nn.Linear(one_hot_length, embedding_length, dtype= torch.float64)
        self.outputlayer = nn.Linear(embedding_length, one_hot_length, dtype= torch.float64)
        
    
    def forward(self, x):
        hidden_result = torch.zeros(batch_size, self.embedding_length, device='cuda:0', dtype= torch.float64)
        output_result = torch.zeros(batch_size, self.one_hot_length, device='cuda:0', dtype= torch.float64 )
        hidden_result = self.hiddenlayer(x)
        output_result = self.outputlayer(hidden_result)
        return output_result
    
def train_model(model, criterion, optimizer, nepochs, batch_size, training_data):
    model.train()
    for epoch in range(nepochs):
        train_data_set = tool.data_loader(training_data, batch_size)
        train_loss = []
        t_start = time.time()
        for train_data in train_data_set:
            x = train_data.cuda()
            y = train_data.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            train_loss.append([float(loss)])
            optimizer.step()
        # print('train loss: {:.8f} {}'.format(train_loss, time.time() - t_start))
    print("training finished!")
    return model

def eval_model(model, valid_set, batch_size):
    model.eval()
    valid_data_set = tool.data_loader(valid_set, batch_size)
    for valid_data in valid_data_set:
        x = valid_data.cuda()
        y = valid_data.cuda()
        y_pred = model(x)
        y_pred = nn.functional.softmax(y_pred, dim = 1)
        y_pred = torch.argmax(y_pred, dim = -1)
        output_tensor = torch.zeros_like(y)
        output_tensor.scatter_(1, y_pred.unsqueeze(1), 1.0)
        print(y.shape[0])
        for i in range(y.shape[0]):
            if torch.equal(y[i], y_pred[i]):
                print(True)
            else:
                print(False)
            # print(True if y[i] == y_pred[i] else False)




model = Model(batch_size=batch_size, voc_size=voc_size, one_hot_length=one_hot_length, embedding_length=embedding_length).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
model = train_model(model=model, criterion=criterion, optimizer=optimizer, nepochs=nepochs, batch_size=batch_size, training_data=train_data)
eval_model(model=model, valid_set=valid_data, batch_size=batch_size)

