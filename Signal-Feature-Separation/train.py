import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fc import train_fc
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from dataset import *
from testdataset import *
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
model1 = torch.load('./TFA-Net/Separation/segmentation.pth')
model2 = train_fc()
optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model2.parameters())) 
loss_fn =nn.CrossEntropyLoss() 

def example_subset_accuracy(gt, predict):

    ex_equal = np.all(np.equal(gt.cpu().detach().numpy(), predict.cpu().detach().numpy()), axis=1).astype("float32")
    return np.sum(ex_equal)

def train(model1,model2,optimizer,loss_fn,train_loader,epochs=100,device = "cpu"):
    best_acc=0.0
    for epoch in range(epochs):
        tot_correct=0.0
        tot_all=0.0
        tot_loss=0.0
        model1.eval()
        model2.train()
        optimizer.zero_grad()
        for index,batch in enumerate(iter(train_loader)):
            noisysig, classlabel1, classlabel2 = batch
            noisysig = noisysig.cuda()
            classlabel1 = classlabel1.cuda()
            classlabel2 = classlabel2.cuda()
            model1 = model1.cuda()
            model2 = model2.cuda()
            classlabel = torch.cat((classlabel1, classlabel2),dim=0)
            classlabel = classlabel.squeeze(1)
            with torch.no_grad():
                img1, img2 = model1(noisysig) 
            img3 = torch.cat((img1, img2),dim=0)
            pre_class = model2(img3) 
            loss = loss_fn(pre_class,classlabel.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            predict_y = torch.argmax(pre_class.cpu(),1).data.numpy()
            tot_correct += sum(predict_y==classlabel.cpu().data.numpy())
            tot_all += len(predict_y)
            tot_loss += loss.item()
        
        avg_acc = tot_correct/tot_all
        loss_epoch = tot_loss/tot_all
        print('[Epoch %d]avg_loss:%.4f,avg_acc:%.4f%%'%(epoch+1,loss_epoch,avg_acc))

        if avg_acc > best_acc:   
            print("Modified model")
            best_acc = avg_acc
            torch.save(model2, "./TFA-Net/Separation/recognition.pth")
train(model1,model2,optimizer,loss_fn,traindatas,epochs=100)

