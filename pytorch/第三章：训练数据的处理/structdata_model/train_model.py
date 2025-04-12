import torch 
from torch import nn
from torchkeras import summary
import datetime
from sklearn.metrics import accuracy_score
import pandas as pd 
import process_data
import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()


def create_net():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(15,20))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(20,15))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(15,1))
    net.add_module("sigmoid",nn.Sigmoid())
    return net
if __name__=="__main__":
    net = create_net()
    # print(net)
    # summary(net,input_shape=(15,))
    epochs = 10
    log_step_freq = 30

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.01)
    metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy()>0.5)
    metric_name = "accuracy"
    dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name]) 
    dl_train,dl_valid=process_data.get_data(batchsize=256)

    for epoch in range(1,epochs+1):  

        # 1，训练循环-------------------------------------------------
        net.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        #随机，无回放抽取
        for step, (features,labels) in enumerate(dl_train, 1):

            # 梯度清零
            optimizer.zero_grad()

            # 正向传播求损失
            predictions = net(features)

            loss = loss_func(predictions,labels)
            metric = metric_func(predictions,labels)

            # 反向传播求梯度
            loss.backward()
            optimizer.step()

            # 打印batch级别日志
            loss_sum += loss.item()
            metric_sum += metric.item()
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                    (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        net.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):

            predictions = net(features)
            val_loss = loss_func(predictions,labels)
            val_metric = metric_func(predictions,labels)

            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step, 
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
            "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
            %info)


    print('Finished Training...')
    plot_metric(dfhistory,"loss")
    #保存模型
    #保存方法，保存模型参数，不保存结构
    torch.save(net.state_dict(), "./save_model/net_parameter.pkl")

    #保存方法，保存模型参数，不保存结构
    torch.save(net.state_dict(), "./save_model/net_parameter.pkl")
    #保存整个模型
    torch.save(net, './save_model/net_model.pkl')
    # net_clone = create_net()
    # net_clone.load_state_dict(torch.load("./data/net_parameter.pkl"))
    # net_clone.forward(torch.tensor(x_test[0:10]).float()).data