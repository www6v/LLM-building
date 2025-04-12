import torch
import train_model
net_clone = train_model.create_net()
net_clone.load_state_dict(torch.load("./save_model/net_parameter.pkl"))


net_loaded = torch.load('./save_model/net_model.pkl')