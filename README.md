# Bilinear-CNN-for-fine-grained-recognition
细分类双线性池化模型（ICCV2015）  
1.Dataset :   
   I trained it on bird dataset   
2.Requirement   
   python3.5   
   torchvision0.4  
 3.train  
    1)first fine-tune fc layers:  
        lr=1.0,weight decay=1e-8,  
        I achieved 74.543%  
        first you need delete this all rows:this rows can load your pretrained parameters:  
         state_dict="/home/lyh2017/code/bili/bilinear_09_02/model_09_02/100.pkl"  
         pretrained_dict = torch.load(state_dict)  
         model_dict = net.state_dict()  
         pretrained_dict = {k[7: ]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}  
         model_dict.update(pretrained_dict)  
         net.load_state_dict(model_dict)  
    2)fune-tune all layers:  
        you need change net=BCNN(200,True)  
        add delete rows :  
            state_dict="/home/lyh2017/code/bili/bilinear_09_02/model_09_02/100.pkl"  
            pretrained_dict = torch.load(state_dict)  
            model_dict = net.state_dict()  
            pretrained_dict = {k[7: ]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}  
            model_dict.update(pretrained_dict)  
            net.load_state_dict(model_dict)  
        lr=0.01,weight decay=1e-5  
