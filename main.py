import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

from transformers import TrainingArguments, Trainer
from transformers import AutoModel, AutoTokenizer,BertTokenizer
from transformers import ViTFeatureExtractor, ViTModel
from pytorch_pretrained_vit import ViT
from PIL import Image

from models import *
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from tqdm import tqdm
import json


import argparse


def load_models(text_model_path="monologg/kobigbird-bert-base",vit_model_path="google/vit-base-patch16-224-in21k"):

    # tokenizer, feature extractor
    text_embedder=AutoModel.from_pretrained(text_model_path)
    tokenizer = BertTokenizer.from_pretrained(text_model_path)  # BertTokenizer
    
    vit = ViTModel.from_pretrained(vit_model_path)
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_path)
    
    return text_embedder,tokenizer,vit,feature_extractor

def load_dataset(tokenizer,feature_extractor,max_frame_len,data_dir='/home/ubuntu/blim/mining/news_dataset/'):

    # df
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    
    num_topic_labels=len(train_df["topic_label"].unique())

    # load dataset
    train_dataset = VidTextDataset(train_df['video_dir'], 
                                train_df['text'], 
                                train_df['topic_label'], 
                                max_frame_len = max_frame_len, 
                                feature_extractor = feature_extractor,
                                token_max_len = 1024,
                                tokenizer = tokenizer)

    val_dataset = VidTextDataset(val_df['video_dir'], 
                                val_df['text'], 
                                val_df['topic_label'], 
                                max_frame_len = max_frame_len, 
                                feature_extractor = feature_extractor,
                                token_max_len = 1024,
                                tokenizer = tokenizer)

    test_dataset = VidTextDataset(test_df['video_dir'], 
                                test_df['text'], 
                                test_df['topic_label'], 
                                max_frame_len = max_frame_len, 
                                feature_extractor = feature_extractor,
                                token_max_len = 1024,
                                tokenizer = tokenizer)
    
    return train_dataset,val_dataset,test_dataset,num_topic_labels


def load_dataloader(tokenizer,feature_extractor,max_frame_len,train_batch_size=1,val_batch_size=1,data_dir='/home/ubuntu/blim/mining/news_dataset/'):
    
    train_dataset,val_dataset,test_dataset,num_topic_labels=load_dataset(tokenizer,feature_extractor,max_frame_len=max_frame_len,data_dir=data_dir)
    

    
    train_dataloader=DataLoader(train_dataset,batch_size=train_batch_size)
    val_dataloader=DataLoader(val_dataset,batch_size=val_batch_size)
    test_dataloader=DataLoader(test_dataset,batch_size=val_batch_size)
    
    
    return train_dataloader,val_dataloader,test_dataloader,num_topic_labels
    
def train(args):
    
    text_embedder,tokenizer,vit,feature_extractor=load_models()
    train_dataloader,val_dataloader,_,num_topic_labels=load_dataloader(tokenizer=tokenizer,feature_extractor=feature_extractor,
                                                      max_frame_len=args.max_frame_len,
                                                      data_dir = os.path.join('/home/ubuntu/blim/mining/news_dataset/', args.data))
    
    # train_dataset,val_dataset,_=load_dataset(tokenizer,feature_extractor)
        
    num_topic_labels=num_topic_labels
    epochs=args.epoch_size
    lr=args.lr
    total_steps=0
    
    
    if args.model_type == "text":
        model = BigBirdForDocumentClassification(text_embedder,num_topic_labels)
    elif args.model_type=="video":
        vidt=VideoTransformer(vit)
        model=VideoTransformerClassificationModel(vidt,num_topic_labels)
    elif args.model_type =="videotext":
        vidt=VideoTransformer(vit)
        model=VidTextClassificationModel(text_embedder,vidt,num_topic_labels=num_topic_labels)

    model=model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_loss=9999999999
    
    model.train()
    
    
    for epoch in range(epochs):
        
        loss_total=0
        
        for idx,batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader),desc=f"Training epoch {epoch+1}"):
            optimizer.zero_grad()
            model.zero_grad()
            image_ipt,text_ipt,labels=batch
            # print(image_ipt.shape)
            text_ipt_dict = {}
            for key,val in text_ipt.items():
                text_ipt_dict[key] = val.squeeze(dim=1).to(args.device)
                
            image_ipt=image_ipt.squeeze(dim = 0).to(args.device)
            # text_ipt=text_ipt.to(args.device)
            labels=labels.squeeze(dim = 0).to(args.device)
            

            if args.model_type=="videotext":
                loss,logits,features=model(text_ipt_dict,image_ipt,labels)
                
            elif args.model_type == "text":
                loss,logits,last_hidden_states,pooler_outputs=model(text_ipt_dict,labels)
                
            elif args.model_type =="video":
                loss,logits,pooler_outputs=model(image_ipt,labels)
                
            loss_total+=loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_steps+=1
            
            
    
            
            if total_steps%args.eval_step==0:
                val_loss,predictions,_=evaluate(model,val_dataloader,args)
                if val_loss<best_loss:
                    best_loss=val_loss
                    torch.save({"model_state_dict":model.state_dict(),
                                "optimizer_state_dict":optimizer.state_dict(),
                               "scheduler_state_dict":scheduler.state_dict()},
                               f"{args.model_dir}/{args.model_type}/{args.model_type}_{args.data}_best_model.pth")
                    
    val_loss,predictions,_=evaluate(model,val_dataloader,args)
    if val_loss<best_loss:
        best_loss=val_loss
        torch.save({"model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "scheduler_state_dict":scheduler.state_dict()},
                    f"{args.model_dir}/{args.model_type}/{args.model_type}_{args.data}_best_model.pth")

                
            
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss_total/total_steps}")
        
        
def evaluate(model,val_dataloader,args):
    
    print("Evaluating")
    
    model.eval()
    loss_total=0
    
    groundtruth=[]
    predictions=[]
    
    avg_acc=0
    avg_f1=0
    
    for idx,batch in tqdm(enumerate(val_dataloader),total=len(val_dataloader),desc="Evaluating"):
        image_ipt,text_ipt,labels=batch
        image_ipt=image_ipt.squeeze(dim = 0).to(args.device)
        
        text_ipt_dict = {}
        for key,val in text_ipt.items():
            text_ipt_dict[key] = val.squeeze(dim=1).to(args.device)
        
        # text_ipt=text_ipt.to(args.device)
        labels=labels.squeeze(dim = 0).to(args.device)
        
        with torch.no_grad():
            
            if args.model_type=="videotext":
                loss,logits,features=model(text_ipt_dict,image_ipt,labels)
                
            elif args.model_type == "text":
                loss,logits,last_hidden_states,pooler_outputs=model(text_ipt_dict,labels)
                
            elif args.model_type =="video":
                loss,logits,pooler_outputs=model(image_ipt,labels)

            loss_total+=loss.item()
            # pred_total.append(logits.cpu().numpy())
            
            # if logits.dim==2:
            #     logits=logits.squeeze(1)
            
            
            if logits.dim()>=2:
                softmax=torch.nn.Softmax(dim=1)
                logits=softmax(logits)
                pred_idx=torch.argmax(logits,dim=1)
                
            else:
                softmax=torch.nn.Softmax(dim=0)
                logits=softmax(logits)
                pred_idx=torch.argmax(logits,dim=0)
            
            print("logits : ",logits)
            print("pred_idx : ",pred_idx)
            
            predictions.append(pred_idx.cpu().numpy())
            groundtruth.append(labels.cpu().numpy())
            
    
            
    print("gt : ",groundtruth)
    print("pred : ",predictions)
            
    accuracy = accuracy_score(y_true=groundtruth, y_pred=predictions)
    f1_macro = f1_score(y_true=groundtruth, y_pred=predictions,average="macro")
    avg_loss=loss_total/len(val_dataloader)
    
    print(f"avg_loss: {avg_loss} avg_acc: {accuracy} avg_f1: {f1_macro}")
    
    result_dict={"avg_loss":avg_loss,
                 "accuracy":accuracy,
                "f1_macro":f1_macro,
                }
    
    
    # print(f"avg_loss: {avg_loss}")
    
    return avg_loss,predictions,result_dict


def test(args):
    text_embedder,tokenizer,vit,feature_extractor=load_models()
    _,_,test_dataloader=load_dataloader(tokenizer,feature_extractor)
    
    num_topic_labels=7
    
    if args.model_type == "text":
        model = BigBirdForDocumentClassification(text_embedder,num_topic_labels)
    elif args.model_type=="video":
        vidt=VideoTransformer(vit)
        model=VideoTransformerClassificationModel(vidt,num_topic_labels)
    elif args.model_type =="videotext":
        vidt=VideoTransformer(vit)
        model=VidTextClassificationModel(text_embedder,vidt,num_topic_labels=num_topic_labels)

    ckpt=torch.load(f"{args.model_dir}/{args.model_type}/{args.model_type}_best_model.pth")
    model.load_state_dict(ckpt["model_state_dict"])
    model=model.to(args.device)
    
    _,_,result_dict=evaluate(model,test_dataloader,args)
    
    with open(f"{args.model_dir}/{args.model_type}/{args.model_type}_classification_result.json","w") as f:
        json.dump(result_dict,f)
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='argument')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--model_type',type=str, default="videotext")
    parser.add_argument("--epoch_size",type=int,default=10)
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--eval_step",type=int,default=100)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--model_dir",type=str,default="./model")
    # parser.add_argument("--data_dir",type=str,default="./data")
    # parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--test",action = "store_true")
    parser.add_argument('--data', type = str, default = 'original')
    parser.add_argument("--max_frame_len",type=int,default=30)
    parser.add_argument("--max_token_len",type=int,default=256)
    
    args=parser.parse_args()
    
    
    if not os.path.exists(f"{args.model_dir}/{args.model_type}/"):
        os.mkdir(f"{args.model_dir}/{args.model_type}/")
        
    if args.train:
        train(args)
    if args.test:
        test(args)