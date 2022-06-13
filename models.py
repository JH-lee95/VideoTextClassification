import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from PIL import Image
from glob import glob
from torch.utils.data import Dataset

class VidTextDataset(Dataset):
    def __init__(self,
                 video_dir, 
                 text,
                 labels, 
                 max_frame_len,                  
                 feature_extractor,
                 token_max_len,
                 tokenizer):
        self.labels = labels
        self.text = text
        self.video_dir = video_dir
        self.max_frame_len = max_frame_len
        self.feature_extractor = feature_extractor
        self.token_max_len = token_max_len
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        
        label = self.labels[idx]
        # make image sequence
        video = self.video_dir[idx]
        img_dir = os.path.join(video.replace('videos', 'images').replace('.mp4', ''), '*')
        image_list = glob(img_dir)
        img_list = [Image.open(file_name) for file_name in image_list][:self.max_frame_len]
        img_inputs = self.feature_extractor(img_list, return_tensors = "pt")['pixel_values']
        # make text sequence
        text = self.text[idx]
        text_input = self.tokenizer(text,max_length=self.token_max_len,padding="max_length",return_tensors="pt",truncation=True)
        
        return img_inputs, text_input, label

class VideoTransformer(nn.Module):
    def __init__(self,vit,input_dim=768,freeze_vit=True):
        super().__init__()
        
        self.vit=vit # vit model (image embedding model)
        
        if freeze_vit:
            self.freeze_vit()
        
        self.transformer_encoder=nn.Transformer(d_model=input_dim).encoder
        self.output_dim=input_dim
    
    
    def forward(self,sequence_of_images):
        
        with torch.no_grad():
            image_embs=self.vit(sequence_of_images)[1]   # (last_hidden_states,pooler_outputs)
        
        # new_image_embs=[]
        # # batch_mask=[]
        
        # for item in batch_image_embs:
            
        #     sequence_length=item.size()[0]
            
        #     if sequence_length>=self.max_frame_len:
        #         item=item[:self.max_frame_len,:]
        #         new_image_embs.append(item)
                
        #     else:
        #         new_image_embs.append(item)
                
        # #     elif sequence_length<self.max_frame_len:
                
        # #         padding_len=self.max_frame_len-sequence_length
        # #         pad_tensor=torch.full((padding_len,768),-float("Inf"))
                
        # #         mask=torch.zeros((self.max_frame_len))
        # #         mask[sequence_length:]=1
        # #         mask=mask.bool()
                
        # #         batch_mask.append(mask)
        # #         item=torch.cat([item,pad_tensor],dim=0)
        # #         new_image_embs.append(item)
    
        # new_image_embs=torch.stack(new_image_embs)
        # # batch_mask=torch.stack(batch_mask)
        
        # # print(batch_mask)
        
        # # print("batch mask shape : ",batch_mask.shape)
              
        video_embs=self.transformer_encoder(src=image_embs,
                                            #src_key_padding_mask=batch_mask.transpose(0,1),
                                            )
        
        return video_embs,video_embs.mean(dim=0) # avg pooling
        
    def freeze_vit(self):  # freeze vit
        for param in self.vit.parameters():
            param.requires_grad=False
            
class BigBirdForDocumentClassification(nn.Module):
    def __init__(self,embedder,num_topic_labels=None,dropout_p=0.2):
        super().__init__()
        self.embedder=embedder
        #self.tokenizer=tokenizer
        self.num_topic_labels=num_topic_labels
        self.hidden_size=embedder.config.hidden_size
        
        self.classifier=nn.Linear(self.hidden_size,num_topic_labels)
        self.dropout=nn.Dropout(p=dropout_p)
        
        
    def forward(self,inputs,labels=None):
        
        last_hidden_states,pooler_outputs=self.embedder(**inputs)[:]
        pooler_outputs=self.dropout(pooler_outputs)
        logits=self.classifier(pooler_outputs)
        
        if labels is not None:
            loss_fc=nn.CrossEntropyLoss()
            loss=loss_fc(logits,labels.view(-1))
            
            return loss,logits,last_hidden_states,pooler_outputs
        else:
            return logits,last_hidden_states,pooler_outputs
        
        
        
class VideoTransformerClassificationModel(nn.Module):
    def __init__(self,vidt,num_topic_labels,dropout_p=0.2):
        super().__init__()
        self.vidt=vidt
        self.input_dim=self.vidt.output_dim
        
        self.classifier=nn.Linear(self.input_dim,num_topic_labels)
        self.dropout=nn.Dropout(p=dropout_p)
        
        
    def forward(self,sequecne_of_images,labels=None):
        
        #video embeddings
        video_embs,avg_pooled=self.vidt(sequecne_of_images)
        avg_pooled=self.dropout(avg_pooled)
    
        logits=self.classifier(avg_pooled)
        if labels is not None:
            loss_fc=nn.CrossEntropyLoss()
            loss=loss_fc(logits,labels)
            
            return loss,logits,avg_pooled
        
        else:
            return logits,avg_pooled
                      
class VidTextClassificationModel(nn.Module):
    def __init__(self,textembedder,vidt,num_topic_labels=None,dropout_p=0.2,):
        super().__init__()
        
        self.textembedder=textembedder #text embedding model ex)big bird
        self.vidt=vidt  #videotransformer
        self.text_emb_size=self.textembedder.config.hidden_size
        self.video_emb_size=self.vidt.output_dim
        
        
        self.classifier=nn.Linear(self.text_emb_size+self.video_emb_size,num_topic_labels)
        self.dropout=nn.Dropout(p=dropout_p)
        
        
    def forward(self,sequence_of_texts,sequecne_of_images,labels=None):
        
        # text embeddings
        last_hidden_states,pooler_outputs=self.textembedder(**sequence_of_texts)[:]
        pooler_outputs=self.dropout(pooler_outputs).squeeze(0)
        
        #video embeddings
        _,video_embs=self.vidt(sequecne_of_images)
        
        features=torch.cat([pooler_outputs,video_embs])
        features=self.dropout(features)
        
        logits=self.classifier(features)
        if labels is not None:
            loss_fc=nn.CrossEntropyLoss()
            loss=loss_fc(logits,labels)
            
            return loss,logits,features
        
        else:
            return logits,features
        
        
    
        
        
    
            
        