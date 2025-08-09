import torch
import torch.nn as nn
import copy

from models.clip.prompt_learner import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner
from utils.class_names import core50_classnames, domainnet_classnames, cddb_classnames


class soyo_clip(nn.Module):
    def __init__(self, args):
        super(soyo_clip, self).__init__()
        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_model = clip_model

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.total_sessions = args['total_sessions']

        if args['dataset'] == 'cddb':
            dataset_classnames = cddb_classnames
            self.class_num = 2
        elif args['dataset'] == 'domainnet':
            dataset_classnames = domainnet_classnames
            self.class_num = 345
        elif args['dataset'] == 'core50':
            dataset_classnames = core50_classnames
            self.class_num = 50
        else:
            raise ValueError('Unknown datasets: {}.'.format(args['dataset']))
        
        ##### classifier ########################################
        self.classifier = nn.ModuleList([
            PromptLearner(self.cfg, list(dataset_classnames.values()), self.clip_model)
            for i in range(args['total_sessions'])
        ])

        ##### prompt ############################################################
        self.prompt_pool = nn.ModuleList([
            nn.Linear(args['image_dim'], args['prompt_length'], bias=False)
            for i in range(args['total_sessions'])])
        
        ##### offset ############################################################
        self.down_pool = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(args['image_dim'], args['hidden_dim'], dtype=self.dtype)
                for layer in range(12)
            ])
            for i in range(args['total_sessions'])
        ])
        self.up_pool = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(args['hidden_dim'], args['image_dim'], dtype=self.dtype)
                for layer in range(12)
            ])
            for i in range(args['total_sessions'])
        ])
        for layer in range(12):
            for i in range(args['total_sessions']):
                nn.init.xavier_uniform_(self.down_pool[i][layer].weight)
                nn.init.zeros_(self.down_pool[i][layer].bias)
                nn.init.xavier_uniform_(self.up_pool[i][layer].weight)
                nn.init.zeros_(self.up_pool[i][layer].bias)

        self.numtask = 0
        
    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def forward(self, image):
        logits = []
        image = image.type(self.dtype)
        
        ##### prompt ############################################################
        img_prompt = self.prompt_pool[self.numtask-1].weight
        
        ##### backbone ############################################################
        image_features = self.image_encoder(image,
                                            img_prompt = img_prompt, 
                                            up_pool = self.up_pool[self.numtask-1], 
                                            down_pool = self.down_pool[self.numtask-1],
                                            ca_mode = 'deep')
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        ##### text ############################################################
        prompts = self.classifier[self.numtask-1]
        tokenized_prompts = prompts.tokenized_prompts
        text_features = self.text_encoder(prompts(), tokenized_prompts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits.append(logit_scale * image_features @ text_features.t())

        return {
            'logits': torch.cat(logits, dim=1),
        }

    def interface(self, image, selection):
        logits = []
        image = image.type(self.dtype)
        
        ##### prompt ############################################################
        instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        
        ##### backbone ############################################################
        _feat_list = []
        for i in range(self.total_sessions):
            _feat = self.image_encoder(image, 
                                        img_prompt = instance_batch, 
                                        up_pool = self.up_pool[i], 
                                        down_pool = self.down_pool[i],
                                        ca_mode = 'deep')
            _feat_list.append(_feat)
        image_features = torch.stack(_feat_list, 0)[selection, torch.arange(_feat.shape[0])]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        ##### text ############################################################
        _text_features_list = []
        for i in range(self.total_sessions):
            prompt = self.classifier[i]
            tokenized_prompts = prompt.tokenized_prompts
            _text_features = self.text_encoder(prompt(), tokenized_prompts)
            _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
            _text_features_list.append(_text_features)
        text_features = torch.stack(_text_features_list, 0)[selection]
        
        logit_scale = self.logit_scale.exp()
        logits.append(logit_scale * torch.einsum('bd,bcd->bc', image_features, text_features))
    
        return {
            'logits': torch.cat(logits, dim=1),
        }

    def update_fc(self, nb_classes):
        self.numtask += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
