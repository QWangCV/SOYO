import torch
import torch.nn as nn
import copy

from models.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.clip.prompt_learner import cfgc, load_clip_to_cpu


class ViT_Prompts(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)


    def forward(self, x, img_prompt=None, up_pool=None, down_pool=None, align_bias=None, get_attn_mask=False, attn_mask=None, **kwargs):
        if get_attn_mask:
            assert img_prompt is None
        
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if img_prompt is not None:
            img_prompt = img_prompt.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        x = x + self.pos_embed.to(x.dtype)
        if img_prompt is not None:
            x = torch.cat([x[:,:1,:], img_prompt, x[:,1:,:]], dim=1)

        x = self.pos_drop(x)
        
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, attn_mask=attn_mask) # x: [bs, 197/207, 768]
            if up_pool is not None:
                assert down_pool is not None
                assert isinstance(up_pool[0], nn.Linear)
                assert isinstance(down_pool[0], nn.Linear)
                x = x + up_pool[i](down_pool[i](x))
        
        x = self.norm(x[:, 0])
        
        return x



def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_Prompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class soyo_vit(nn.Module):
    def __init__(self, args):
        super(soyo_vit, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)
        
        self.total_sessions = args['total_sessions']
        
        if args['dataset'] == 'cddb':
            self.class_num = 2
        elif args['dataset'] == 'domainnet':
            self.class_num = 345
        elif args['dataset'] == 'core50':
            self.class_num = 50
        else:
            raise ValueError('Unknown datasets: {}.'.format(args['dataset']))
        
        ##### classifier ########################################
        self.classifier = nn.ModuleList([
            nn.Linear(args['image_dim'], self.class_num, bias=True)
            for i in range(args['total_sessions'])
        ])
        
        ##### prompt ############################################
        self.prompt_pool = nn.ModuleList([
            nn.Linear(args['image_dim'], args['prompt_length'], bias=False)
            for i in range(args['total_sessions'])
        ])
        
        ##### offset ############################################
        self.down_pool = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(args['image_dim'], args['hidden_dim'])
                for layer in range(12)
            ])
            for i in range(args['total_sessions'])
        ])
        self.up_pool = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(args['hidden_dim'], args['image_dim'])
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
        return self.image_encoder.out_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image):
        logits = []
        ##### prompt ############################################################
        img_prompt = self.prompt_pool[self.numtask-1].weight
        
        ##### backbone ############################################################
        image_features = self.image_encoder(image, 
                                            img_prompt = img_prompt, 
                                            up_pool = self.up_pool[self.numtask-1], 
                                            down_pool = self.down_pool[self.numtask-1])
        
        ##### classifier ############################################################
        logits.append(self.classifier[self.numtask-1](image_features))

        return {
            'logits': torch.cat(logits, dim=1),
        }

    def interface(self, image, selection):
        ##### prompt ############################################################
        instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        
        ##### backbone ############################################################
        _feat_list = []
        for i in range(self.total_sessions):
            _feat = self.image_encoder(image, img_prompt = instance_batch, 
                                        up_pool = self.up_pool[i], 
                                        down_pool = self.down_pool[i])
            _feat_list.append(_feat)
        image_features = torch.stack(_feat_list, 0)[selection, torch.arange(_feat.shape[0])]
        
        ##### classifier ############################################################
        _logits_list = []
        for i in range(self.total_sessions):
            logits = self.classifier[i](image_features)
            _logits_list.append(logits)
        logits = torch.stack(_logits_list, 0)[selection, torch.arange(_logits_list[0].shape[0])]
        
        return {
            'logits': logits,
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
