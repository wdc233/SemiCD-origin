import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import set_trainable
from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
from utils.losses import CE_loss

class Consistency_ResNet50_CD(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, cons_w_unsup=None, testing=False,
            pretrained=True, use_weak_lables=False, weakly_loss_w=0.4):

        self.num_classes = num_classes
        if not testing:
            assert (sup_loss is not None) and (cons_w_unsup is not None)

        super(Consistency_ResNet50_CD, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        else:
            self.mode = 'semi'  # yes

        # Supervised and unsupervised losses
        if conf['un_loss'] == "KL":
        	self.unsuper_loss = softmax_kl_loss
        elif conf['un_loss'] == "MSE": # yes
            self.unsuper_loss = softmax_mse_loss
        elif conf['un_loss'] == "JS":
        	self.unsuper_loss = softmax_js_loss
        else:
        	raise ValueError(f"Invalid supervised loss {conf['un_loss']}")
        
        self.unsup_loss_w = cons_w_unsup
        self.sup_loss_w = conf['supervised_w']  # =1
        self.softmax_temp = conf['softmax_temp']  # =1
        self.sup_loss = sup_loss  # 交叉熵
        self.sup_type = conf['sup_loss']  #Ce-loss

        # Use weak labels
        self.use_weak_lables= use_weak_lables  # false
        self.weakly_loss_w  = weakly_loss_w  # =0.4
        # pair wise loss (sup mat)
        self.aux_constraint     = conf['aux_constraint']   # false
        self.aux_constraint_w   = conf['aux_constraint_w']  # =1
        # confidence masking (sup mat)
        self.confidence_th      = conf['confidence_th']   # =0.5
        self.confidence_masking = conf['confidence_masking']  # false

        # Create the model
        self.encoder = Encoder(pretrained=pretrained)

        # The main decoder
        upscale             = 8
        num_out_ch          = 2048
        decoder_in_ch       = num_out_ch // 4
        self.main_decoder   = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)
        #  "drop": 5, "drop_rate": 0.5, "spatial": true, "cutout": 5, "erase": 0.4, "vat": 2, "context_masking": 2,
        #         "object_masking": 2,
        #         "feature_drop": 5,
        #         "feature_noise": 5,
        #         "uniform_range": 0.3
        # The auxilary decoders
        if self.mode == 'semi' or self.mode == 'weakly_semi':
            vat_decoder     = [VATDecoder(upscale, decoder_in_ch, num_classes, xi=conf['xi'],  # 8, 512, 2, 1e-6, 2.0, 2
            							eps=conf['eps']) for _ in range(conf['vat'])]
            drop_decoder    = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
            							drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'])
            							for _ in range(conf['drop'])]
            cut_decoder     = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=conf['erase'])
            							for _ in range(conf['cutout'])]
            context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes)
            							for _ in range(conf['context_masking'])]
            object_masking  = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes)
            							for _ in range(conf['object_masking'])]
            feature_drop    = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes)
            							for _ in range(conf['feature_drop'])]
            feature_noise   = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
            							uniform_range=conf['uniform_range'])
            							for _ in range(conf['feature_noise'])]
            # 加不同扰动的辅助decoder
            self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
                                    *context_m_decoder, *object_masking, *feature_drop, *feature_noise])

    def forward(self, A_l=None, B_l=None, target_l=None, A_ul=None, B_ul=None, target_ul=None, curr_iter=None, epoch=None):
        if not self.training:
            return self.main_decoder(self.encoder(A_l, B_l))

        # We compute the losses in the forward pass to avoid problems encountered in muti-gpu 

        # Forward pass the labels example
        input_size  = (A_l.size(2), A_l.size(3))  # h,w
        output_l    = self.main_decoder(self.encoder(A_l, B_l))
        if output_l.shape != A_l.shape:
            output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)

        # Supervised loss
        if self.sup_type == 'CE':  #yes
            loss_sup = self.sup_loss(output_l, target_l, temperature=self.softmax_temp) * self.sup_loss_w   # *1
        elif self.sup_type == 'FL':
            loss_sup = self.sup_loss(output_l,target_l) * self.sup_loss_w
        else:
            loss_sup = self.sup_loss(output_l, target_l, curr_iter=curr_iter, epoch=epoch) * self.sup_loss_w

        # If supervised mode only, return
        if self.mode == 'supervised':
            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs

        # If semi supervised mode
        elif self.mode == 'semi':
            # Get main prediction
            x_ul = self.encoder(A_ul, B_ul)
            output_ul = self.main_decoder(x_ul)

            # Get auxiliary predictions
            outputs_ul = [aux_decoder(x_ul, output_ul.detach(), pertub=True) for aux_decoder in self.aux_decoders]
            # 使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变
            targets = F.softmax(output_ul.detach(), dim=1)  # 把maindecoder的预测结果作为target
            #  detech和原张量的数据相同，但requires\_grad=Falserequires_grad=False

            # Compute unsupervised loss
            loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, \
                            conf_mask=self.confidence_masking, threshold=self.confidence_th, use_softmax=False)
                            for u in outputs_ul])
            loss_unsup = (loss_unsup / len(outputs_ul))  # 在每个扰动decoder上的结果的平均mse损失
            curr_losses = {'loss_sup': loss_sup}  # 监督损失

            if output_ul.shape != A_ul.shape:
                output_ul = F.interpolate(output_ul, size=input_size, mode='bilinear', align_corners=True)
            outputs = {'sup_pred': output_l, 'unsup_pred': output_ul}

            # Compute the unsupervised loss
            weight_u = self.unsup_loss_w(epoch=epoch, curr_iter=curr_iter)  # 一致性损失权重
            loss_unsup  = loss_unsup * weight_u
            curr_losses['loss_unsup'] = loss_unsup
            total_loss  = loss_unsup  + loss_sup  # 半监督的总体损失

            # If case we're using weak lables, add the weak loss term with a weight (self.weakly_loss_w)
            outputs_ul_reshaped = []
            for temp in outputs_ul:
                temp_reshped = F.interpolate(temp, size=input_size, mode='bilinear', align_corners=True)
                outputs_ul_reshaped.append(temp_reshped)
            
            if self.use_weak_lables:
                weight_w = (weight_u / self.unsup_loss_w.final_w) * self.weakly_loss_w
                loss_weakly = sum([CE_loss(outp, target_ul) for outp in outputs_ul_reshaped]) / len(outputs_ul_reshaped)
                loss_weakly = loss_weakly * weight_w
                curr_losses['loss_weakly'] = loss_weakly
                total_loss += loss_weakly

            # Pair-wise loss
            if self.aux_constraint:
                pair_wise = pair_wise_loss(outputs_ul) * self.aux_constraint_w
                curr_losses['pair_wise'] = pair_wise
                loss_unsup += pair_wise

            return total_loss, curr_losses, outputs

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'semi':
            return chain(self.encoder.get_module_params(), self.main_decoder.parameters(), 
                        self.aux_decoders.parameters())

        return chain(self.encoder.get_module_params(), self.main_decoder.parameters())

