import copy
import torch
from torch import nn
from backbone.linears import CosineLinear
from models.ast_models import ASTModel
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if name == "ast_audioset_10_10_0.4593" :
        if args["dataset"] == 'FMC':
            model = ASTModel(input_tdim=44, label_dim=527,  imagenet_pretrain  = True, audioset_pretrain=True)
        if 'nsynth' in args["dataset"]:
            model = ASTModel(input_tdim=63, label_dim=527, imagenet_pretrain  = True,audioset_pretrain=True)
        if 'librispeech' in args["dataset"]:
            model = ASTModel(input_tdim=201, label_dim=527, imagenet_pretrain  = True,audioset_pretrain=True)
        model.out_dim = 768
        return model.eval()



class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]
        self.args = args
        self.fc_for_train = None

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'


    @property
    def feature_dim(self):
        return self.backbone.out_dim


    def extract_vector(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        return self.backbone(x)

    def forward(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class MLPWeightedFeaturesFusion(nn.Module):
    def __init__(self, num_layers, feature_dim, hidden_dim):
        super(MLPWeightedFeaturesFusion, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.num_layers * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_layers),
            nn.Softmax(dim=-1)  # 使得权重参数之和为1
        )

    def forward(self, outputs):
        stacked_outputs = outputs 
        stacked_outputs = stacked_outputs.transpose(1, 0) 
        weights = self.mlp(stacked_outputs.reshape(-1, self.num_layers * stacked_outputs.size(2))) 

        weighted_output = weights.unsqueeze(-1) * stacked_outputs
        sum_weighted_output = torch.sum(weighted_output, dim=1)
        return sum_weighted_output, weighted_output


class FusionVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.set_fea_extractor()
        self.num_features = 768
        self.fusion_module= MLPWeightedFeaturesFusion(self.args['blocks'], 768, 64)


    def set_fea_extractor(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        self.fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        self.ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        self.ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.args['sigma'] == True:
            fc = CosineLinear(in_dim, out_dim,sigma=True)
        else:
            fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        with torch.no_grad():
            _,specific_features = self.backbone(x)
            features = torch.cat((specific_features[6:,:,:],specific_features[6:,:,:]),dim=0)
        return features

    def forward(self, x, old_layers_features = None):
        x = self.mel_feature(x)
        x = (x + 4.26) / (4.57 * 2)
        x, layer_features= self.backbone(x)
        specific_features = layer_features[6:,:,:]
        weighted_features,_ = self.fusion_module(specific_features)
        out = self.fc(weighted_features)
        out.update({"features": weighted_features})
        return out

    def mel_feature(self, x):
        if x.shape[1] == 44100:
            x = self.fs_spectrogram_extractor(x)
            x = self.fs_logmel_extractor(x)
        elif x.shape[1] == 64000:
            x = self.ns_spectrogram_extractor(x)
            x = self.ns_logmel_extractor(x)
        elif x.shape[1] == 32000:
            x = self.ls_spectrogram_extractor(x)
            x = self.ls_logmel_extractor(x)
        x = x.squeeze(1)
        return x


