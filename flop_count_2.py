import torch
import torchaudio
from torchaudio.utils import download_asset
from ptflops import get_model_complexity_info
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
    FairseqDropout
)

def get_lnorm_flops(module, input, output):
    # breakpoint()
    # print("!!!!!!!!!!!!! module: ", module)
    input = input[0]
    B, N, C = input.size()[0], input.size()[1], input.size()[2]
    flops = 0
    # in_features
    in_features, out_features = C, output.size()[2]
    assert in_features == module.normalized_shape[0], f"input feature dim must equal to {module} dim."
    flops += B * N * in_features * out_features
    module.__flops__ += flops

# dir(model.w2v_model.feature_extractor.conv_layers[1][2][1])

# macs, params = get_model_complexity_info(model, (sample["net_input"],), as_strings=True,input_constructor=get_first_item,print_per_layer_stat=False, verbose=True, custom_modules_hooks={Fp32LayerNorm: get_lnorm_flops, LayerNorm: get_lnorm_flops})

def get_msa_flops(module, input, extra):
    # TODO: add dropout for training
    # breakpoint()
    extra = extra[0]
    B, N, C = extra.size()[1], extra.size()[0], extra.size()[2]
    flops = 0
    # x -> q, k, v
    flops += N * module.embed_dim * 3 * module.embed_dim
    # q @ k
    flops += B * module.num_heads * N * (module.embed_dim // module.num_heads) * N
    # attn @ v
    flops += B * module.num_heads * N * N * (module.embed_dim // module.num_heads)
    # proj(x)
    flops += B * N * module.embed_dim * module.embed_dim
    module.__flops__ += flops

SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = bundle.get_model().to(device)
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

def get_first_item(tuple_):
    return tuple_[0]
macs, params = get_model_complexity_info(model, (waveform,), as_strings=True, input_constructor=get_first_item,
                                                 print_per_layer_stat=True, verbose=True,
                                                 custom_modules_hooks={
                                                Fp32LayerNorm: get_lnorm_flops,
                                                LayerNorm: get_lnorm_flops,
                                                MultiheadAttention: get_msa_flops,
            })


print('MAC COUNT {}, and PARAMETERS {}'.format(macs, params))
print(model)


