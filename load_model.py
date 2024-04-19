from utils import trunc_normal_
import torch
import vision_transformer as vit
import diffusion_transformer as dit


def load_models():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    encoder = vit.VisionTransformer(
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
    )
    print('Loaded encoder')
    
    # count number of parameters
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'Number of encoder parameters: {num_params}')
    
    # predictor = ''
    predictor = vit.VisionTransformerPredictor(
        input_dim=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=12.0,
        qkv_bias=True,
        drop_rate=0.05,
        attn_drop_rate=0.05,
    )
    print('Loaded predictor')

    # count number of parameters
    num_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f'Number of predictor parameters: {num_params}')

    # action_conditioner = ''
    action_conditioner = vit.ActionConditioningNetwork(
        embed_dim=768,
        action_dim=28,
        mlp_hidden_dim=768*4,
        mlp_output_dim=768,
        num_heads=12,
        depth=16,
    )
    print('Loaded action_conditioner')
    
    # count for action
    num_params = sum(p.numel() for p in action_conditioner.parameters() if p.requires_grad)
    print(f'Number of action parameters: {num_params}')

    diffusion_model = vit.DiffusionModel(
        hidden_dim=768,
        output_dim=196*768,
        num_heads=8,
        depth=12,
        mlp_ratio=8
    )
    print('Loaded diffusion_model')
    
    # count for diffusion
    num_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    # Initialize weights
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Embedding):
            trunc_normal_(m.weight, std=0.02)

    print('Initializing encoder weights')
    for m in encoder.modules():
        init_weights(m)

    print('Initializing predictor weights')
    for m in predictor.modules():
        init_weights(m)

    print('Initializing action weights')
    for m in action_conditioner.modules():
        init_weights(m)

    print('Initializing diffusion weights')
    for m in diffusion_model.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    action_conditioner.to(device)
    diffusion_model.to(device)

    return encoder, predictor, action_conditioner, diffusion_model