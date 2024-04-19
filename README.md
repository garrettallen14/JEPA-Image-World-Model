# Minecraft JEPA Image World Model


A recent paper ([https://arxiv.org/abs/2403.00504](https://arxiv.org/abs/2403.00504)) introduced the JEPA Image World Model, which was trained on a subset of OpenAI's Visual PreTraining (VPT) Minecraft contractor dataset. This model can predict the future state of the Minecraft environment based on an initial frame and a sequence of future actions.

![Given a single frame and a series of future keyboard and mouse inputs, the model predicts the sequence of future images (future environment)](https://imgur.com/Zq8ojmc)

## Architecture Details


The JEPA Image World Model consists of several components:

1.  Encoder (~113M parameters): A Vision Transformer that takes a single frame at time t=0 as input and produces a latent representation.
2.  Action Conditioning Network (~43M parameters): This network performs cross-attention between the encoded frame representation and the user's keyboard and mouse inputs from time t=-24 to t=0.
3.  Predictor Network (~213M parameters): This network takes the output of the Action Conditioning Network and produces a representation approximating the frame at time t=1.
4.  Diffusion Transformer (~141M parameters): Used for visualization and evaluation purposes.

The model was trained over two weeks on a single NVIDIA RTX 3060 (13GB VRAM). The Encoder's weights are frozen using a moving average, and the Predictor is trained to produce a representation similar to the Encoder's representation of the actual frame at t=1.

## JEPA IWM vs Sora

![The architecture has implications to be a far more efficient/effective World Model than Sora.](https://imgur.com/V0afEDS)
OpenAI's Sora has recently gained attention, with claims that diffusion world models are key to AGI. However, the JEPA Image World Model offers a more efficient approach to world modeling.

![A recent presentation by Sora leads claimed video generation is the key to AGI.](https://imgur.com/4TGGsgV)
The video generation portion of Sora is arguably wasteful, as the primary benefit of the architecture is the ability to simulate future environments and plan optimal actions based on different simulated outcomes. Sora maps image pixels back to image pixels, which is inherently inefficient, as the generated video needs to be converted back into latent space for planning.

JEPA World Models, on the other hand, encode the environment into a latent space representation and produce future versions of the environment while maintaining this representation. This approach is more efficient and better suited for world modeling and action planning.

## Possible Scale Implications

![Empirical scaling for Sora](https://imgur.com/76s9YD0)

According to a [recent report](https://www.factorialfunds.com/blog/under-the-hood-how-openai-s-sora-model-works) by Factorial Funds, the entire Sora architecture is estimated to have ~20B parameters. This implies that the base compute shown above totals ~600M parameters, which is approximately 1.5x the compute of the trained JEPA Image World Model.

This implies that the base compute shown above totals ~600M parameters. This is around ~1.5x the compute of the JEPA Image World Model trained.

![](https://imgur.com/NM9pKsD)

It would be interesting to see the performance of the JEPA Image World Model when scaled up to a similar level as Sora. The potential for more efficient and effective world modeling and action planning is promising.
