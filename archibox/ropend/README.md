# ND RoPE Experiments

https://jerryxio.ng/posts/nd-rope/

Relevant scripts:
- `cifar_classification.py`: for training CIFAR models
- `imagenet_classification.py`: for training ImageNet models
- `imagenet_generalization.py`: for evaluating ImageNet resolution generalization
- `positional_embeddings.py`: implementations of positional embeddings

Note that this code depends on bnnp, my library of nn stuff, located here:
https://github.com/jxiong21029/bnnp, which should be pip installed.

Usage examples:
- train on ImageNet with SinCos: `torchrun --standalone --nproc_per_node=4
archibox/ropend/imagenet_classification.py pos_emb=fixed`
- train on ImageNet with uniform RoPE: `torchrun --standalone --nproc_per_node=4
archibox/ropend/imagenet_classification.py pos_emb=uniform_rotary`
- train on ImageNet with mixed RoPE: `torchrun --standalone --nproc_per_node=4
archibox/ropend/imagenet_classification.py pos_emb=uniform_rotary direction_spacing=None
learnable_rope=True`
