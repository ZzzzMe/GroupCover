# Code for "GroupCover: A Secure, Efficient and Scalable Inference Framework for On-device Model Protection based on TEEs"

## Environment
As same as knockoff[],:
- Python 3.3
- Pytorch 1.1

## Code Structure
- knockoff: This section focuses on the basic Model Stealing methods.
    -- model: Here is the model implement for alexnet, resnet, vgg and so on. Corresponding to our obfuscation method, we rewrite the inference function of the network here.
- models: Store attack results, both model checkpoint(do not upload) and attack logs. Here, adversary fold stores black-box attack result.
- scripts: Provide scripts to replicate and attack other schemes. 
- statisitcal: mainly analysis and reverse code.
- implement & plot: some simulation verification and plot code.
