# Code for "GroupCover: A Secure, Efficient and Scalable Inference Framework for On-device Model Protection based on TEEs"

## Environment
As same as [knockoff]([url](https://github.com/tribhuvanesh/knockoffnets)),:
- Python 3.8
- Pytorch 1.1

## Code Structure
- knockoff: This section focuses on the basic Model Stealing methods.
    - model: Here is the model implement for alexnet, resnet, vgg and so on. Corresponding to our obfuscation method, we rewrite the inference function of the network here.
- models: Store attack results, both model checkpoint(do not upload) and attack logs. Here, adversary fold stores black-box attack result.
- scripts: Provide scripts to replicate and attack other schemes. 
- statisitcal: mainly analysis and reverse code.
- implement & plot: some simulation verification, plot code and real implement in TEEs. Go into this folder to check the details. 

## Reproduce
- The attack pipeline is based on Knockoff. That means, you should execute `train_victim.sh` and `random.sh` first. Then, you can take `gen_xxx.sh` to generate the obfuscated model. Here we give the reproduce of `SOTER`, `ShadowNet`, `NNSplitter` and `Magnitude`. You can change models and datasets by conf params ahead of the file.
- Use 'knockoff_xxx.sh' to attack them. Have fun!
- In real TEES. Maybe it's hard to find a TDX/SEV machine. If you are interest in our work, plz contact our teams. We have RICH machine resources for research. Work for efficient!

## Note: 
We use advanced model stealing attacks to measure scheme security protection capabilities. In order to protect the intellectual property of others, we keep the author's information in the code, which does not affect the fact that GroupCover is anonymous.
