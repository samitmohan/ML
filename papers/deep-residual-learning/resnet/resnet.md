# Resnet From Scratch
# Resnet Pytorch
# Resnet Hugging Faces (resnet-50)


## Resnet
Suppose you have an image, you need to classify it? How do you do it.
Simple answer is : You use some sort of CNN / Backprop and have multiple layers.
    Problem : Vanishing Gradients bcs of so many layers.

How to fix this problem?
Resnet.


Residual connections (skip connections) makes it easier for network to learn.
Instead of learning a direct mapping from input x to output y (which is what causes vanishing gradients for large amount of layers) resnet learns the residual (difference) between input and output.

y = F(x) {output of series of layers} + x (input to block)
The addition of x (skip connection) allows network to skip certain layers during training making it easier to optimise.

If a layer is not useful, the network can simply learn $(F(x) = 0\$, and the output becomes $(y = x\$, effectively skipping the layer.

Resnet has = Residual Blocks
    - Conv Layer
    - Batch Normalisation
    - reLU activation
    - Skip connection (add input x to output of block)

y = reLU(F(x) + x)

Multiple types of resnets -> resnet50 has 50 layers and so on...

Allows the network to "skip" certain layers during training if those layers are not useful.

This means that the network can "skip" learning unnecessary transformations, but the layers themselves are not dropped or deactivated.

### More intuitive example
You have a lower resolution image and you want to pass it to your network so it outputs the correct resolution image for the input.

lowerResImage (input)
higherResImage (output)

input - output : residual (the missing piece we need) : minimisedFeatures

now we can just do input + residual to get the output. (network only learns the actual bit we care about which is the residual)

Doesn't need to retain the entire signal.


---

So you have blocks (residual blocks) let's say of size 3
input -> block1 -> block1 + inp -> block 2 -> block2 + inp -> block3 -> block3 + inp

where block1 -> 3 layers (assume) and same for other blocks...

Why does this work?
The function it computes augments the existing data.
Since the input is being passed along and will be available to later layers of the network the job of a layer is no longer to figure out everything imp about the input that needs to be passed along but rather to figure out what information it can add ON TOP of the input to make the process easier.

The block doesn't have to start by figuring out what info the input contains.
Instead the block starts by passing along all the input.
(Simpler to learn, better info)

Any layer in the network will have short path by which loss gradients can arrive and usefully update that loss. (Since each block has one path that goes around them and one path that goes through the layers, the gradient flows through both those paths (relatively short paths))

Much better than our shallow simple network. 
Speeds up the backprop
Modularity (each block has same struture and we are able to short-circuit around those blocks in training it's relatively easy to add more blocks and build a deeper neural network

## Concerns
- Shape mismatch (input and output by addition/concatenation) -> activation output has a mismatch of size wrt input size.

We need some operation that will reshape the input (probably add 0 padding to match the output size) :: SLOW

OR

We need to add input to a specific part of that block's output.

Soln : some special case on how we're getting shape of input to match up with the shape of first block and then subsequent blocks have a common shape so that this problem is minimised.

Concatenation many times: Many more weights : Large activation tensor -> We can end up in an explosion in parameters.

Addition > Concatenation.

What dimensions should match up?
- Height and Width
- Channels 

For Height and Width matchup we use convolutional block shape matching
- Strides = 1 to preserve height * width

1 * 1 strides and appr padding to ensure that ht and wdth of image is preserve and that way when we add input to output atleast the height and width will match up.

But we also want channel dimensions to also match between input and output.

We do this using 1 * 1 convulation layer (one pixel matching) 
Output block = same ht and wdth as input block AND each of the neurons in output will get it's input from just 1 pixel from input
BUT it will get inputs from the entire depth of that pixel's channel.


---
## BottleNeck
In Resnets there are two kinds of residual blocks
- Basic Block (Used in Resnet18,34)
- Bottlneck (Used in Resnet50,101)

It's designed to reduce cost of deep network by 'squeezing' (bottlenecking) number of channels before doing the expensive 3x3 convolution and then unsqueezing back.
- 1x1 conv reduces channel dimension from Cin to Cmid = Cn/4
- 3x4 conv on reduced Cmid channels to Cout = Cin * expansion

(Resnet50 has expansion factor 4)

The 1×1 “squeeze” reduces channels → far fewer multiplications in the 3×3 layer.
The 1×1 “unsqueeze” restores the full representational capacity.
Overall, the block has three convolutions but less cost than naively stacking two 3×3 layers at full width.

The expansion factor in a ResNet Bottleneck block is a multiplier that determines how many more channels the output feature map will have compared to the input feature map within that block. Specifically, it's the factor by which the number of channels is increased in the final 1x1 convolution of the bottleneck.
In the common ResNet architectures (like ResNet-50, ResNet-101, ResNet-152), the expansion factor is 4.

So only for the first convolution we squeeze the data so that it is faster to convulate and then in second layer we unsqueeze it again by the expansion factor so that its same as output channels.

It's the factor by which the number of channels is expanded in the final 1x1 convolution to match the desired output channels, and often to match the input channels for the shortcut connection. This expansion is key to restoring the feature map's representational capacity after the initial channel reduction.