Title: Hello World for Neural Networks
## Intro to neurons

I want you to read this and think of how your brain learns new things. How does your brain adapt to situations. That is essently what we are trying to replicate with the computer when it comes to AI (for now)
Also this is a heavy read so feel free to take your time when you're reading this. There are math sections which yall losers who failed in 11th grade math like me can skip.

Visual Pattern Recofnigition is a hard program to write on the computer if not for neural networks
 Simple intuitions about how we recognize shapes - "a 9 has a loop at the top, and a vertical stroke in the bottom right" - turn out to be not so simple to express algorithmically. When you try to make such rules precise, you quickly get lost in a morass of exceptions and caveats and special cases. It seems hopeless.

Idea is to take a large number of handwritten digits (training data) -> develop system that learns from this (infer rules for recognizing handwritten digits) -> test on new image and see how well it predicts

Idea of neural networks comes from the basic principle of perceptron (used to model kind of some decision making)
*insert perceptron image and equation*
Large values of weights indicates that it matters a lot. By varying the weights and the threshold, we can get different models of decision-making.

Where the output from one layer is used as input to the next layer. Such networks are called feedforward neural networks. This means there are no loops in the network - information is always fed forward, never fed back

*insert neural network image*

The input layer of the network contains neurons encoding the values of the input pixels. As discussed in the next section, our training data for the network will consist of many 28 by 28 pixel images of scanned handwritten digits, and so the input layer contains 784=28×28 neurons. For simplicity I've omitted most of the 784 input neurons in the diagram above. The input pixels are greyscale, with a value of 0.0 representing white, a value of 1.0 representing black, and in between values representing gradually darkening shades of grey.

The second layer of the network is a hidden layer. We denote the number of neurons in this hidden layer by n, and we'll experiment with different values for n. The example shown illustrates a small hidden layer, containing just n=15 neurons.

The output layer of the network contains 10 neurons. If the first neuron fires, i.e., has an output ≈1
, then that will indicate that the network thinks the digit is a 0. If the second neuron fires then that will indicate that the network thinks the digit is a 1. And so on. A little more precisely, we number the output neurons from 0 through 9, and figure out which neuron has the highest activation value. If that neuron is, say, neuron number 6, then our network will guess that the input digit was a 6. And so on for the other output neurons.


## Notes for 3Blue1Brown Neural Network Videos

Neuron -> thing that holds a number between 0 to 1 (28 * 28 = 784 grids each holding a number b/w 0 to 1) : greyscale (white numbers : 1 black numbers : 0) 
This number is called it's activation function

Weights define strength of connections between each of neurons
Activation fn : transform sum(weights)
Biases : Shift the act function to left or right (for flexibility of network) 

Input : 784 neurons -> Output : 10 neurons (Probability of number being that number (0 to 9))
In between hidden layers -> activation of first layer determines activation of second layer and so on...

One Neuron:
    First layer weights to next layer neuron : wi
    Activations of first layer neurons : ai
    sigmoid(sum(ai * wi) ) = 0 to 1 probability. (Why sigmoid? It's an activation function that transforms any number to range 0 to 1 [see graph of sigmoid])
    One common function that does this is called the “sigmoid” function, also known as a logistic curve, which we represent using the symbol σσ.
     Very negative inputs end up close to 0, very positive inputs end up close to 1, and it steadily increases around 0. So the activation of the neuron here will basically be a measure of how positive the weighted sum is.
    Only activate meaningfully when weighted sum > 10 lets say.
    So the weights tell you what pixel pattern this neuron in the second layer is picking up on, and the bias tells you how big that weighted sum needs to be before the neuron gets meaningfully active.
    Hence we + bias to our weighted sum before applying the sigmoid function -> sigmoid(sum(ai * wi + bias))

total : 784 (inp) * 16 (neurons in second layer weights) + 16 (bias) * 16 + 16 (2nd layer) * 10 (last layer) = 13002 total weights and biases that can be tweaked and turned
Learning: Finding right weights and bias

This equation sigmoid(ai * wi + bias) is cumbersome to write.
Better notation : activation functions = column vector =    [a0]
                                                            [a1]
                                                            [a2]
                                                            [a3]
                                                            [a4]
                                                            .
                                                            [an]
all the weights as matrix : each row represents connection between one layer and a particular neuron in the next layer
[w0,1 w0,1 ..... w0,n]
[w1,0 w1,1 ....  w1,n]
[w2,0 w2,1 ....  w2,n]
[w3,0 w3,1 ....  w3,n]
...
[wk,0 wk,1 ....  wk,n]

multiple weights * activation function column + BIAS VECTOR
bias = [b0
        b1
        b2
        ..
        bn]

sigmoid(weights * activation + bias) = sigmoid([x]
                                               [y]
                                               [z])

Neuron is just a function that takes output of all neurons in prevs layer and spits out number between 0-1
Also no one uses sigmoid anymore. People use RELU = max(0, a) : just works better

The problem is that the sigmoid function becomes super flat at the extremes, when the weighted sum being passed in has a large magnitude.

That might not seem like an issue. But as we’ll learn in the next lesson, the process of training the neural network essentially boils down to wiggling the values of all the weights and biases and watching what happens, like a 13,000-dimensional game of hot and cold. When a little wiggle to a weight is effective, you do more of that, and when it isn’t helpful, you do the opposite.

But since the sigmoid function gets so flat, wiggling the weights doesn’t really do much of anything! Which means that the process of learning takes a long time, and training networks is painful.

So nowadays, people tend to use a function called ReLU (Rectified Linear Unit) instead. It’s an absurdly pompous name for a very simple function. ReLU spits out 0 for any negative input, and doesn’t change the positive inputs at all.

So wiggling the weights always gives useful feedback about how the network should change. This makes the training process much faster and more efficient, especially as there are more layers involved, which is where the “deep” in “deep learning” comes from.

You’ll notice how in these drawings each neuron from one layer is connected to each neuron of the next with a little line. 
This is meant to indicate how the activation of each neuron in one layer, the little number inside it, has some influence on the activation of each neuron in the next layer.

However, not all these connections are equal. Some will be stronger than others, and as you’ll see shortly, determining how strong these connections are is really the heart of how a neural network operates, as an information processing mechanism.

What we’ll do is assign a weight to each of the connections between our neuron and the neurons from the first layer. These weights are just numbers.
Each weight is an indication of how its neuron in the first layer is correlated with this new neuron in the second layer.
If the neuron in the first layer is on, then a positive weight suggests that the neuron in the second layer should also be on, and a negative weight suggests that the neuron in the second layer should be off.
So to actually compute the value of this second-layer neuron, you take all the activations from the neurons in the first layer, and compute their weighted sum.
w1a1+w2a2+w3a3+w4a4+⋯+wnan
w1​a1​+w2​a2​+w3​a3​+w4​a4​+⋯+wn​an​

So maybe first layer picks up on the edges -> Second layer picks up on the patterns -> Final layer actually knows what number we have as input.
So how does it learn? Whole point of training the model is that it can make the error as minimum as possible -> How does it do that?

# Gradient Descent
Instead of shiffting weights and calculating errors again and again (expensive) we can figure out the slope directly and can figure the direction we need to adjust our weight without going all the way back to our nerual network and calculating
slope? relation of change in weight and change in error : mathematically de/dw : change in error when i change the weight (basics calculus)
Neural Network
    Input: 784 numbers / pixels
    Output: 10 numbers
    Parameters: 13000 weights / biases

Initialise weights and bias randomly.
Define Cost function:
Input 3: output: grey and white colors on multiple neurons (1-9)
    What you gave me is utter trash (prediction), i need the output to be 3 {actual}
    Cost : square of mean error (MSE) : (actual - prediction) ** 2
        This is large when network doesn't seem to know what its doing and small when its close to 3. So this should work.
    Average cost of all training data : Measure of how lousy the network is.

Furthermore, the cost C(w,b) becomes small, i.e., C(w,b)≈0, precisely when y(x) is approximately equal to the output, a, for all training inputs, x. So our training algorithm has done a good job if it can find weights and biases so that C(w,b)≈0. By contrast, it's not doing so well when C(w,b) is large - that would mean that y(x) is not close to the output a for a large number of inputs. So the aim of our training algorithm will be to minimize the cost C(w,b) as a function of the weights and biases. How do you minimise a function? (High school math) you find the derivative.

our goal in training a neural network is to find weights and biases which minimize the quadratic cost function C(w,b).

Cost Function
    Input: 13000 weights / biases
    Output: 1 Number (cost)
    Parameters: Many many many training examples


How do we change these weights and biases so it makes it better? 
How to find an input that minimise value of function -> derivative of the cost function
Shift to the left if slope is pos, shift inp to right is slope is negative. 
If you keep repeating -> Approach local minimum of the function (ball rolling down a hill : lowest point)

We'd randomly choose a starting point for an (imaginary) ball, and then simulate the motion of the ball as it rolled down to the bottom of the valley. We could do this simulation simply by computing derivatives (and perhaps some second derivatives) of C - those derivatives would tell us everything we need to know about the local "shape" of the valley, and therefore how our ball should roll. Forget about all the predefined physics and ask yourself this - what law or laws of motion could we pick that would make it so the ball always rolled to the bottom of the valley?

we'll choose them (weights and biases) so the ball is rolling down into the valley. 

But what's really exciting about the equation is that it lets us see how to choose change in weights and bias so as to make cost function negative. In particular, suppose we choose
Δw = −η * ∇C
where η is a small, positive parameter (known as the learning rate).
Then we'll use this update rule again, to make another move. If we keep doing this, over and over, we'll keep decreasing C until - we hope - we reach a global minimum (bottom of the bowl) *insert bowl image here*
The magnitude of the update is proportional to the error term.

So gradient descent can be viewed as a way of taking small steps in the direction which does the most to immediately decrease C.

Which direction decreases C(x, y) most quickly? Here x, y is weights and bias
Gradient of function gives you direction of steepest increase (which dirn should you step to increase fn most quickly)
Taking negative of gradient gives you direction of step which decreases fn most quickly : WHAT WE WANT

Algorithm:
    Compute G(Cost Function) (G is gradient)
    Take small step in -G(C) direction.
    Repeat.


You'll get G(C(Weights)) and G(C(Bias)) -> Now update your inital weight and biases by:
    weights -= G(C(weights))
    bias -= G(C(bias))

Usually we use learning rate = 0.1 (alpha) for sampling
This is called backpropagation.

Each component of the negative gradient vector tells us two things. The sign, of course, tells us whether the corresponding component of the input vector should be nudged up or down. But importantly, the relative magnitudes of the components in this gradient tells us which of those changes matters more.

So after all this training, how does it perform on new data (testing) ? Pretty well.

## Backpropagation in depth
How does it calculate error what do you mean? Why does this work?
What is learning? Which weights and bias minimise a certain cost function 
    cost of single training exmaple : (actual output - prediction) -> add up square of differences -> total cost of network

Algorithm for computing the gradient:

The negative gradient cost vector (that we calculate) tells us two things:
    Magnitude = how sensitive it is to our data. eg: 3.20 -> 32 times greater 

Its just a lot of adjustments going on toghether. Fuck the 4 equations.

Focus on one example:
Input 2, lets say this is our activations based on random assignments of weights and biases

Activations
0 : 0.5
1 : 0.8
2 : 0.2
3 : 1.0 (white)
4 : 0.4
5 : 0.6
6 : 1.0 (white)
7 : 0.0 (black)
8 : 0.2
9 : 0.1

This is utter trash like we talked about.
We want it to be like:

0 : 0.0
1 : 0.0
2 : 1.0 (white)
3 : 0.0
4 : 0.0
5 : 0.0
6 : 0.0 
7 : 0.0
8 : 0.0
9 : 0.0

We want that third value to get nudged up while all others get nudged down. 
The sizes of these nudges should be proportional to how far away each curr value is from target val.
    What I mean by this:
        Increase in number 2 activation is more important than decrease in number 8 neuron (which is alr 0.2 close to 0)

So let's take 2 neuron
0.2 = sigmoid(sum(wn * an + b))
What can we change:
    Increase bias
    Incerase weights
    Change activations from prevs layer

It makes sense to increase weights in proportion to activation function of the neuron. 
    If neuron already has high activation function, you want to increase weight of that neuron.

We need to increase weights of neurons that contribute to neuron 2 in a more powerful way. That makes sense.
Similarily we change the activation function of previous layers in proportion to weights.

biggest increases in weights—the biggest strengthening of connections—happens between neurons that are the most active and the ones which we wish to become more active.

In practice it's really time consuming for the computer to calculate all gradients and upgrade weights for ALL training data. Hence we use schoastic gradient descent where we:
    Shuffle Training Data
    Randomly subdivide into mini batch
    Go thru all mini batches and make these adjustments until you converge for your local loss function

The bias associated with the digit-2 neuron should be increased, because that will cause the activation of the digit-2 neuron to increase. All the other output activations should be decreased, which requires decreasing their associated biases.


### What is bias

bias is just a number added all the time, even when the inputs are zero, so the neuron’s “neutral” point can shift up or down.
Mathematically:
output = w₁·x₁ + w₂·x₂ + … + b
Without that little + b, if every input xᵢ is zero, the output is forced to be zero—no wiggle room.

Intercept term:  It’s exactly like the intercept in a line y = m x + c.  Without c, all lines must pass through (0, 0).  With c, you can move the line up or down.

In short: bias gives each neuron a way to produce a nonzero output “by default” or to adjust its firing threshold, making the network much more flexible and able to fit real‐world data.


## Ok time for some loser math (math grads)
So this loss function huh? Pre-requisite -> Chain Rule.

backpropagation is an algorithm for calculating the gradient of the cost function of a network.
Entries of gradient vector are partial derivatives of the cost function wrt weights and biases. 

Once you know how sensitive the cost function is to activations in this second-to-last layer, you can just repeat the process for all the weights and biases feeding into that layer.

It’s remarkable that at the heart of every modern neural network lies nothing more exotic than a handful of basic math tools—matrix multiplication, derivatives, and the chain rule.
• Matrix multiplication elegantly encodes how signals flow from one layer of neurons to the next: each weight in the matrix acts like a tiny valve, amplifying or dampening its inputs in one sweeping operation.
• By defining a cost function that measures “how wrong” our network is, we turn the fuzzy notion of “learning” into the concrete problem of finding a minimum.  Derivatives and gradients become our compass, pointing us downhill through a vast, multidimensional landscape of error.
• And the chain rule—often dismissed in calculus class as “just another formula”—is the secret ingredient that unravels this entire web of dependencies.  It lets us trace exactly how a small tweak to any weight or bias will ripple through every layer, so we can update each parameter in just the right direction.
When you line up these simple ideas in the right order, you get back-propagation—and with it, the ability to build systems that can recognize images, translate languages, and more.

So what do these scary equations mean anyway?
Consider a simple neural network -> 4 neurons connected via 3 links hence 4 activations, 3 weights and 3 biases (for each neuron except first)

Goal = How sensitive the cost function is to these variables -> That way we need which knobs to turn in order to make the cost function decrease/minimal.

We're only going to focus on the activation of last 2 neurons 
The weight, prevs action and bias all together are used to compute z which in turn computes a which finally along with constant y helps us compute Cost (and a^(L-1) is influenced by its own bias, weight and activation of prevs prevs layer (a^(L-2)) and so on...)

Goal 1: How sensitive the cost function is wrt weight of that layer (What is the derivative of C with wrt W) : partial derivatives (small changes)
When you see this ∂w(L) term, think of it as meaning “some tiny nudge to w(L)”, like a change by 0.01. And think of this ∂C0​ term as meaning “whatever the resulting nudge to the cost is.” We want their ratio.

So this tiny nudge/change to weight(Layer) causes some nudge to z(Layer) which in turn causes some change to activation(layer) which directly influences Cost (C0) : makes sense.

multiplying these three ratios gives us the sensitivity of C0​ to small changes in w(L)

So calculating all these three ratios seperately using basic logic ->


z(L)a(L)C0​​=w(L)a(L−1)+b(L)=σ(z(L))=(a(L)−y)2​⟶w(L)∂z(L)​⟶∂z(L)∂a(L)​⟶∂a(L)∂C0​​​=a(L−1)=σ′(z(L))=2(a(L)−y)​
z(L) = w(L) * a(L-1) + b(L) = ∂z(L) / ∂w(L) = partial derivative of z wrt w (keeping w constant what is the derivative) = a(L-1)
a(L) = sigmoid(z(L)) = ∂a(L) / ∂z(L) = partial derivative of activation wrt z (keeping z constant) = sigmoid_derivative(z(L))
C0 = (a(L) - y)^2 = ∂C0 / ∂a(L) = 2(a(L) - y) # basic derivative

Putting it together:
∂C0 / ∂w(L) ​ ​=a(L−1) * σ′(z(L)) * 2(a(L) − y)

This formula tells us how a nudge to that one particular weight in the last layer will affect the cost for that one particular training example.

For the entire cost function just take the average of all Costs.

 Unfortunately, when the number of training inputs is very large this can take a long time, and learning thus occurs slowly. Since you need to find partial derivatives of Cost function wrt all training inputs -> ∇Cx separately for each training input, x, and then average them,
  ∇C = 1/n ∑ ∇Cx.

### Stochastic GD

An idea called stochastic gradient descent can be used to speed up learning. The idea is to estimate the gradient ∇C by computing ∇Cx for a small sample of randomly chosen training inputs. By averaging over this small sample it turns out that we can quickly get a good estimate of the true gradient ∇C, and this helps speed up gradient descent, and thus learning


We can think of stochastic gradient descent as being like political polling: it's much easier to sample a small mini-batch than it is to apply gradient descent to the full batch, just as carrying out a poll is easier than running a full election.


Partial Derivative wrt Bias
For bias it's 1.


Determined how changes to the last weight and last bias in our super-simple neural network will affect the overall cost, which means we already have two of the entries of our gradient vector.

By tracking the dependencies through our tree, and multiplying together a long series of partial derivatives, we can now calculate the derivative of the cost function with respect to any weight or bias in the entire network. We’re simply applying the same chain rule idea we’ve been using all along!

And since we can get any derivative we want, we can compute the entire gradient vector. Job done! At least for this network.
 *insert images here of equations*

These chain rule expressions are the derivatives that make up the gradient vector, which lets us minimize the cost of the network by repeatedly stepping downhill. 

## Backpropagation in Depth 
For fixed data, value of Loss depends on the curve -> parameters of the curve to be precise -> this means its a function of parameters (in our case weights and biases) hence it's called a loss function.

To get the best curve function(y = mx + c) we define a loss function(wi * ai + bi)
which basically does 3 things:
    - Construct Curve
    - Find Distance to data points
    - Output loss
Aim -> We need to find configuration of the parameters (weights and bias) to figure the minimum loss (basically minimising loss function wrt configurations/parameters)
Plugging the coefficients obtained by minimising the loss function into curve equation
gives us best description of the data.

How do we find these configurations?
We can keep iteratively change the knob of weights/biases one at a time to see whether the resulting curve is a better fit. 

This is bad. Slow. It does work.

How to be more intelligent?
Calculus. Since these are curves, they follow the property of diffrentiability.
Allows us to get the optimised knob settings much better.
Computer tells us which direction to change the knob and by how much
    We are essentially asking the computer to predict the future and estimate the effect of knob adjustments on the loss function without doing trial&error (changing knobs one at a time and seeing how it fits the curve)


This is based on derivatives.
Lets take only one parameter w1. Try to change it's knob to see if it predicts our desired output. 
We can draw a graph easily of x axis = w1, y axis = loss
What we need to do is to calculate the minimal point on this curve (the minimal loss)
It helps if we know how the curve is shifting.
    Knowing local behaviour of function (going up or down) can guide us to better knob adjustmenets.

Amount of change in output (y) per unit change in the input (x) = slope of straight line.
limit delta(x) -> 0 [delta(y)/delta(x)] = dy/dx = This is called a derivative.
Derivative tells us about instantaneous rate of change / steepness of the curve around that point. Which is what we need in order to get the minimal loss function.

Ball rolling down the hill of a graph.
Take a loss value -> take it's derivative (steepness curve)
Adjust/Move in the direction opposite of the derivative -> as you keep moving once derivative is 0 -> stop = Loss is minimal (This is what minima is)

Now applying to general sense.
If we have 2 parameters/knobs.
So the derivative has multiple choices -> wrt param1, param2 and both
Derivative is just change in output rate / change in input rate (in this case we have 2 inputs)

The loss function has 2 seperate (partial) derivatives.

Parameters: k1, k2
1:partial(Loss)/partial(k1) : partial derivative wrt one parameter specifies how the loss changes as we nudge it, keeping the other parameters constant.

2:same for partial(loss)/partial(k2) : rate of change of output if u hold k1 constant and slightly nudge k2.

These two values are plot into a vector called the gradient vector.
G = [[1]
     [2]]

Mapping from two inp values to another two numbers (first signifies how much the output changes for tiny change in first inp, same for second)

Gradient gives the direction of steepest ascend.
So to minimise the function -> take steps in the opposite direction.

This iterative process of nudging the parameters in the opposite direction of gradient vector is called gradient descent.


More:
Components of gradient vector guide the adjustments we need to make.
if for a particular configuration:
    the partial derivative of loss wrt k1 is positive means increase k1 = increase loss
    hence we need to decrease (opposite sign of gradient) k1 to decrease loss.

How do we access the derivative of the loss function in the first place?
Chain Rule -> tells you how to compute derivative of combination of functions.
d/dx f(g(x)) = ? 
How to compute derivative of combination of two functions when one of them is an input to another (Neural Network)

Suppose we know the indivial derivatives of the two machines f and g.
f'(x0)  = slope of f(x) at pt x0

Machine 1
x -> value : g(x) -> derivative : g'(x) {local steepness of first function}
Number that is fed into second machine isn't x (that's processed by the first function)
So the thing that is being plugged into second machine is g(x)

Machine 2
Value : f(g(x)) -> Derivative : f'(g(x))

Imagine you nudge the knob x by a tiny amount delta (x+delta)
This input nudge when it comes out the first machine will be multiplied by the derivative of g since derivative is rate of change in output / rate of change in input.
The output will increase by delta = g'(x)*delta {this is essentially a nudge to the second machine input}

This means for each delta increase in input we bump the output by f'(g(x)) * g'(x)*delta
Hence the derivative when you divide by delta
looks like this : f'(g(x)) * g'(x) -> which is essentially what chain rule says.

For each of the parameters write down its effect on the loss function in terms of simple diffrentiable ops

k0 -> f1(k0) -> f2(f1(k0)) -> loss

Sequentially apply the chain rule to compute gradient of loss wrt each parameter

The derivatives of loss wrt input data don't really matter much
But the derivatives of loss wrt to parameters (weights and biases) matter.

Once these gradients are found.
We're going to tweak these parameters based on the loss function gradient.
Slightly tweak the knobs oppositve to the gradient.
ki = ki - learning_rate * [partial(L) / partial(k1)]

After each adjustment we need to redo forward and backward passes since the loss functions have changed.

Performing this loop of:
    - Forward Pass
    - Backward Pass
    - Nudge knobs
    - Repeat 1-3
is the essence of training modern machine system.
As long as your fancy model can be decomposed as a sequence of differentiable functions, you can apply backprop to optimise the parameters.


### More about backprop
Four equations: heavy on the math side but this is a great resource if you want to dive deep into the math behind [backprop](https://neuralnetworksanddeeplearning.com/chap2.html)
Why is it fast?
In what sense is backpropagation a fast algorithm? To answer this question, let's consider another approach to computing the gradient. Imagine it's the early days of neural networks research. Maybe it's the 1950s or 1960s, and you're the first person in the world to think of using gradient descent to learn! But to make the idea work you need a way of computing the gradient of the cost function. You think back to your knowledge of calculus, and decide to see if you can use the chain rule to compute the gradient. But after playing around a bit, the algebra looks complicated, and you get discouraged. So you try to find another approach. You decide to regard the cost as a function of the weights C=C(w) alone (we'll get back to the biases in a moment). You number the weights w1,w2,…, and want to compute ∂C/∂wj for some particular weight wj. An obvious way of doing that is to use the approximation
∂C∂wj≈C(w+ϵej)−C(w)ϵ,(46)
where ϵ>0 is a small positive number, and ej is the unit vector in the jth direction. In other words, we can estimate ∂C/∂wj by computing the cost C for two slightly different values of wj, and then applying Equation (46). The same idea will let us compute the partial derivatives ∂C/∂b

with respect to the biases.

This approach looks very promising. It's simple conceptually, and extremely easy to implement, using just a few lines of code. Certainly, it looks much more promising than the idea of using the chain rule to compute the gradient!

Unfortunately, while this approach appears promising, when you implement the code it turns out to be extremely slow. To understand why, imagine we have a million weights in our network. Then for each distinct weight wj
we need to compute C(w+ϵej) in order to compute ∂C/∂wj. That means that to compute the gradient we need to compute the cost function a million different times, requiring a million forward passes through the network (per training example). We need to compute C(w)

as well, so that's a total of a million and one passes through the network.

What's clever about backpropagation is that it enables us to simultaneously compute all the partial derivatives ∂C/∂wj
using just one forward pass through the network, followed by one backward pass through the network. Roughly speaking, the computational cost of the backward pass is about the same as the forward pass* *This should be plausible, but it requires some analysis to make a careful statement. It's plausible because the dominant computational cost in the forward pass is multiplying by the weight matrices, while in the backward pass it's multiplying by the transposes of the weight matrices. These operations obviously have similar computational cost.. And so the total cost of backpropagation is roughly the same as making just two forward passes through the network. Compare that to the million and one forward passes we needed for the approach based on (46)! And so even though backpropagation appears superficially more complex than the approach based on (46), it's actually much, much faster.

So what does the backprop really do?
let's imagine that we've made a small change Δwljk to some weight in the network, wljk: 
That change in weight will cause a change in the output activation from the corresponding neuron: 
That, in turn, will cause a change in all the activations in the next layer: 
Those changes will in turn cause changes in the next layer, and then the next, and so on all the way through to causing a change in the final layer, and then in the cost function: 

The change ΔC in the cost is related to the change Δwljk in the weight by the equation
ΔC≈∂C∂wljkΔwljk.(47)
This suggests that a possible approach to computing ∂C∂wljk is to carefully track how a small change in wljk propagates to cause a small change in C. If we can do that, being careful to express everything along the way in terms of easily computable quantities, then we should be able to compute ∂C/∂wljk.


Let's try to carry this out. The change Δwljk causes a small change Δalj in the activation of the jth neuron in the lth layer. This change is given by
Δalj≈∂alj∂wljkΔwljk.(48)
The change in activation Δalj will cause changes in all the activations in the next layer, i.e., the (l+1)th layer. We'll concentrate on the way just a single one of those activations is affected, say al+1q, 

We're computing the rate of change of C with respect to a weight in the network. What the equation tells us is that every edge between two neurons in the network is associated with a rate factor which is just the partial derivative of one neuron's activation with respect to the other neuron's activation. The edge from the first weight to the first neuron has a rate factor ∂alj/∂wljk. The rate factor for a path is just the product of the rate factors along the path. And the total rate of change ∂C/∂wljk is just the sum of the rate factors of all paths from the initial weight to the final cost

backpropagation algorithm as providing a way of computing the sum over the rate factor for all these paths. Or, to put it slightly differently, the backpropagation algorithm is a clever way of keeping track of small perturbations to the weights (and biases) as they propagate through the network, reach the output, and then affect the cost.




### Time for some code
We have gathered enough theory to build our own program that correctly identifies number from handwritten images. This is the hello world of neural networks.

### About the dataset
The MNIST data comes in two parts. The first part contains 60,000 images to be used as training data. These images are scanned handwriting samples from 250 people, half of whom were US Census Bureau employees, and half of whom were high school students. The images are greyscale and 28 by 28 pixels in size. The second part of the MNIST data set is 10,000 images to be used as test data. Again, these are 28 by 28 greyscale images. We'll use the test data to evaluate how well our neural network has learned to recognize digits. To make this a good test of performance, the test data was taken from a different set of 250 people than the original training data (albeit still a group split between Census Bureau employees and high school students). This helps give us confidence that our system can recognize digits from people whose writing it didn't see during training.

### MNIST from Scratch (Using Numpy)

*insert code here*

**input**
sample_digit = handwritten 9 and training data


**output**
Epoch 0: 2718 / 10000
Epoch 1: 3465 / 10000
Epoch 2: 3928 / 10000
Epoch 3: 4300 / 10000
Epoch 4: 4650 / 10000
Epoch 5: 5049 / 10000
Epoch 6: 5531 / 10000
Epoch 7: 5907 / 10000
Epoch 8: 6255 / 10000
Epoch 9: 6504 / 10000
Epoch 10: 6704 / 10000
Epoch 11: 6888 / 10000
Epoch 12: 7043 / 10000
Epoch 13: 7165 / 10000
Epoch 14: 7266 / 10000
Epoch 15: 7358 / 10000
Epoch 16: 7441 / 10000
Epoch 17: 7508 / 10000
Epoch 18: 7583 / 10000
Epoch 19: 7661 / 10000
Epoch 20: 7722 / 10000
Epoch 21: 7793 / 10000
Epoch 22: 7830 / 10000
Epoch 23: 7878 / 10000
Epoch 24: 7914 / 10000
Epoch 25: 7957 / 10000
Epoch 26: 7991 / 10000
Epoch 27: 8052 / 10000
Epoch 28: 8073 / 10000
Epoch 29: 8119 / 10000
Accuracy : 81.19% (It means it predicted 8119 images correctly out of 10000)

Network Output Activations (probability for each digit 0-9):
  Digit 0: 0.21%
  Digit 1: 0.08%
  Digit 2: 0.06%
  Digit 3: 0.53%
  Digit 4: 17.36%
  Digit 5: 2.49%
  Digit 6: 2.85%
  Digit 7: 0.12%
  Digit 8: 48.34%
  Digit 9: 70.89% # highest out of all these hence answer is 9 

Predicted digit: 9 # HELL YEAH

### MNIST from Pytorch
makes our lives much easier
*insert code here*

**output**
Using device: cuda
Loading pre-trained model from /content/model_state.pt
Model loaded. Skipping training for now.

--- Prediction for sample_digit.jpg ---
Network Output Activations (probability for each digit 0-9):
  Digit 0: 0.00%
  Digit 1: 0.00%
  Digit 2: 0.00%
  Digit 3: 0.00%
  Digit 4: 0.00%
  Digit 5: 0.00%
  Digit 6: 0.00%
  Digit 7: 0.00%
  Digit 8: 0.00%
  Digit 9: 100.00%

Predicted digit: 9

### MNIST from MiniTorch (my own implementation of pytorch)
You can see the project here: [MiniTorch](https://samitmohan.github.io/minitorch)
*insert code here*

### Comparing Accuracy across these.
Pytorch clearly wins.

Also loss vs accuracy graph:
*insert image here*

Makes sense, loss goes down and accuracy of the model goes up over time.

## References
[3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
[Backpropagation](https://www.youtube.com/watch?v=SmZmBKc7Lrs)
[Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)
[StatQuest](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
Books : MLforStatQuest, Why Machines Learn, Grokking Deep Learning, Deep Learning with Python, [Michael Nelson](https://neuralnetworksanddeeplearning.com/)

## Conclusion

In a much larger sense these parameter / weights are just probabilities of all english words that are given to the LLM.
LLMs have hundreds of these parameters. They begin at random and repadetely refined based on text (chatgpt uses wikipedia text) -> Backpropagation is used to tweak all paramters so that it makes the model a little more likely to choose the true last word (accurate word)instead of other words
I am going to the bank to deposit some _

Probabilties : money, cash, blood, ashes, etc... (True last word here is money) 
A good video on this: [LLMs explained](https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)

Classyfying Images or Generating Texts (GPT) uses these ideas at the core. 

Neural networks are the base for understanding AI. Transformers, Attention and all cutting edge AI papers are based on this simple key idea of y = mx + c and how to find the slope and intercept of this line in order to minimise our loss. It's so interesting to see that behind these complicated ideas it's just high school math and common sense. I spent the weekend reading these resources and writing code and it felt so nice to be back in touch with school math (although I was never good at it) and discover AI from first principles (perceptrons) I was going to write a section about attention and transformers but that is a blog for another time, this is a heavy read as it is ;)

Code for all of this can be found here -> github link (MNIST)
