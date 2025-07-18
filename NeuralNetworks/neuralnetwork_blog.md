# Index
1. [Introduction](#introduction)
2. [Gradient Descent](#gradient-descent)
3. [Backpropagation](#backpropagation)
4. [Time 4 Math](#time-4-math)
5. [Code](#coding-time-coding-time-coding-time)
6. [Problems](#problems-with-gradient-descent)
7. [Conclusion](#conclusion)
8. [Optional : More Math](#optional)
9. [References](#references)

## Introduction 

I want you to read this and think of how your brain learns new things. 

How does your brain adapt to situations? That is essentially what we are trying to replicate with the computer when it comes to AI (for now)

Also this is a heavy read so feel free to take your time when you're reading this. There are sections which yall losers who failed in 11th grade math like me can skip.

### Perceptron

The idea of neural networks originates from perceptrons used in the early 1950s.

![Perceptron](https://i.ibb.co/MyYjM37N/perceptron.png)

Networks composed of perceptrons where outputs from one layer feed into the next are called feedforward neural networks.

While perceptrons were limited to linear decision boundaries, modern neural networks use multiple layers and non-linear activation functions to learn complex patterns.

### Problem Statement

*Given an image which with a handwritten number, how you identify the number?*

It's simple for us because we've seen numbers all our lives. But how do *WE* identify them? Sure we've seen them all our lives but how does our brain see them? 

We know what strokes 2's have, it has a little horizontal line as a base & a slight curved S shaped line. So it must be a 2.

**How can we make a computer learn this?**

*Visual Pattern Recognition* is a hard program to write on the computer if not for neural networks.

Simple intuitions about how we recognize shapes - **"a 9 has a loop at the top, and a vertical stroke in the bottom right"** - turn out to be not so simple to express to a computer.

When you try to make such rules precise it doesn't work.

Idea is:
- Take a large number of handwritten digits (training data)  
- Develop system that learns from this (infer rules for recognizing handwritten digits) 
- Test on new image and see how well it predicts.

Try to think about this in a smarter way.

### Framework 
- Each image is *28 * 28* pixels (which can be represented as numbers)
- Each of these pixels has a *weight* value attached to it which indicates how much those pixels matter.
    - *Example:* in a '2' only pixels containing horizontal line as a base and a S shaped curve pixels will matter, all other pixels are just dead weight.
    - Large values of weights indicates that it matters a lot. By varying the weights and the threshold, we can get different models of decision-making.
- Somehow make sure all these images learn some underlying pattern and are able to recognise what the image is saying.
    - This happens via some trial and error, predicting some jargon at first (known as the *forward pass*) 
    - Then hopefully minimising that error and making some sense out of it (known as the *backward pass*)
- Given a new image, how well can the neural network predict what number this image is?


### Basics Revisited

- **Neuron**: Thing that holds a number between 0-1 i.e activation value (0-1, grayscale, white numbers : 1, black numbers : 0)
- **Activation Functions**: Introduce non-linearity in data otherwise every data is just a bunch of matrix multiplications on other layers (this is basic linear algebra if you don't know this kys #1) 
    - *Examples:* Sigmoid, tanh, ReLU (more on these later, just think of them as math functions which transforms your data x into f(x))
- **Weights**: Determine the strength of connections.
- **Biases**: Shift activations to improve flexibility.

#### Bias and Weights Explained
---

### **Bias and Weights Explained**

Bias and weights are the building blocks of neural networks. Let’s break them down:

#### **Bias: The Intercept**
Bias is like the intercept in the equation of a line \(y = mx + c\). It shifts the output up or down, even when the inputs are zero. Without bias, the output is forced to be zero when all inputs are zero, which limits the flexibility of the model.

Mathematically:
$\text{output} = w_1 x_1 + w_2 x_2 + \dots + b$

Here, \(b\) (bias) ensures that the neuron’s “neutral” point can shift up or down, allowing the model to better fit the data.

Think of bias as the constant $c$ in $y = mx + c$. Without $c$, all lines must pass through the origin (0, 0). With $c$, you can move the line up or down, giving the model more flexibility.

Without bias, zero input forces zero output.

Bias is just a number added all the time, even when the inputs are zero, so the neuron’s “neutral” point can shift up or down.


#### **Weights: The Slope**
Weights determine the strength of the connection between inputs and outputs. They act like the slope \(m\) in \(y = mx + c\), controlling how much influence each input $x_i$ has on the output.

Large weights amplify the importance of certain inputs, while small weights reduce their impact. By adjusting weights during training, the model learns which inputs matter most for making accurate predictions.



## Neural Network

- Input : 784 neurons → Output : 10 neurons (Probability of number being that number (0 to 9))
- In between hidden layers → activation of first layer determines activation of second layer and so on... Just how neurons in our brain work.

**Visually:**

![Neural Network](https://i.ibb.co/RGjGVsjF/nn.png)
![Neural Network2](https://i.ibb.co/HDtj57cf/neural-Network-Photo.png)


Training data for the network will consist of many 28 * 28 pixel images of scanned handwritten digits, and so the input layer contains **784=28×28 neurons.**

The input pixels are greyscale, with a value of **0.0** representing **white**, a value of **1.0** representing **black**, and in between values representing gradually darkening shades of **grey.**

The second layer of the network is a *hidden layer*. We denote the number of neurons in this hidden layer by n, and we'll experiment with different values for n. The example shown illustrates a small hidden layer, containing just n=15 neurons.

The output layer of the network contains 10 neurons. If the first neuron fires (i.e., has an output = digit 0) which indicates that network thinks the digit is a 0. If the second neuron fires then that will indicate that the network thinks the digit is a 1. And so on. 

You’ll notice how in these drawings each neuron from one layer is connected to each neuron of the next with a little line. 

This is meant to indicate how the activation of each neuron in one layer, the little number inside it, has some influence on the activation of each neuron in the next layer.

However, not all these connections are equal. Some connections are more equal than others (*shoutout George Orwell*) and determining how strong these connections are is really the heart of how a neural network operates.

### Neural Network Setup
- Think about any math problem, y = x (we usually solve for x, y is known to us)
- Any ML problem involves some sort of y = mx + c (slope equation of a line) where we already have y (output) and x (our sample training data), we just need to find m (slope or weights) and c (intercept or bias) so that we can fit a model to our input and get some meaningful data / insights from it.

Question:  What happens in one layer?
- First layer weights to next layer neuron : $w_i$
- Inputs of first layer neurons : $x_i$

- Calculate slope line for all, y = mx + c = $sum(w_i*x_i + bi)$ ($x_i$ is replaced by $a_i$ here after 'activating' all inputs $x_i$)

Weights tell you what pixel pattern this neuron in the second layer is picking up on, and the bias tells you how big that weighted sum needs to be before the neuron gets meaningfully active.

Basic equation → y = mx + c translates to $y = sum(weights * activations + bias)$ for all neurons
    also this y is just a bunch of random numbers, they don't make sense, we need to squash this data into a range so that we truly can find some meaning to these. That's where activation functions come in.

A lot of functions can do this, one of them is sigmoid which is just a fancy name for a function that translates any real-valued number to a value between 0 and 1.

You put this $sum(weights * activations + bias) = z$ into the sigmoid activation function = $sigmoid(z)$

Once again:
- Input : 28 * 28 = 784 neurons, Let's say I have 10 neurons in second layer.
- Total : 784 (input) * 10 (neurons in second layer weights) + 10 (bias) * 10 + 10 (2nd layer) * 10 (last layer) ~ *8000* total weights and biases that can be tweaked and turned on or off per your choice.

Learning: Finding right weights and bias (so that our network learns underlying patterns of these numbers (i.e strokes))


The equation for a single neuron's output, often written as:
$\sigma\left(\sum_i a_i w_i + b\right)$
can be cumbersome to write, especially for entire layers. We can use the tools of linear algebra—vectors and matrices—to express this much more cleanly.


### Representing Layers with Vectors and Matrices

- **Activations (\(a\))**: The outputs from a previous layer can be represented as a single column vector:

$$
a = \begin{bmatrix}
a_0 \\
a_1 \\
a_2 \\
\vdots \\
a_n
\end{bmatrix}
$$

- **Weights (\(W\))**: All the weights connecting the previous layer to the current layer can be organized into a single matrix:

  $$
  W = \begin{bmatrix}
  w_{0,0} & w_{0,1} & \dots & w_{0,n} \\
  w_{1,0} & w_{1,1} & \dots & w_{1,n} \\
  \vdots  & \vdots  & \ddots & \vdots \\
  w_{k,0} & w_{k,1} & \dots & w_{k,n}
  \end{bmatrix}
  $$

  Here, $w_{j,i}$ is the weight connecting the $i$-th neuron of the previous layer to the $j$-th neuron of the current layer.

- **Biases (\(b\))**: The biases for each neuron in the current layer can be represented as a column vector:

  $$
  b = \begin{bmatrix}
  b_0 \\
  b_1 \\
  b_2 \\
  \vdots \\
  b_k
  \end{bmatrix}
  $$

Using this notation, the calculation for the pre-activation values $z$ of an entire layer becomes a matrix-vector operation:

$$
z = W a + b
$$

Finally, to get the output activations of the current layer, we apply the sigmoid function $\sigma$ element-wise:

$$
a' = \sigma(z) = \begin{bmatrix}
\sigma(z_0) \\
\sigma(z_1) \\
\sigma(z_2) \\
\vdots \\
\sigma(z_k)
\end{bmatrix}
$$

Neuron is just a function that takes output of all neurons in prevs layer and spits out number between 0-1.

Also no one uses sigmoid anymore. 

They use $\text{ReLU}(a) = \max(0, a)$ (if any input is negative → make it 0, otherwise → let it be)

Why is this the case? The problem is that the sigmoid function becomes super flat at the extremes, when the weighted sum being passed in has a large magnitude (tldr; relu is better, trust me)

So wiggling the weights always gives useful feedback about how the network should change. 

This makes the training process much faster and more efficient, especially as there are more layers involved, which is where the “deep” in “deep learning” comes from.

What we’ll do is assign a weight to each of the connections between our neuron and the neurons from the first layer, these weights are just random numbers (for now)

**Each weight is an indication of how its neuron in the first layer is correlated with this new neuron in the second layer.**

If the neuron in the first layer is on, then a positive weight suggests that the neuron in the second layer should also be on, and a negative weight suggests that the neuron in the second layer should be off.

So to actually compute the value of this second-layer neuron, you take all the activations from the neurons in the first layer, and compute their weighted sum.

Something like this:

$w1a1 + w2a2 + w3a3 + w4a4 + ... + wn * an$

The expression `w1a1 + w2a2 + w3a3 + w4a4 + ... + wn * an` represents a **weighted sum**. 

Here are two common and more formal ways to write this using mathematical notation:

This is the most direct way to represent a sum of a sequence of terms.

#### Writing in summation form
$$
\sum_{i=1}^{n} w_i a_i
$$

* **$\sum$** is the Greek letter Sigma, which means "sum up."
* **$i = 1$** below the Sigma indicates that the sum starts with the index $i$ equal to 1.
* **$n$** above the Sigma indicates that the sum ends when the index $i$ reaches $n$.
* **$w_i a_i$** is the expression for each term in the sum.

This notation compactly says: "Sum the product of $w_i$ and $a_i$ for all integer values of $i$ from 1 to $n$."

#### Writing in linear algebra (standard form)
In linear algebra, this operation is known as the **dot product** of two vectors. This notation is extremely common in machine learning because it's how these operations are implemented efficiently in libraries like NumPy and PyTorch.

First, we define a **weight vector** \(W\) and an **activation (or input) vector** \(a\):

$$
W = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}, \quad a = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix}
$$

The weighted sum can then be written as the dot product $W \cdot a$, or more commonly in machine learning literature, as a matrix multiplication involving a transpose:

$$
W^T a
$$


* **$W^T$** is the **transpose** of the vector $W$, which turns the column vector into a row vector: $[w_1, w_2, \dots, w_n]$.
* When you multiply the row vector $W^T$ by the column vector $a$, you get the exact same weighted sum:

$$
W^T a = [w_1, w_2, \dots, w_n] \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} = w_1 a_1 + w_2 a_2 + \dots + w_n a_n (Same)
$$  


So maybe first layer picks up on the edges → Second layer picks up on the patterns → Final layer actually knows what number we have as input.

*So how does it learn? Whole point of training the model is that it can make the error as minimum as possible → How does it do that?*

### Defining the error
How does a computer define error? Especially in a machine learning model. 

There are multiple error functions but the basic premise we need to understand this is:
$$
{error} = \hat{y} - y
$$

where,
- $\hat{y}$ = what our neural network predicts.
- $y$ =  the actual number recognised from the image correctly

Our job is to make these the same, which means make error as minimum as possible.

How do you make a function as minimum as possible? Put on yo thinking caps its **calculus** time.

# Gradient Descent
One solution is to have random weights at first → calculate error → tweak weights and calculate error again (Expensive, takes too much time to do this for all neurons and layers)

Error is basically predicted_output (y') - actual_output (y) and 

- $y' = \text{output} = w_1 x_1 + w_2 x_2 + \dots + b$

So y' depends on weights and biases which means the loss is dependent on weights and biases (parameters of the curve to be precise)

This means our loss is a function of paramters → hence called **loss function.**
So we need to find configuration of these parameters to figure the minimum loss.

Plugging the coefficients obtained by minimising the loss function into curve equation gives us best description of the data.

How do we find these configurations? More imporantly how do we find minimum loss of a function? We derivate it (if you don't know this kys #2)

We can keep iteratively change the knob of weights/biases one at a time to see whether the resulting curve is a better fit (This is bad. Slow. It does work though)


**How can we be more intelligent?**

Calculus. Since these are curves, they follow the property of diffrentiability which allows us to get the optimised knob settings much better.  

Computer tells us which direction to change the knob and by how much.

We are essentially asking the computer to predict the future and estimate the effect of knob adjustments on the loss function without doing trial & error.


### Revisiting the setup
Neural Network
- Input: 784 numbers / pixels
- Output: 10 numbers
- Parameters: 8000 weights / biases

Initialise weights and bias randomly.
$$
\text{Loss} = (\text{Predicted} - \text{Actual})^2
$$

In expanded form

$$
C = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$


Why do we square it? We take average of this cost function and sometimes average might have positive and negative values which cancel out giving the illusion that cost function is minimum when it's actually not. Hence the square.

Assume we send an image of '9' =
Error is large when network doesn't seem to know what it's doing and small when it's close to 9.

Furthermore, the cost $C(w, b)$ becomes small, $C(w,b) ≈ 0$ precisely when $\hat{y} = y$ for all inputs x.

- Input : The number 9
- Output: Grey and White colors on multiple neurons (1-9)

*This is what the output looks like*

![Output2](https://i.ibb.co/p6qHZs3z/Screenshot-2025-06-11-at-11-47-08-PM.png)

*This is what we want*

![Output](https://i.ibb.co/qY5t0H7p/Screenshot-2025-06-11-at-11-47-12-PM.png)

What you gave me is utter trash (prediction), I need the output to be 9 (actual) ~ only cell 9 should be activated/fired.

So our training algorithm has done a good job if it can find weights and biases so that $C(w,b) ≈ 0.$ 

By contrast, it's not doing so well when $C(w,b)$ is large.

Aim of our training algorithm will be to minimize the cost $C(w,b)$ as a function of the weights and biases. 

Gradient descent helps neural networks learn by iteratively updating weights and biases to minimize the cost function:

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial C}{\partial \mathbf{W}}, \quad \mathbf{b} \leftarrow \mathbf{b} - \eta \frac{\partial C}{\partial \mathbf{b}}
$$
We'll talk more about this in a while. 


---

### Tuning Knobs to Minimize Loss

Imagine your model has one “knob” (parameter) $w$. Turning that knob up or down changes your model’s prediction—and thus the **loss**. Our goal is to find the setting of $w$ that makes the loss as small as possible.

Taking only one parameter : weights for now (let's say bias is fixed)

#### Plotting Loss vs. ($w$)
- **Horizontal axis** ($x$) = parameter $w$  
- **Vertical axis** ($y$) = loss $L(w)$

This plot is a curve $y=L(w)$. We want the **minimum point** on that curve.

![Plot](https://i.ibb.co/bgFrWDMB/Screenshot-2025-06-12-at-10-48-36-AM.png)
#### The Derivative

The **derivative** at a point tells us the instantaneous slope of the curve:

$$
\frac{dL}{dw}
= \lim_{\Delta w \to 0}
  \frac{L(w + \Delta w) - L(w)}{\Delta w}
$$

- If $\tfrac{dL}{dw} > 0$, the curve is rising as we move right (loss increases if $w$ increases).  
- If $\tfrac{dL}{dw} < 0$, the curve is falling as we move right (loss decreases if $w$ increases).

#### Ball on a Hill  
Picture a ball sitting on the curve. The derivative is “which way is downhill?”  
- If slope positive, the ball rolls left (decrease $w$).  
- If slope negative, the ball rolls right (increase $w$).

Gradient gives the direction of steepest ascend.
So to minimise the function → take steps in the opposite direction.

#### The Update Rule

We take a small step **opposite** the slope:

$$
w \leftarrow
w - \underbrace{\eta}_{\text{learning rate}}
\frac{dL}{dw}
$$

- $\eta$ controls the step size.

- We repeat: compute slope, step downhill, repeat… until the slope is (nearly) zero.

### Same but for two variables k1 and k2 (weights and bias in our case)
How do we measure change with respect to multi variables? Multi variate calculus (no shit)

Partial Derivatives : only focus on the term wrt which you're differentiating, treat everything else like a constant.

Now suppose we have **two knobs** $(k_1, k_2)$. 

We measure how the loss changes if we wiggle one parameter at a time:

$$
\frac{\partial L}{\partial k_1}
\quad \text{and} \quad
\frac{\partial L}{\partial k_2}
$$

- $\tfrac{\partial L}{\partial k_1}$ = rate of change if we nudge $k_1$ while keeping $k_2$ fixed.  
- $\tfrac{\partial L}{\partial k_2}$ = rate of change if we nudge $k_2$ while keeping $k_1$ fixed.

#### The Gradient Vector

Collect these partials into the **gradient**:

$$
\nabla L = 
\begin{bmatrix}
  \dfrac{\partial L}{\partial k_1} \\
  \dfrac{\partial L}{\partial k_2}
\end{bmatrix}
$$

- This points in the direction of **steepest ascent** on the loss surface.  
- To go **downhill**, we move in the **opposite** direction.

#### Updating again

$$
\begin{bmatrix} k_1 \\ k_2 \end{bmatrix}
\leftarrow
\begin{bmatrix} k_1 \\ k_2 \end{bmatrix}
- \eta\,
\nabla L
=
\begin{bmatrix}
  k_1 \\
  k_2
\end{bmatrix}
- \eta\,
\begin{bmatrix}
  \tfrac{\partial L}{\partial k_1} \\[6pt]
  \tfrac{\partial L}{\partial k_2}
\end{bmatrix}
$$

And in **$n$-dimensional** parameter space $\mathbf{k}=(k_1,\dots,k_n)$:


$$
\mathbf{k} \leftarrow
\mathbf{k}
- \eta\, \nabla L(\mathbf{k}),
\quad
\nabla L(\mathbf{k}) =
\begin{bmatrix}
  \dfrac{\partial L}{\partial k_1} \\
  \vdots \\
  \dfrac{\partial L}{\partial k_n}
\end{bmatrix}
$$

Continuing with our *ball rolling* example:
- **Draw the hill**: Visualize loss as a surface over your parameters.  
- **Find the slope**: Compute (partial) derivatives—these tell you “uphill” vs. “downhill.”  
- **Step downhill**: Update each parameter by subtracting a fraction (learning rate) of its derivative.  
- **Iterate**: Keep repeating until you reach a flat spot (derivative ≈ 0), i.e. a **minimum**.

> Gradient descent is simply “roll the ball downhill, one small step at a time.” 

We can figure out the slope directly & the direction we need to adjust our weight without going all the way back to our neural network and calculating. Pretty cool.

**Question:** How do we change these weights and biases so it makes it better? 
**Answer:** Shift inputs to the left if slope is positive, shift inputs to right is slope is negative. 

Input in this case are the parameters of the cost function (weights and bias)

If you keep repeating → Approach local minimum of the function (ball rolling down a hill : lowest point)

![Bowl](https://i.ibb.co/Kp2Cdd6s/bowl.png)

**We need to reach the bottom of the bowl.**

- Randomly choose starting point for a ball (imaginary) then simulate the motion of ball as it's rolled down.
- We can do this by computing derivatives of cost function wrt parameters (weights and bias) : those derivatives would tell us everything about the local shape of the valley, therefore how our ball should roll.
- What law or laws of motion could we pick that would make it so the ball always rolled to the bottom of the valley?
where η is a small, positive parameter (known as the learning rate).

The learning rate $\eta$ is a parameter in neural networks that controls the size of the steps taken during gradient descent to minimize the loss function. It determines how much the weights and biases are updated in each iteration based on the computed gradients 

Use case? Controls Step Size: The learning rate determines how large the updates to the weights and biases are. A larger learning rate results in bigger steps, while a smaller learning rate results in smaller steps.

A small learning rate ensures stable convergence, while a large learning rate can cause the model to overshoot the minimum.

Gradient Descent can be viewed as a way of taking small steps in the direction which does the most to immediately decrease Cost/Loss function.
> Lets say you're at a hill and you want to go from one village to the other. And all you can see are paths.
We can stand at the edge of the terrace and look for the steepest route to the terrace below.
That's also the shortest path down to the next piece of level ground, if we repeat the process from terrace to terrace we will eventually reach the village.
In doing so we will have taken the path of steepest descent (might not be a straight line, can be zig zag)


What we did was evaluate the slope/gradeint of the hillside as we looked in differnt directions while standing at the edge of a terrace and then took the steepest path down each time.

**This is gradient descent. An algorithm use to compute the minimum of the cost function.**

Which direction decreases C(weights, bias) most quickly?

Gradient/Derivative of function gives you direction of steepest increase (which dirn should you step to increase function most quickly)

Taking negative of gradient gives you direction of step which decreases function most quickly : **WHAT WE WANT**

This can be visuallied using the ball rolling diagram, draw a few slopes on the curve, 
- if you're on the left : you need to move right to reach the bottom
- if you're on the right : you need to move left to reach the bottom

*Algorithm:*
- Compute G(Cost Function) (G is gradient)
- Take small step in the opposite direction of gradient : -Gradient(C) direction.
- Repeat.

```python
def gradient_descent_pure_python(learning_rate=0.1, num_iterations=100):
    """
    Minimizes the function f(x) = x^2 using gradient descent.
    The derivative f'(x) = 2x.
    """
    weight = 10.0  # Initial guess for x

    print(f"Starting x: {x}, f(x): {weight**2:.4f}")

    for i in range(num_iterations):
        gradient = 2 * weight # x^2 derivative is 2x if you don't know this kys #3

        # Update weight by moving in the negative direction of the gradient
        weight = weight - learning_rate * gradient

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"Iteration {i+1}: x = {weight:.4f}, f(x) = {weight**2:.4f}, "
                f"Gradient = {gradient:.4f}"
            )

    print(f"\nOptimization finished. Minimum found at x = {weight:.4f}")
    print(f"Minimum function value f(x) = {weight**2:.4f}")
```

**Output**
```text
smol@samit ~  > python3 a.py
Starting x: 10.0, f(x): 100.0000
Iteration 1: x = 8.0000, f(x) = 64.0000, Gradient = 20.0000
Iteration 10: x = 1.0737, f(x) = 1.1529, Gradient = 2.6844
Iteration 20: x = 0.1153, f(x) = 0.0133, Gradient = 0.2882
Iteration 30: x = 0.0124, f(x) = 0.0002, Gradient = 0.0309
Iteration 40: x = 0.0013, f(x) = 0.0000, Gradient = 0.0033
Iteration 50: x = 0.0001, f(x) = 0.0000, Gradient = 0.0004
Iteration 60: x = 0.0000, f(x) = 0.0000, Gradient = 0.0000
Iteration 70: x = 0.0000, f(x) = 0.0000, Gradient = 0.0000
Iteration 80: x = 0.0000, f(x) = 0.0000, Gradient = 0.0000
Iteration 90: x = 0.0000, f(x) = 0.0000, Gradient = 0.0000
Iteration 100: x = 0.0000, f(x) = 0.0000, Gradient = 0.0000
Optimization finished. Minimum found at x = 0.0000
Minimum function value f(x) = 0.0000
```

As you can clearly tell, the minimum of $f(x) = x^2$ is at x = 0, f(x) = 0 and that is what the output represents also.

Here we only optimised weights because that was the only parameter, in our original equation of line we have 2 parameters : weights and biases, need to optimise both.

How do you diffrentiate a function and find it's minimum wrt 2 parameters? **Partial derivatives BAM.** Hell yeah. More on this later.

> Here is that update rule in concise mathematical form.

$$
\frac{\partial C}{\partial \mathbf{W}^{(l)}} = \delta^{(l)}(a^{(l-1)})^T,
\quad
\frac{\partial C}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}
$$

$$
\text{Compute gradients:} \quad
\frac{\partial C}{\partial W}, \quad
\frac{\partial C}{\partial b}
$$

$$
\text{Gradient‐descent update:} \quad
W \leftarrow W - \eta\,\frac{\partial C}{\partial W},
\qquad
b \leftarrow b - \eta\,\frac{\partial C}{\partial b}
$$

Usually we use learning rate = 0.1 or 0.01 (So we take small steps and not move too fast (we might miss the bottom of the bowl and directly jump to the other side))

Each component of the negative gradient vector tells us two things-:
- The sign, of course, tells us whether the corresponding component of the input vector should be nudged up or down. 
- The relative magnitudes of the components in this gradient tells us which of those changes matters more.


**Backpropagation (efficiently calculating the gradients (derivatives) that Gradient Descent needs to update the weights of a neural network)**

It's just a fancy word for applying gradient descent multiple times efficiently.

## Backpropagation
How does it calculate error what do you mean? Why does this work?
What is learning? Which weights and bias minimise a certain cost function?

Algorithm for computing the gradient:
Don't be scared by these equations they don't mean shit for now, but it'll make sense later. If you know enough math to understand this in the first go,,,, why are you still reading this?

$$
\delta^{L} = (\hat{\mathbf{y}} - \mathbf{y}) \odot \sigma'(z^{L})
$$

$$
\delta^{l} = ((\mathbf{W}^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})
$$

$$
\frac{\partial C}{\partial \mathbf{W}^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

$$
\frac{\partial C}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}
$$

Fuck the 4 equations.

Focus on one example:


Input 2, lets say this is our activations based on random assignments of weights and biases.

**Activations**

<pre>
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
</pre>

This is utter trash like we talked about.
We want it to be like:

**Activations**
<pre>
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
</pre>
We want that third value to get nudged up while all others get nudged down. 

The sizes of these nudges should be proportional to how far away each current value is from target value.

What I mean by this:
- Increase in number 2 activation is more important than decrease in number 8 neuron (which is alr 0.2 close to 0)

So let's take 2 neuron
0.2 = $\sigma\left(\sum_n w_n a_n + b\right)$

What can we change:
- Increase bias.
- Incerase weights.
- Change activations from previous layer.

It makes sense to increase weights in proportion to activation function of the neuron. 

If neuron already has high activation function (high probability of being that number), you want to increase weight of that neuron (more importance to that neuron)

We need to increase weights of neurons that contribute to neuron 2 in a more powerful way. That makes sense.

Similarily we change the activation function of previous layers in proportion to weights.

Biggest increases in weights — the biggest strengthening of connections—happens between neurons that are the most active and the ones which we wish to become more active.

In practice it's really time consuming for the computer to calculate all gradients and upgrade weights for ALL training data. Hence we use schoastic(randomized) gradient descent where we:
- Shuffle Training Data.
- Randomly subdivide into mini batches.
- Go through all mini batches and make these adjustments until you converge for your local loss function.

The bias associated with the digit-2 neuron should be increased, because that will cause the activation of the digit-2 neuron to increase. All the other output activations should be decreased, which requires decreasing their associated biases.

**Backpropagation Error Flow**

![Backpropagation Error Flow](https://i.ibb.co/WWKZ3fdz/generated-image.png)


## Time 4 Math

Feel free to skip this part if you are not going to make it.

![Backprop](https://i.ibb.co/fT6q64t/backprop.png)
![Chain Rule](https://i.ibb.co/SX9YKZ0M/chain-rule.png)

Backpropagation is an algorithm for calculating the gradient of the cost function of a network.

**Entries of gradient vector are partial derivatives of the cost function with respect to weights and biases.**

Once you know how sensitive the cost function is to activations in this second-to-last layer, you can just repeat the process for all the weights and biases feeding into that layer.

It’s remarkable that at the heart of every modern neural network lies nothing more exotic than a handful of basic math tools—matrix multiplication, derivatives, and the chain rule.

- Matrix multiplication elegantly encodes how signals flow from one layer of neurons to the next: each weight in the matrix acts like a tiny valve, amplifying or dampening its inputs in one sweeping operation.
- By defining a cost function that measures “how wrong” our network is, we turn the fuzzy notion of “learning” into the concrete problem of finding a minimum.  Derivatives and gradients become our compass, pointing us downhill through a vast, multidimensional landscape of error.
- And the chain rule—often dismissed in calculus class as “just another formula”—is the secret ingredient that unravels this entire web of dependencies.  It lets us trace exactly how a small tweak to any weight or bias will ripple through every layer, so we can update each parameter in just the right direction.


When you line up these simple ideas in the right order, you get back-propagation—and with it, the ability to build systems that can recognize images, translate languages, and more.

So what do these scary equations mean anyway?


Consider a simple neural network → 4 neurons connected via 3 links hence 4 activations, 3 weights and 3 biases (for each neuron except first)
![Simple NN](https://i.ibb.co/7xLb4ZSX/Screenshot-2025-06-12-at-4-42-26-AM.png)

**Goal** 
- How sensitive the cost function is to these variables?
- Which knobs to turn in order to make the cost function minimal?

We're only going to focus on the activation of last 2 neurons.

The **weight**, **previous activation**, and **bias** are combined to compute the pre-activation:

$$
z = \sum_i w_i a_i + b
$$

This in turn is passed through the **sigmoid activation** function:

$$
a = \sigma(z)
$$

Finally, the activation $a$ — along with the true label $y$ — is used to compute the **cost** (e.g., using mean squared error or cross-entropy).

Note that $a^{(L-1)}$ is itself computed from:

- its own weights and biases,
- and activations from the previous layer $a^{(L-2)}$,

and this pattern **repeats backward** through the network all the way to the input.

> Each layer’s output becomes the next layer’s input, all the way to the final cost.

![Image1](https://i.ibb.co/cSMJwZ0g/Screenshot-2025-06-12-at-12-08-55-AM.png)
![Image2](https://i.ibb.co/8k9PD2S/Screenshot-2025-06-12-at-12-08-48-AM.png)
![Image3](https://i.ibb.co/GQ5rMvk8/Screenshot-2025-06-12-at-12-08-00-AM.png)

#### Sensitivity of the Cost to a Single Weight
Our **goal** is to measure how a tiny change (“nudge”) in one weight in the last layer, $w^{(L)}$, affects the loss $C$.  
In calculus terms we want the partial derivative:

$$
\frac{\partial C}{\partial w^{(L)}}
$$

When you see $\partial w^{(L)}$, think “a small nudge to $w^{(L)}$”;  
when you see $\partial C$, think “the resulting nudge to the cost.”  
Their ratio $\tfrac{\partial C}{\partial w^{(L)}}$ is the **instantaneous rate of change** of the loss with respect to that weight.

---
#### 1. Chain Rule Decomposition

Because $C$ depends on $w^{(L)}$ only through the pre‐activation $z^{(L)}$ and the activation $a^{(L)}$, the chain rule gives

$$
\frac{\partial C}{\partial w^{(L)}}
=
\frac{\partial C}{\partial a^{(L)}}
\;\times\;
\frac{\partial a^{(L)}}{\partial z^{(L)}}
\;\times\;
\frac{\partial z^{(L)}}{\partial w^{(L)}}.
$$

A neat trick to verify if this is right is to cancel all denom and next numerator from the start, you'll get the original function. 

---

#### 2. Individual Derivatives

1. **Pre‐activation**  
   $$
   z^{(L)} = w^{(L)}\,a^{(L-1)} + b^{(L)}
   \quad\Longrightarrow\quad
   \frac{\partial z^{(L)}}{\partial w^{(L)}} = a^{(L-1)}.
   $$

2. **Activation (sigmoid)**  
   $$
   a^{(L)} = \sigma\!\bigl(z^{(L)}\bigr)
   \quad\Longrightarrow\quad
   \frac{\partial a^{(L)}}{\partial z^{(L)}} 
   = \sigma'\!\bigl(z^{(L)}\bigr)
   = \sigma\!\bigl(z^{(L)}\bigr)\bigl[1 - \sigma\!\bigl(z^{(L)}\bigr)\bigr].
   $$

3. **Loss for one example** (Mean Squared Error)  
   $$
   C_0 = \bigl(a^{(L)} - y\bigr)^2
   \quad\Longrightarrow\quad
   \frac{\partial C_0}{\partial a^{(L)}} = 2\,\bigl(a^{(L)} - y\bigr).
   $$

---

#### 3. Combined Gradient

Putting it all together,

$$
\frac{\partial C_0}{\partial w^{(L)}}
= \underbrace{2\bigl(a^{(L)} - y\bigr)}_{\frac{\partial C_0}{\partial a^{(L)}}}
\;\times\;
\underbrace{\sigma'\!\bigl(z^{(L)}\bigr)}_{\frac{\partial a^{(L)}}{\partial z^{(L)}}}
\;\times\;
\underbrace{a^{(L-1)}}_{\frac{\partial z^{(L)}}{\partial w^{(L)}}}
\;=\;
2\,\bigl(a^{(L)} - y\bigr)\,\sigma'\!\bigl(z^{(L)}\bigr)\,a^{(L-1)}.
$$

This tells us **exactly** how a small change in $w^{(L)}$ for one training example will change the loss.

---

#### 4. Averaging Over the Dataset

For $N$ training examples, we average these per–example gradients to get the gradient of the full cost $C$:

$$
\nabla C
= \frac{1}{N}\sum_{i=1}^{N}
\frac{\partial C^{(i)}}{\partial w^{(L)}}.
$$

---

#### 5. Update Rule (Gradient Descent)

Finally, we update the weight by stepping **opposite** to the gradient:

$$
w^{(L)} \;\leftarrow\;
w^{(L)} \;-\;\eta\,
\frac{\partial C}{\partial w^{(L)}},
$$
where $\eta$ is the learning rate.  
Likewise for the bias:
$$
b^{(L)} \;\leftarrow\;
b^{(L)} \;-\;\eta\,
\frac{\partial C}{\partial b^{(L)}}.
$$

### Stochastic Gradient Descent:

When your dataset is huge, computing the exact gradient  
$\displaystyle \nabla C = \frac{1}{N}\sum_{i=1}^N \nabla C^{(i)}$  
on all $N$ examples each update can be very slow.  
**Stochastic Gradient Descent (SGD)** speeds things up by estimating the true gradient with a small random **mini-batch** of $m\ll N$ examples:

$$
\nabla C_{\text{batch}}
= \frac{1}{m}\sum_{i\in\text{batch}}\nabla C^{(i)}.
$$

1.  **Sample** a mini-batch of $m$ training examples at random.  
2.  **Compute** the average gradient $\nabla C_{\text{batch}}$.  
3.  **Update** parameters:  
    $$
      \theta \;\leftarrow\; \theta \;-\;\eta\,\nabla C_{\text{batch}},
    $$  
    where $\eta$ is the learning rate.  

### Why It Works

- **Cost per update** falls from $O(N)$ to $O(m)$.  
- **Unbiased estimate**: $\mathbb{E}[\nabla C_{\text{batch}}] = \nabla C$.  

> Stochastic GD is like political polling:  
> it’s far cheaper to poll 1,000 voters (a mini-batch)  
> than to canvass an entire electorate (the full dataset),  
> yet the poll’s average still points in the right direction.

---

#### Partial Derivative w.r.t. Bias

For a neuron with pre-activation  
$$
z = W\,x \;+\; b,
$$  
the bias $b$ enters **linearly**, so

$$
\frac{\partial z}{\partial b} \;=\; 1
\;\;\Longrightarrow\;\;
\frac{\partial C}{\partial b}
= \frac{\partial C}{\partial z}\,\frac{\partial z}{\partial b}
= \frac{\partial C}{\partial z}.
$$

---

### The Full Gradient Vector

We’ve already seen how to compute two entries of the gradient vector for the **last** layer:
1.  **Weight gradient**  
    $$
    \frac{\partial C}{\partial W^{(L)}}
    = \delta^{(L)}\,(a^{(L-1)})^{\top}
    $$

2.  **Bias gradient**  
    $$
    \frac{\partial C}{\partial b^{(L)}}
    = \delta^{(L)}
    $$
    
    where $\delta^{(L)} = \tfrac{\partial C}{\partial z^{(L)}}$.
By **propagating** these errors backward through the network—multiplying along the chain of dependencies—we obtain every partial derivative $\tfrac{\partial C}{\partial w}$ and $\tfrac{\partial C}{\partial b}$ for **every** layer. Stacking them gives the **gradient vector**:

$$
\nabla C \;=\;
\begin{bmatrix}
\vdots \\[4pt]
\displaystyle \frac{\partial C}{\partial w_1} \\[4pt]
\displaystyle \frac{\partial C}{\partial b_1} \\[4pt]
\vdots
\end{bmatrix}.
$$

Once we have $\nabla C$, we perform **gradient descent** (or SGD)  
to step **downhill** and minimize the loss:

$$
\theta \;\leftarrow\; \theta \;-\;\eta\,\nabla C.
$$

*Components of gradient vector guide the adjustments we need to make.*

If for a particular configuration:
- The partial derivative of loss with respect to k1 is positive means increase k1 implies increase in loss
- Hence we need to decrease (opposite sign of gradient) k1 to decrease loss.

How do we access the derivative of the loss function in the first place?
**Chain Rule** → tells you how to compute derivative of combination of functions.

The **chain rule** tells us how to compute the derivative of a function that is itself the input to another function.  In symbols, if
$$
h(x) = f\bigl(g(x)\bigr),
$$
then
$$
\frac{d}{dx}h(x)
= \frac{d}{dx}\bigl[f\bigl(g(x)\bigr)\bigr]
= f'\bigl(g(x)\bigr)\;\cdot\;g'(x).
$$


*Components of gradient vector guide the adjustments we need to make.*

If for a particular configuration:
- The partial derivative of loss with respect to k1 is positive means increase k1 implies increase in loss
- Hence we need to decrease (opposite sign of gradient) k1 to decrease loss.

How do we access the derivative of the loss function in the first place?
**Chain Rule** → tells you how to compute derivative of combination of functions.

The **chain rule** tells us how to compute the derivative of a function that is itself the input to another function.  In symbols, if
$$
h(x) = f\bigl(g(x)\bigr)
$$

then

$$
\frac{d}{dx}h(x)
= \frac{d}{dx}\bigl[f\bigl(g(x)\bigr)\bigr]
= f'\bigl(g(x)\bigr) \cdot g'(x)
$$
### Intuition: Two “Machines” in Series

1. **Machine 1**  
   - Input: $x$
   - Output: $g(x)$
   - Local slope: $g'(x)$

2. **Machine 2**  
   - Input: $u = g(x)$  
   - Output: $f(u)$
   - Local slope: $f'(u)$

When you feed $x$ into the first machine, it spits out $g(x)$.  
That becomes the input to the second machine, which outputs $f\bigl(g(x)\bigr)$.

#### Small Nudge Argument

1.  **Nudge** $x$ by a tiny amount $\Delta x$.  
2.  **Machine 1** outputs  
    $$
    g(x + \Delta x) \approx g(x) + g'(x)\,\Delta x
    $$  
    So the **change** in its output is  
    $$
    \Delta u \approx g'(x)\,\Delta x,
    $$  
    where $u = g(x)$.

3.  **Machine 2** then sees its input change by $\Delta u$, so its output changes by  
    $$
    f(u + \Delta u)
    \approx
    f(u)
    + f'(u)\,\Delta u
    $$

4.  **Combined change** in the final output:  
    $$
    \Delta h
    = f'\bigl(g(x)\bigr)\,\Delta u
    \approx
    f'\bigl(g(x)\bigr)\,\bigl[g'(x)\,\Delta x\bigr]
    = \bigl[f'\bigl(g(x)\bigr)\,g'(x)\bigr]\,\Delta x
    $$

5.  Dividing by $\Delta x$ and taking the limit $\Delta x \to 0$ gives the chain rule:  
    $$
    \frac{d}{dx}f\bigl(g(x)\bigr)
    = f'\bigl(g(x)\bigr) \cdot g'(x)
    $$
### Why It Matters in Neural Networks

In backpropagation we have many such compositions—each layer’s output is the input to the next layer’s activation function.  


The chain rule lets us **multiply** all the local derivatives together to find how the final loss changes with respect to any parameter deep in the network.

To calculate the derivative of the cost function with respect to weights and bias for a simple sigmoid-activated neuron:

We want to find the gradient of our Cost Function, which is usually the Mean Squared Error (MSE), with respect to the weights $w$ and bias $b$.

**1. Define the building blocks:**

- **Input Features:** $X$ (a vector or matrix of input values, $x_1, x_2, \dots, x_n$)
- **Weights:** $W$ (a vector of weights, $w_1, w_2, \dots, w_n$)
- **Bias:** $b$ (a single scalar value)
- **Activation Function:** Sigmoid function,  
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}
  $$
  - The derivative of the sigmoid function is  
    $$
    \sigma'(z) = \sigma(z)(1 - \sigma(z))
    $$
- **Actual Labels:** $Y$ (the true output values)

**2. The Forward Pass**

First, let's understand how we get from inputs to our error.

*   **Linear Combination (Weighted Sum + Bias):** This is the input to the activation function.
    $$ z = X \cdot W + b $$
    In code, this is `z = np.dot(features, weights) + bias`. This is a sum of $x_i w_i$ for all input features, plus the bias.

*   **Predicted Output ($\hat{y})$:** Apply the sigmoid activation function to \(z\).
    $$ \hat{y} = \sigma(z) $$
    In code, this is `predictions = sigmoid(z)`. This is your `predicted_output`.

*   **Cost Function (Mean Squared Error - MSE):** We measure how "wrong" our prediction is. For a single training example, the error is \(( \hat{y} - y )^2\). For multiple samples, we typically take the mean.
    $$ C = \text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 $$
    In code, this is `mse = np.mean((predictions - labels) ** 2)`. This is your `Cost`.

The process of calculating $z$, $\hat{y}$, and $C$ is called the **forward pass**.

$$
z^{(l)} = \mathbf{W}^{(l)} a^{(l-1)} + \mathbf{b}^{(l)}, 
\quad 
a^{(l)} = \sigma(z^{(l)})
$$



```python
# forward pass
z = np.dot(features, weights) + bias # sum(wx + b)
predictions = sigmoid(z) # apply sigmoid function to this (this is y_hat)
mse = np.mean((predictions - labels) ** 2) # take mean square error (this is C)
mse_values.append(round(mse, 4)) # add it to mean square error list
```

**3. The Backward Pass**

Now, we use the **Chain Rule** to find how the Cost $C$ changes with respect to each weight $w_j$ and the bias $b$.  
The Chain Rule helps us break down complex derivatives into a product of simpler ones.

We want to find $\frac{\partial C}{\partial w_j}$ and $\frac{\partial C}{\partial b}$.

Let's trace the dependencies backward from \(C\):
$$ C \leftarrow \hat{y} \leftarrow z \leftarrow (W, b) $$

**First, the derivative of the Cost with respect to the Predicted Output (\(\hat{y}\)):**
$$ \frac{\partial C}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} \left( \frac{1}{N} \sum (\hat{y}_i - y_i)^2 \right) = \frac{1}{N} \sum 2(\hat{y}_i - y_i) $$
In code, `error = predictions - labels`. So, $(\frac{\partial C}{\partial \hat{y}}$ is proportional to `2 * error`. When we take the mean over $N$ samples, it becomes $(\frac{2}{N} \times \text{sum of errors}$

**Second, the derivative of the Predicted Output ($\hat{y}$) with respect to $z$:**
This is the derivative of the sigmoid function.
$$ \frac{\partial \hat{y}}{\partial z} = \frac{\partial}{\partial z} \sigma(z) = \sigma(z)(1 - \sigma(z)) $$
In code, this is `sigmoid_derivative = predictions * (1 - predictions)`.

**Combining the first two steps to get \(\frac{\partial C}{\partial z}\):**
Using the chain rule: $\frac{\partial C}{\partial z} = \frac{\partial C}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$
$$ \frac{\partial C}{\partial z} = \left( \frac{2}{N} \sum (\hat{y}_i - y_i) \right) \cdot \sigma(z_i)(1 - \sigma(z_i)) $$
In code, `delta = error * sigmoid_derivative` effectively calculates `(predictions - labels) * sigmoid_derivative` for each sample. The `(2/len(labels))` factor will be applied later when we average over samples. So, `delta` represents $(\hat{y} - y) \cdot \sigma'(z)$  for each sample.

**Now, for the final gradients:**

**Derivative of Cost with respect to a Weight $w_j$:**
Using the chain rule: $\frac{\partial C}{\partial w_j} = \frac{\partial C}{\partial z} \cdot \frac{\partial z}{\partial w_j}$
We know that $(z = X \cdot W + b = x_1 w_1 + x_2 w_2 + \dots + x_j w_j + \dots + x_n w_n + b$

So, $\frac{\partial z}{\partial w_j} = x_j$
Therefore:
$$ \frac{\partial C}{\partial w_j} = \left( \frac{2}{N} \sum (\hat{y}_i - y_i) \cdot \sigma(z_i)(1 - \sigma(z_i)) \right) \cdot x_{ij} $$
(where $(x_{ij}$ is the $j$-th feature for the $i$-th sample).
In your code:
`partial_derivative_error_wrt_weights = (2 / len(labels)) * np.dot(features.T, delta)`
This `np.dot(features.T, delta)` correctly performs the sum $ \sum \text{delta}_i \cdot x_{ij} $ for all features, which is exactly what the formula requires.

**Derivative of Cost with respect to the Bias \(b\):**

Using the chain rule $\frac{\partial C}{\partial b} = \frac{\partial C}{\partial z} \cdot \frac{\partial z}{\partial b}$
We know $z = X \cdot W + b$

So,$\frac{\partial z}{\partial b} = 1$
Therefore

$$ \frac{\partial C}{\partial b} = \left( \frac{2}{N} \sum (\hat{y}_i - y_i) \cdot \sigma(z_i)(1 - \sigma(z_i)) \right) \cdot 1 $$

$$ \frac{\partial C}{\partial b} = \frac{2}{N} \sum (\hat{y}_i - y_i) \cdot \sigma(z_i)(1 - \sigma(z_i)) $$

In your code
`partial_derivative_error_wrt_bias = (2 / len(labels)) * np.mean(delta)`
This `np.mean(delta)` effectively calculates $\frac{1}{N} \sum \text{delta}_i$ for all samples, which is what the formula requires.


Calculate error at the output

$$
\delta^{(L)} = (\hat{\mathbf{y}} - \mathbf{y}) \odot \sigma'(z^{(L)})
$$

Propagate errors backward:

Code for the **Backward Pass**:

```python
error = predictions - labels # This is (y' - y) for each sample
sigmoid_derivative = predictions * (1 - predictions) # This is sigma'(z) for each sample
delta = error * sigmoid_derivative # This is (y' - y) * sigma'(z) for each sample

# Calculate the partial derivatives (gradients) of the Cost with respect to weights and bias
# partial_derivative_error_wrt_weights corresponds to dC/dW
# partial_derivative_error_wrt_bias corresponds to dC/db
partial_derivative_error_wrt_weights = (2 / len(labels)) * np.dot(features.T, delta)  # using chain rule
partial_derivative_error_wrt_bias = (2 / len(labels)) * np.mean(delta) # using chain rule

# update weights and bias using the calculated gradients (Gradient Descent step)
weights -= learning_rate * partial_derivative_error_wrt_weights
bias -=  learning_rate * partial_derivative_error_wrt_bias
gradients = [] # assume this is a matrix which just stores all results of dC/dw and dC/db
gradients.append(weights, bias) # This line looks a bit off in the original, `gradients` should likely be a list of gradient vectors, not the updated weights/bias themselves.
```

backward because as the gradients are computed by propagating sensitivity backward from the output layer to the input layer.


Once these gradients are found,
*We're going to tweak these parameters based on the loss function gradient.*

Slightly tweak the knobs oppositve to the gradient.
**$ updatedWeight = weight - learningRate * [partial(Cost) / partial(weight)]$**

After each adjustment we need to redo forward and backward passes since the loss functions have changed.

Performing this loop of:
- Forward Pass
- Backward Pass
- Nudge knobs
- Repeat 1-3

**is the essence of training modern machine system.**

As long as your fancy model can be decomposed as a sequence of differentiable functions, you can apply backprop to optimise the parameters.


### Why is it fast?
In what sense is backpropagation a fast algorithm? To answer this question, let's consider another approach to computing the gradient. 

Imagine it's the early days of neural networks research. Maybe it's the 1950s or 1960s, and you're the first person in the world to think of using gradient descent to learn! But to make the idea work you need a way of computing the gradient of the cost function.

You think back to your knowledge of calculus, and decide to see if you can use the chain rule to compute the gradient. But after playing around a bit, the algebra looks complicated, and you get discouraged. 

So you try to find another approach. You decide to regard the cost as a function of the weights C=C(w) alone (we'll get back to the biases in a moment). You number the weights w1,w2,…, and want to compute ∂C/∂wj for some particular weight wj. An obvious way of doing that is to use the approximation where ϵ>0 is a small positive number, and ej is the unit vector in the jth direction. In other words, we can estimate ∂C/∂wj by computing the cost C for two slightly different values of wj.

### 1. Bruteforce Gradient Descent

To estimate 
$\tfrac{\partial C}{\partial w_j} $
by finite differences, you’d do

$$
\frac{\partial C}{\partial w_j}
\approx
\frac{C\bigl(w + \varepsilon\,e_j\bigr)\;-\;C(w)}{\varepsilon},
$$

where $e_j$ is the unit vector in direction $j$.
If you have \(N\) weights, that requires $N+1$ full forward‐passes through the network per training example!

### 2. Backpropagation via the Chain Rule

Backprop avoids that explosion by **computing all** partial derivatives  
$\tfrac{\partial C}{\partial w_j}$ at once with:

1. **One forward pass** (compute all activations).  
2. **One backward pass** (propagate “error signals” and multiply local derivatives).

#### 2.1 Small‐Change Analysis

A small change $\Delta w_{l,j,k}$ in weight $w_{l,j,k}$
causes a small change in the neuron’s activation:

$$
\Delta a_{l,j}
\approx
\frac{\partial a_{l,j}}{\partial w_{l,j,k}}\,
\Delta w_{l,j,k}.
$$

That, in turn, perturbs the next layer’s activation:

$$
\Delta a_{l+1,q}
\approx
\frac{\partial a_{l+1,q}}{\partial a_{l,j}}\,
\Delta a_{l,j}.
$$

… and so on, all the way to the cost \(C\):

$$
\Delta C
\approx
\frac{\partial C}{\partial w_{l,j,k}}\,
\Delta w_{l,j,k}.
$$

#### 2.2 Chain Rule for a Single Weight

By following **every path** from $w_{l,j,k}$ to the final cost,  
and **multiplying** the local partials along each edge,  
then **summing** over all such paths, you get:

$$
\frac{\partial C}{\partial w_{l,j,k}}
= \sum_{\text{paths }p}
  \prod_{\substack{\text{edges }(u\to v)\\\in p}}
    \frac{\partial v}{\partial u}.
$$

Backpropagation is simply a dynamic‐programming implementation of this sum‐of‐products **in linear time** $O(N)$, rather than $O(N^2)$ or worse.

---
- **Finite‐difference**: $O(N)$ forward passes for $N$ weights → **slow**.  
- **Backpropagation**: 1 forward pass + 1 backward pass → **fast**.  
- Uses the **chain rule** to share and reuse intermediate derivatives,  
  computing the entire gradient vector $\nabla C$ efficiently.

*Backpropagation algorithm as providing a way of computing the sum over the rate factor for all these paths. Or, to put it slightly differently, the backpropagation algorithm is a clever way of keeping track of small perturbations to the weights (and biases) as they propagate through the network, reach the output, and then affect the cost.*


### CODING TIME CODING TIME CODING TIME
Although the math is quite brilliant & also pretty simple to understand once you really sit down and think about it, that shit is still 4 losers we're cs students not math majors...

We have gathered enough theory to build our own program that correctly identifies number from handwritten images. This is the hello world of neural networks.

### About the dataset
The MNIST data comes in two parts. The first part contains 60,000 images to be used as training data. These images are scanned handwriting samples from 250 people, half of whom were US Census Bureau employees, and half of whom were high school students. 

All images are like this:

![MNIST](https://i.ibb.co/n8jq9KQd/Screenshot-2025-06-12-at-10-02-06-AM.png)

The images are greyscale and 28 by 28 pixels in size. The second part of the MNIST data set is 10,000 images to be used as test data. 

We'll use the test data to evaluate how well our neural network has learned to recognize digits. To make this a good test of performance, the test data was taken from a different set of 250 people than the original training data. This helps give us confidence that our system can recognize digits from people whose writing it didn't see during training.

What we want it to predict *(Ideal Output)*

![Output](https://i.ibb.co/RkmDmCky/mnist-Hotcoded.png)

### MNIST from Scratch (Using Numpy)
```python
import numpy as np
"""
if we want to create a Network object with 2 neurons in the first layer, 3 neurons in the second layer, and 1 neuron in the final layer:
    net = Network([2, 3, 1])

net.weights[1] is a Numpy matrix storing the weights connecting the second and third layers of neurons (Python counts from 0)
"""
class Network:
    def __init__(self, sizes) -> None:
        ''' This creates a network with layers and indicates how many neurons in each layer (sizes)'''
        self.num_layers = len(sizes)
        self.sizes = sizes
        # randomly initialise the weights and bias
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z) -> float:
        ''' Activation function to convert probability in range 0 -> 1 '''
        return 1.0 / (1.0 + np.exp(-z))
        # relu : return max(0, z) 

    def sigmoid_derivative(self, z) -> float:
        ''' Derivative of sigmoid : we need this for calculating partial derivative of cost function '''
        return self.sigmoid(z) * (1 - self.sigmoid(z)) 

    def forward(self, activation) -> float:
        ''' Return output of neural network '''
        for b, w in zip(self.bias, self.weights):
            activation = self.sigmoid(np.dot(w, activation) + b) # sigmoid(wx + b) (wx = dot product if you think about how matrix works)
        return activation

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.  
        The "training_data" is a list of tuples "(x, y)" representing the training inputs and the desired outputs.
        If "test_data" is provided then the network will be evaluated against the test data after each epoch, and partial progress printed out.
        epochs is just a fancy name for number of iterations.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data) # otherwise
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            # for each mini_batch we apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            if test_data:
                accuracy = self.evaluate(test_data)
                accuracy_percentage = (accuracy / n_test) * 100
                print( f"Epoch {epoch}: {accuracy} / {n_test}") 
            else:
                print(f"Epoch {epoch} complete") 

        print(f"Accuracy : {accuracy_percentage:.2f}%")
            
    def update_mini_batch(self, mini_batch, learning_rate):
        ''' Update weights and biases by applying Gradient Descent to single mini batch (list of tuples (x,y) representing training inputs and their desired outputs for this mini batch) '''
        # initialise gradients to 0
        # the sum of the gradients of the loss function with respect to the biases for all examples in the mini-batch.
        sum_of_bias_gradients = [np.zeros_like(b) for b in self.bias]
        sum_of_weight_gradients = [np.zeros_like(w) for w in self.weights]
        # Calculate gradient for each example in mini batch.
        for x, y in mini_batch:
            grad_b, grad_w = self.backprop(x, y)
            # update sum of bias and weight gradients (elementwise addition for gradients)
            # the overall gradient used to update the weights and biases is the average of the gradients computed for each individual training example within that mini-batch.
            sum_of_bias_gradients = [bias_sum + gradient_bias for bias_sum, gradient_bias in zip(sum_of_bias_gradients, grad_b)]
            sum_of_weight_gradients = [weight_sum + gradient_weight for weight_sum, gradient_weight in zip(sum_of_weight_gradients, grad_w)]
        
        mini_batch_size = len(mini_batch)
        learning_rate_scaled = learning_rate / mini_batch_size
        # update weights and biases using avg gradients
        self.weights = [
            w - learning_rate_scaled * weight_sum
            for w, weight_sum in zip(self.weights, sum_of_weight_gradients)
        ]

        self.biases = [
            b - learning_rate_scaled * bias_sum
            for b, bias_sum in zip(self.bias, sum_of_bias_gradients)
        ]

    def backprop(self, x, y):
        """
        Calculates the gradient for the cost function C_x
        for a single training example (x, y) using backpropagation.

        Args:
            x (np.ndarray): The input feature vector.
            y (np.ndarray): The true label/target output vector.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing
            (grad_b, grad_w), which are layer-by-layer lists of numpy
            arrays representing the gradients for biases and weights,
            respectively.
        """
        grad_b, grad_w = [np.zeros_like(b.shape) for b in self.bias], [np.zeros(w.shape) for w in self.weights]
        # forward pass
        # input layer activation and list to store all activations (layer by layer)
        activation = x
        activations = [x]
        all_z = [] # list to store all zs = (wx + b) and activation is basically sigmoid(z)
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, activation) + b
            all_z.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backward pass
        # activations[-1] is the output layer's activation (by output layer I mean last layer)
        # all_z[-1] is the weighted input for the output layer
        error = self.cost_derivative(activations[-1], y) * self.sigmoid_derivative(all_z[-1])
        # update last layers
        grad_b[-1] = error
        grad_w[-1] = np.dot(error, activations[-2].T) # a^(L-2) * error = partial derivative of cost wrt weight of last layer

        # Iterate backward through the layers (from second-to-last to input)
        # We iterate in reverse order of layers (from output towards input).
        # Layer 1 is the first hidden layer (weights[0], biases[0]).
        # Layer L-1 is the last hidden layer (weights[-2], biases[-2]).
        for layer_idx in reversed(range(self.num_layers - 1)):
            # skip last layer (we handled that above)
            if layer_idx == self.num_layers - 2: 
                continue 
            z = all_z[layer_idx]
            sd = self.sigmoid_derivative(z)
            # Error propagation: delta for current layer from delta of next layer
            error = np.dot(self.weights[layer_idx + 1].T, error) * sd

            # Gradients for current layer's biases and weights
            grad_b[layer_idx] = error
            grad_w[layer_idx] = np.dot(error, activations[layer_idx].T)

        return grad_b, grad_w

    
    def cost_derivative(self, output_activations, y):
        ''' Return prediction - actual '''
        return output_activations - y


    def evaluate(self, test_data):
        ''' Return the number of test inputs for which the network outputs correct result '''
        test_results = [
                        (np.argmax(self.forward(x)), y) for (x, y) in test_data
                        ]
        return sum(int(x == y) for (x, y) in test_results)

training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()

# 784 input image pixels, 30 hidden neurons, 10 output neurons consisting of probability of images
net = network.Network([784, 30, 10])

# use stochastic gradient descent to learn from the MNIST training_data over 30 epochs, with a mini-batch size of 10, and a learning rate of η = 0.01
net.SGD(training_data, 30, 10, 0.01, test_data=testing_data)

# Predicting new image
def preprocess_image(image_path):
    """
    Loads an image, converts to grayscale, resizes to 28x28,
    normalizes pixels, and reshapes to (784, 1) numpy array.
    """
    try:
        img = Image.open(image_path).convert('L') # grayscale
        img = img.resize((28, 28)) 
        img_array = np.array(img) 

        # MNIST pixels are 0 (black) to 255 (white).
        # Flatten and transpose to (784, 1) column vector
        img_vector = np.reshape(img_array, (784, 1)) / 255.0
        return img_vector

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

image_path = Path(__file__).parent / "data" / "sample_digit.jpg"
processed_image = preprocess_image(image_path)
if processed_image is not None:
    output_activation = net.forward(processed_image)
    prediction = np.argmax(output_activation) # idx of neuron with highest activation
    print("\nNetwork Output Activations (confidence for each digit 0-9):")
    for i, val in enumerate(output_activation.flatten()):
            print(f"  Digit {i}: {val * 100:.2f}%")
    print(f"\nPredicted digit: {prediction}")
else:
    print("Could not make a prediction due to image processing error.")


```
Let's try it for this sample image:

![9](https://i.ibb.co/mMQXZbS/Screenshot-2025-06-12-at-12-36-17-AM.png)

**Output**
```text
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
  Digit 9: 70.89% # highest

Predicted digit: 9 # HELL YEAH
```

This was tedious, hence we have libraries like PyTorch/TensorFlow that already does this calculus shit for us.


### Basic Model in Libraries like Pytorch/Tensorflow
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(), # Converts image to tensor and scales to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # Standardize with MNIST's mean and std
])

# Download and load the training, testing data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load data
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

class PyTorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_to_hidden = nn.Linear(3, 3) # inp to hidden layer
        self.relu = nn.ReLU() # activation function
        self.hidden_to_pred = nn.Linear(3, 3) # hidden to output layer

    # Takes input
    def forward(self, x):
        x = self.inp_to_hidden(x)
        x = self.relu(x)
        x = self.hidden_to_pred(x)
        return x 

model = PyTorchNet()
loss = nn.CrossEntropyLoss() # target - pred 

optimizer = optim.Adam(model.parameters(), lr=0.01) # weights = (weights - learning_rate) * derivative

epochs = 2000
for epoch in range(epochs):
    for xb, yb in loader:
        optimizer.zero_grad() # reset weights to 0 after each batch so that they are updated correctly
        pred = model(xb) # calculate prediction
        final_loss = loss(pred, yb) # loss of prediction
        final_loss.backward() # apply backpropagation to this 
        optimizer.step() # update weights correctly
    if epoch % 500 == 0: # every 500th iteration : print loss
        print(f"Epoch {epoch}, Loss : {final_loss.item():.4f}")

# prediction
with torch.no_grad():
    pred = model(features)
    probs = torch.softmax(pred, dim=1)
    prediction = torch.argmax(probs, dim=1) # take the highest probability (example : correct digit in our MNIST example)
    print("\nPredicted class probabilities:\n", probs) # probability of each class (example : 0 - 9 all digit probabilities)
    print("Predicted classes:", prediction) # predicted outputs
    print("True classes:", targets) # actual outputs
```


### MNIST from MiniTorch (Custom Implementation of Pytorch)

[MiniTorch](https://samitmohan.github.io/minitorch)

```python
import numpy as np
from minitorch.tensor import Tensor
from minitorch.layers import Linear, ReLU
from minitorch.loss import cross_entropy_loss
from minitorch.optim import SGD, Adam

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def run_mnist_example():
    print("Fetching MNIST data...")
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train_all[:1000]
    y_train = y_train_all[:1000]
    X_test = X_test_all[:200]
    y_test = y_test_all[:200]

    def one_hot(labels, num_classes=10):
        oh = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
        oh[np.arange(labels.shape[0]), labels] = 1.0
        return oh

    y_train_oh = one_hot(y_train, 10)
    y_test_oh = one_hot(y_test, 10)

    X_t = Tensor(X_train, requires_grad=False)
    y_t = Tensor(y_train_oh, requires_grad=False)

    layer1 = Linear(784, 128)
    relu = ReLU()
    layer2 = Linear(128, 10)

    params = layer1.parameters() + layer2.parameters()
    opt = SGD(params, lr=0.01, momentum=0.9)

    epochs = 10
    batch_size = 100
    num_batches = X_train.shape[0] // batch_size

    print("Training")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            xb = Tensor(X_train[start:end], requires_grad=False)
            yb = Tensor(y_train_oh[start:end], requires_grad=False)

            out1 = layer1(xb)
            act1 = relu(out1)
            logits = layer2(act1)
            loss = cross_entropy_loss(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.data

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    X_test_t = Tensor(X_test, requires_grad=False)
    out1_test = layer1(X_test_t)
    act1_test = relu(out1_test)
    logits_test = layer2(act1_test)
    preds = np.argmax(logits_test.data, axis=1)
    accuracy = np.mean(preds == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    run_mnist_example()
```
**Output**
```output
(.venv) smol@samit ~/fun/minitorch git:(master) > python3 mnist_example.py 
Fetching MNIST data...
Training
Epoch 1/10, Loss: 71.2395
Epoch 2/10, Loss: 24.0194
Epoch 3/10, Loss: 11.0858
Epoch 4/10, Loss: 6.6417
Epoch 5/10, Loss: 4.4537
Epoch 6/10, Loss: 3.4053
Epoch 7/10, Loss: 2.5522
Epoch 8/10, Loss: 1.8609
Epoch 9/10, Loss: 1.4868
Epoch 10/10, Loss: 1.0654
Test Accuracy: 78.50% # I know this sucks, can improve it via CNN, but it works.
```

![Output](https://i.ibb.co/zT52v6kZ/outp-ut.png)


All of this shit is the same, just think of it as same sentence but in a different language. You can see the underlying pattern behind the minitorch/pytorch/tensorflow are same (they all follow numpy which just follows basic calculus)

For better results you can also use CNN (Convolution Neural Network) which is just a fancy word for optimising our neural network via multiple stages and feature extraction, you can see this [video](https://www.youtube.com/watch?v=FmpDIaiMIeA&list=PLJf-Umv9fV2N8n4g83wvCwtj2wOX7RniG&index=63) for more. 

But at a quick glance this is how it works:

![CNN](https://i.ibb.co/yBhrrSyN/cnn.png)

- *Input:* 28 * 28 pixel image.

- *Convolution layer:* multiples this image by some filter (2x2 matrix) to get a new matrix which learns the underlying patterns of the image (in this case, the strokes)

- *Max Pooling:* Extracts the most important features and reduces size of the original image (14 * 14 now)

- This goes through another Convolution Layer followed by another Max Pooling layer to identify more patterns and in the end we end up with 7 * 7 image and 64 of them which are then *flattened* into a single array of 7*7*64 numbers = 3136

- After flattening this we find the probabilities of our numbers by applying the *dense* layer which outputs 10 numbers (0 to 9) and predicts which number has the highest activation / proabability (i.e predicting what number our image is)


## Loss vs Accuracy graph:
![AccuracyLoss](https://i.ibb.co/7JYwZ4hV/accyracy.png)

**Makes sense, loss goes down and accuracy of the model goes up over time.**

This is the heart of all algorithms. Every program you will ever code in machine learning / AI will involve these basic chain rule calculus and matrix multiplications at it's heart to learn the proper parameters (weights and bias) for your model. 

## Problems with Gradient Descent 
### Vanishing Gradient Problem
So what happens when you backprop? You are essentially using an activation function (sigmoid/tanh) to squash numbers between 0 to 1 and then using chain rule multiply them to achieve backpropagation. 

If you have multiple (ALOT) of layers & you're doing this → The value at which the initail weights learn (by the time it gets there) becomes close to 0 so they don't end up learning.

If you're using sigmoid activations in a deep network, and many of your neurons' inputs fall into the "saturated" regions, their derivatives will be close to zero (e.g., 0.1, 0.01, or even smaller).


When you multiply many such small numbers together during backpropagation  
(e.g., $0.25 \times 0.25 \times 0.25 \dots$),  the result becomes an extremely small number very quickly.

This "vanishing" gradient means that the weights in the early layers receive an almost zero gradient signal and the network never really learns anything.

### Dead Neuron Problem
Negative Input: If the weighted sum of inputs to a ReLU neuron (before applying the activation function) is negative, the ReLU function will output 0 (defination of $RelU : f(a) = max(0, a)$)

*Zero Gradient:* When the output is 0, the derivative of the ReLU function with respect to its input is also 0. 

During backpropagation, the gradient of the loss with respect to the neuron's weights and biases is calculated by multiplying the upstream gradient with the local gradient. 

If the local gradient is 0, then the gradients for the weights and biases connected to this neuron become 0.

*No Weight Updates:* If the gradients are 0, the weights and biases of that neuron will not be updated during subsequent training iterations.

Permanent Inactivity: This means the neuron is effectively "dead." It will always output 0, regardless of the input, and it will never learn or contribute to the network's learning process. It's stuck in an inactive state.

Solution to this? Use Leaky ReLU function which never gets 0.

$$
f(x) =
\begin{cases} 
x & \text{if } x > 0 \\ 
\alpha x & \text{if } x \leq 0 
\end{cases}
$$
Where:
- $x$: Input to the activation function.  
- $\alpha$: A small positive constant (e.g., $(0.01$) that determines the slope for negative input values.

## Conclusion

In a much larger sense these parameter / weights are just probabilities of all english words that are given to an LLM like chatGPT.

LLMs have hundreds of these parameters. They begin at random and repeatedly refined based on text (chatgpt uses wikipedia text) → Backpropagation is used to tweak all paramters so that it makes the model a little more likely to choose the true last word (accurate word) instead of other words.

*Example:*
I am going to the bank to deposit some _

Probabilties : money, cash, blood, ashes, etc... (Actual last word here is money) 

A good video on this: [LLMs explained](https://www.youtube.com/watch?v=LPZh9BOjkQs&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=5)

Classifying Images or Generating Texts (GPT) uses these ideas at the core. 

Neural networks are the base for understanding AI. Transformers, Attention and all cutting edge AI papers are based on this simple key idea of y = mx + c and how to find the slope and intercept of this line in order to minimise our loss. 

It's so interesting to see that behind these complicated ideas it's just high school math and common sense. I spent the weekend reading these resources, writing code and it felt so nice to be back in touch with school math (although I was never good at it) and discover AI from first principles (perceptrons) I was going to write a section about attention and transformers and how GPT works but that is a blog for another time, this is a heavy read as it is :P


Code for all of this can be found here → [MNIST](https://github.com/samitmohan/ML/tree/master/NeuralNetworks/MNIST)


## Optional
Just a little more math.

Congratulations you made it till the end you are officially lame.

Basically explains the 4 equations of backpropagation that we saw earlier, not needed but I think it's fun to visit calculus when you haven't in a long time, teaches you a lot about math and you realise that the symbols only look scary but they aren't once you understand them.

If you've read this entirely, this is going to make a lot of sense.

### Single Neuron Chain Rule

Let's define the operations for a single neuron:

- **Linear Combination:** $z = Wx + b$, where $W$ is the weight, $x$ is the input, and $b$ is the bias.
- **Activation:** $a = \sigma(z)$, where $a$ is the activation (this is basically the `predicted_output`). The activation function $\sigma$ can be sigmoid, ReLU, etc.—anything that squashes our function into the range $[0, 1]$.
- **Loss:** $L = \text{loss}(a, y)$, where $L$ is a function of two things: the predicted output $a$ and the actual output $y$.

**Chain Rule: Calculating the Gradient of the Loss**

How does the loss $L$ change with respect to the weight $W$ and bias $b$?  
The loss depends on $W$ only through the activation $a$ and the linear combination $z$.  
We can trace this dependency: $L \leftarrow a \leftarrow z \leftarrow W$.

The chain rule gives us:
$$ \frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial W} $$

Let's calculate these three values individually:
1.  **$\frac{\partial L}{\partial a}$**  
    This measures how the loss changes if you wiggle the neuron's output.  
    For a simple squared error loss $L = \frac{1}{2}(a - y)^2$, this derivative is:
    
    $$
    \frac{\partial L}{\partial a} = a - y
    $$
    (predicted output − actual output)

2.  **$\frac{\partial a}{\partial z}$**  
    This is the derivative of the activation function.  
    For a sigmoid function:

    $$
    \frac{\partial a}{\partial z} = \sigma'(z) = \sigma(z)(1 - \sigma(z))
    $$

3.  **$\frac{\partial z}{\partial W}$**  
    Since $z = Wx + b$, the partial derivative with respect to $W$ is just $x$:

    $$
    \frac{\partial z}{\partial W} = x
    $$

### For Multiple Layers

Now let's consider a simple network: input $x$ $\rightarrow$ hidden layer $h$ $\rightarrow$ output $\hat{y}$



*   **Hidden Layer:** $ h = \sigma(z^{(1)}) = \sigma(W^{(1)}x + b^{(1)}) $
*   **Output Layer:** $ \hat{y} = \sigma(z^{(2)}) = \sigma(W^{(2)}h + b^{(2)})$
*   **Loss:** $ L = \text{loss}(\hat{y}, y)$

To find the gradient with respect to the first layer's weights, \(\frac{\partial L}{\partial W^{(1)}}\), we need a longer chain rule:
$$ \frac{\partial L}{\partial W^{(1)}} = \left( \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \right) \cdot \frac{\partial h}{\partial z^{(1)}} \cdot \frac{\partial z^{(1)}}{\partial W^{(1)}} $$

### Backpropagation: The Full Process

1.  **Forward Pass:** Compute all \(z\) values (linear equations) and \(a\) values (activations) for all layers, from input to output.
2.  **Initialise:** Set $\frac{\partial L}{\partial L} = 1$.  
    This seeds the chain rule so that every other partial derivative in the graph gets scaled correctly.  
    You want to find $\frac{\partial L}{\partial W}$, which can be thought of as  
    $\frac{\partial L}{\partial L} \cdot \frac{\partial L}{\partial W}$.

3.  **Backward Pass:** For each node in reverse topological order (from output to input), call its local derivative function to propagate the "error" or "gradient" to its inputs, and use this to calculate the gradients for the node's parameters (weights and bias).

- one input layer $x$ (column vector)  
- one hidden layer (weights $W^{(1)},\; b^{(1)}$)  
- one output layer (weights $W^{(2)},\; b^{(2)}$)  
- a sigmoid activation $\sigma(\cdot)$ everywhere  
- a mean-squared-error loss  

You can extend exactly the same logic to deeper nets.


## 1. Forward pass

The network prediction \(\hat{y}\) is produced in three little steps.

```text
input  →  linear  →  non-linear  →  linear  →  non-linear  →  ŷ
              (hidden)                (output)
```

1. Hidden-layer pre-activation  
   $z^{(1)} = W^{(1)}x + b^{(1)} $

2. Hidden activation  
   $ h = \sigma\!\bigl(z^{(1)}\bigr)$

3. Output pre-activation  
   $ z^{(2)} = W^{(2)}h + b^{(2)}$

4. Network prediction  
   $\hat{y} = \sigma\!\bigl(z^{(2)}\bigr)$

## 2. Loss

We penalise the squared difference between the prediction and the true label $y$.

$$
L(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2
$$

(Using $\tfrac{1}{2}$ conveniently cancels the factor 2 that appears when we differentiate.)

## 3. Back-propagation: the 4 core equations

Back-prop simply applies the chain rule from right to left, layer by layer.
### Eq ① Output-layer error

$$
\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}}  
= (\hat{y} - y)\,\sigma'\bigl(z^{(2)}\bigr)
$$

*“How much does the loss change if I nudge the output neuron’s input up or down?”*

### Eq ② Gradients for output weights and bias

$$
\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)}\,h^{\top},  
\qquad
\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)}
$$

*“Each weight in layer 2 matters in proportion to the hidden activation it multiplies.”*

### Eq ③ Propagate the error back to the hidden layer

$$
\delta^{(1)} = \frac{\partial L}{\partial z^{(1)}}  
= \left(W^{(2)}\right)^{\top} \delta^{(2)} \odot \sigma'\bigl(z^{(1)}\bigr)
$$

($\odot$ = element-wise product)  
*“Hidden neurons are blamed for the final error in proportion to how strongly they feed it and how steep their own sigmoid is.”*

### Eq ④ Gradients for first-layer weights and bias

$$
\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)}\,x^{\top},  
\qquad
\frac{\partial L}{\partial b^{(1)}} = \delta^{(1)}
$$

*“Each input weight is corrected by the input value it scales, times the hidden error.”*

## 4. Parameter update  (Gradient Descent step)

With learning rate $\eta$:

$$
W^{(i)} \leftarrow W^{(i)} - \eta\,\frac{\partial L}{\partial W^{(i)}}, 
\qquad
b^{(i)} \leftarrow b^{(i)} - \eta\,\frac{\partial L}{\partial b^{(i)}}, 
\quad i \in \{1, 2\}
$$

This finishes one training iteration:  
forward → compute loss → apply the four back-prop equations → update.

Back-prop is nothing more than “apply the chain rule repeatedly”.  
- Eq ① makes the final layer aware of its mistake.
- Eq ② turns that mistake into gradients for its own parameters.
- Eq ③ pushes the blame backwards.
- Eq ④ turns that earlier blame into earlier gradients, and finally we step the parameters downhill with gradient descent.

## References
- [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Backpropagation](https://www.youtube.com/watch?v=SmZmBKc7Lrs)
- [Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [StatQuest](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- Books : MLforStatQuest, Why Machines Learn, Grokking Deep Learning, [Michael Nelson](https://neuralnetworksanddeeplearning.com/)

<!-- ### New Information from Grokking Book -->