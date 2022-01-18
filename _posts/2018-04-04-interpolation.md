---
title: 'Don't use standard cubic spline interpolation'
date: 2199-01-01
permalink: /posts/2018/2018-04-04-interpolation/
tags:
  - approximation
  - numerical computing
---
... or at least, know exactly what you're doing.

Following my numerics course in university, I somehow had this idea that standard cubic splines were a good thing (tm) and that this is what one should generally do when solving an interpolation problem in practice.
Let's see why this may - in many cases - be a bad idea, and how we can do better.

### How does standard cubic spline interpolation work?
Fitting a single, polynomial 

$$ y = f(x) = a_0 + a_1 x + a_2 x^2 + \ldots + a_N x^N$$

with 

$$ f(x_i) = y_i \quad \text{for} \quad i=0,\ldots,N \tag{1} $$

to the data $$(x_i, y_i)$$ [often leads to oscillatory solutions](https://en.wikipedia.org/wiki/Runge%27s_phenomenon).
Hence, it was recognized quite early that using a *piece-wise* (lower-order) polynomial interpolation

$$ y=f_i(x) = a_0 + a_1 x + \ldots + a_M x^M, \quad \text{for} \quad x_i \leq x \leq x_{i+1}$$

with $$M\llN$$ instead of a single (higher-oder) polynomial works very well in many applications. In this setting, the polynomial functions $$f_i$$ are called splines. The simplest instance of this idea is basic linear interpolation ($$M=1$$), where the data are interpolated by piece-wise first-order polynomials (i.e., linear splines).

To obtain a smoother interpolation, piece-wise cubic polynomials ($$M=3$$) can be used that allow for some curvature of the interpolating function.
However, this raises the question of how to select the $$4N$$ parameters $$a_{0,i}, a_{1,i}, a_{2,i}, a_{3,i} $$ of the interpolating polynomials.
Condition (1) gives us $$2N$$ equations that must be fulfilled (one for each of the two splines on either side of each data point, minus two since the end points only have splines on one side), which leaves us with $$2N$$ degrees of freedom (DOFs).
Since we're interested in a smooth interpolation, another useful condition is a continuous first- and second-order derivative, i.e.,

$$f_{i-1}^'(x_i) = f_i^'(x_i) \quad \text{and} \quad f_{i-1}^''(x_i) = f_i^''(x_i) \quad \text{for} \quad i=1, \ldots, N-1, $$

which gives us another $$2N-2$$ equations for the polynomial coefficients, leaving us with just 2 DOFs.
For these last two DOFs, there are several options; the most common ones are:
* Set $$f_0^''(x_0) = f_{N-1}^''(x_N) = 0$$. This choice is often called *Natural Splines* and leads to an interpolating function that turns into a linear function at both end points.
* Also enforce a continuous third-order derivative at $$x_1$$ and $$x_{N-1}$$. This is called the *not-a-knot condition*.

Finally, note that the parameters of the interpolating splines [can easily be determined as the solution to a system of $$4N$$ linear equations](http://mathworld.wolfram.com/CubicSpline.html).

### Why may this be a bad idea?
Monotonicity, locality, smoothness, ...
Example with > 0 restriction...

### How can we do better?
Akima, Pchip...

Refs...(Moler,+?)

### Examples
Let's see how the methods discussed above compare in some small examples.
First, we consider the interpolation of impulse data.

![Interpolation results for impulse data](/images/2018-04-04-interpolation-impulse.png){:height="400px"}
([Python code for generating this figure](/snippets/interpolation/interpolation.py))

The *ringing* behavior of the standard spline interpolation is very obvious; above we have seen several reasons why this may be harmful in practice.
The Akima and Pchip interpolations almost completely coincide for these data, so let's consider a second example to assess their differences.

![Interpolation results for sine data](/images/2018-04-04-interpolation-sine.png){:height="400px"}
([Python code for generating this figure](/snippets/2018-04-04-interpolation.py))

Here it becomes apparent that the Akima interpolation really does not result in a monotone approximation, as opposed to the Pchip interpolation.

### Conclusion
Blah...
Pointer regression...

[The code for reproducing the figures in this post is available on my GitHub page.](https://github.com/e-pet/e-pet.github.io/blob/master/snippets/2018-04-04-interpolation.py)

### References
* [Wolfram Mathworld: Cubic Spline](http://mathworld.wolfram.com/CubicSpline.html)
* [Wikipedia: Spline Interpolation](https://en.wikipedia.org/wiki/Spline_interpolation)
* [Cleve's Corner: Splines and Pchips](https://blogs.mathworks.com/cleve/2012/07/16/splines-and-pchips/)

----
