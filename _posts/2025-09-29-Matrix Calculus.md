---
layout: mysingle
date: 2025-09-29 14:01:16 +0800
title: Matrix calculus
categories: machine_learning
header:
    # overlay_color: "#333"
    overlay_color: "#2f4f4f" #暗岩灰
    # overlay_color: "#e68ab8" #火鹤红
classes: wide
tags: math calculus

toc: true
---

## Layout conventions

The fundamental issue is that the derivative of a vector with respect to a vector, i.e ${\frac {\partial \mathbf {y} }{\partial \mathbf {x} }}$ is often written in two competing ways. If the numerator **y** is of size *m* and the denominator **x** of size *n*, then the result can be laid out as either an *m*×*n* matrix or *n* × *m* matrix, i.e. the *m* elements of **y** laid out in rows and the *n* elements of **x** laid out in columns, or vice versa. This leads to the following possibilities:

### Numerator layout

 Lay out according to **y** and $\mathbf {x}^T$ (i.e. contrarily to **x**). This is sometimes known as the *Jacobian formulation*. This corresponds to the *m*×*n* layout in the previous example, which means that the row number of $\frac {\partial \mathbf {y} }{\partial \mathbf {x} }$ equals to the size of the numerator $ \mathbf {y} $ and the column number of $\frac {\partial \mathbf {y} }{\partial \mathbf {x} }$ equals to the size of $\mathbf {x}^T$.

 The derivative of a [vector function](https://en.wikipedia.org/wiki/Vector_function "Vector function") (a vector whose components are functions)

${\displaystyle \mathbf {y} ={\begin{bmatrix}y_{1}&y_{2}&\cdots &y_{m}\end{bmatrix}}^{\mathsf {T}}}$

, with respect to an input vector,

${\displaystyle \mathbf {x} ={\begin{bmatrix}x_{1}&x_{2}&\cdots &x_{n}\end{bmatrix}}^{\mathsf {T}}}$

, is written (in [numerator layout notation](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)) as

$${\frac {\partial \mathbf {y} }{\partial \mathbf {x} }}={\begin{bmatrix}{\frac {\partial y_{1}}{\partial x_{1}}}&{\frac {\partial y_{1}}{\partial x_{2}}}&\cdots &{\frac {\partial y_{1}}{\partial x_{n}}}\\{\frac {\partial y_{2}}{\partial x_{1}}}&{\frac {\partial y_{2}}{\partial x_{2}}}&\cdots &{\frac {\partial y_{2}}{\partial x_{n}}}\\\vdots &\vdots &\ddots &\vdots \\{\frac {\partial y_{m}}{\partial x_{1}}}&{\frac {\partial y_{m}}{\partial x_{2}}}&\cdots &{\frac {\partial y_{m}}{\partial x_{n}}}\\\end{bmatrix}}$$

#### Vector-by-vector
In [vector calculus](https://en.wikipedia.org/wiki/Vector_calculus "Vector calculus"), the derivative of a vector function **y** with respect to a vector **x** whose components represent a space is known as the **[pushforward (or differential)](https://en.wikipedia.org/wiki/Pushforward_(differential) "Pushforward (differential)")**, or the **[Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix "Jacobian matrix")**.

The pushforward along a vector function **f** with respect to vector **v** in $\mathbf {R}^n$ is given by

$$ d\mathbf {f} (\mathbf {v} )=\frac{\partial \mathbf {f} }{\partial \mathbf {v} } d(\mathbf {v} ) $$

#### Hessian Matrix
Suppose $f:\mathbb {R} ^{n}\to \mathbb {R}$ is a function taking as input a vector $\mathbf {x} \in \mathbb {R} ^{n}$ and outputting a scalar $\displaystyle f(\mathbf {x} )\in \mathbb {R}$

Then the hessian matrix H of H is a $n\times n$ matrix:

$$\mathbf {H} _{f}={\begin{bmatrix}{\dfrac {\partial ^{2}f}{\partial x_{1}^{2}}}&{\dfrac {\partial ^{2}f}{\partial x_{1}\,\partial x_{2}}}&\cdots &{\dfrac {\partial ^{2}f}{\partial x_{1}\,\partial x_{n}}}\\[2.2ex]{\dfrac {\partial ^{2}f}{\partial x_{2}\,\partial x_{1}}}&{\dfrac {\partial ^{2}f}{\partial x_{2}^{2}}}&\cdots &{\dfrac {\partial ^{2}f}{\partial x_{2}\,\partial x_{n}}}\\[2.2ex]\vdots &\vdots &\ddots &\vdots \\[2.2ex]{\dfrac {\partial ^{2}f}{\partial x_{n}\,\partial x_{1}}}&{\dfrac {\partial ^{2}f}{\partial x_{n}\,\partial x_{2}}}&\cdots &{\dfrac {\partial ^{2}f}{\partial x_{n}^{2}}}\end{bmatrix}}.$$

That is, the entry of the ith row and the jth column is

$$(\mathbf {H} _{f})_{i,j}={\frac {\partial ^{2}f}{\partial x_{i}\,\partial x_{j}}}$$ (differentiate first with respect to \\(x\_j\\), then \\(x\_i\\))

The Hessian matrix of a function $\displaystyle f$ is the transpose of the Jacobian matrix of the gradient of the function $\displaystyle f$; that is:

$$\mathbf {H} (f(\mathbf {x} ))=\mathbf {J} (\nabla f(\mathbf {x} ))^{\mathsf {T}}.$$

The symmetry of the Hessian matrix of a continuously differentiable function depends on a key condition: **the continuity of the function’s second-order partial derivatives** (often called "continuous twice differentiability," denoted \\(C^2\\)). In short:

*   A function that is only *continuously differentiable* (i.e., first-order partial derivatives exist and are continuous, denoted \\(C^1\\)) **does not guarantee a symmetric Hessian**—its second-order partial derivatives may not even exist, or if they do, they may be asymmetric.
*   A function that is *continuously twice differentiable* (i.e., second-order partial derivatives exist and are continuous, denoted \\(C^2\\)) **always has a symmetric Hessian**—this is proven by Clairaut’s Theorem (also known as Schwarz’s Theorem).

#### Matrix-by-scalar

$${\displaystyle {\frac {\partial \mathbf {y} }{\partial \mathbf {x} }}={\begin{bmatrix}{\frac {\partial y_{1}}{\partial x_{1}}}&{\frac {\partial y_{1}}{\partial x_{2}}}&\cdots &{\frac {\partial y_{1}}{\partial x_{n}}}\\{\frac {\partial y_{2}}{\partial x_{1}}}&{\frac {\partial y_{2}}{\partial x_{2}}}&\cdots &{\frac {\partial y_{2}}{\partial x_{n}}}\\\vdots &\vdots &\ddots &\vdots \\{\frac {\partial y_{m}}{\partial x_{1}}}&{\frac {\partial y_{m}}{\partial x_{2}}}&\cdots &{\frac {\partial y_{m}}{\partial x_{n}}}\\\end{bmatrix}}}$$

#### Scalar-by-matrix

$${\displaystyle {\frac {\partial y}{\partial \mathbf {X} }}={\begin{bmatrix}{\frac {\partial y}{\partial x_{11}}}&{\frac {\partial y}{\partial x_{21}}}&\cdots &{\frac {\partial y}{\partial x_{p1}}}\\{\frac {\partial y}{\partial x_{12}}}&{\frac {\partial y}{\partial x_{22}}}&\cdots &{\frac {\partial y}{\partial x_{p2}}}\\\vdots &\vdots &\ddots &\vdots \\{\frac {\partial y}{\partial x_{1q}}}&{\frac {\partial y}{\partial x_{2q}}}&\cdots &{\frac {\partial y}{\partial x_{pq}}}\\\end{bmatrix}}.}$$


In analog with [vector calculus](https://en.wikipedia.org/wiki/Vector_calculus "Vector calculus") this derivative is often written as the following.

$\displaystyle \nabla _{\mathbf {X} }y(\mathbf {X} )={\frac {\partial y(\mathbf {X} )}{\partial \mathbf {X} }}$

#### Summary


![](http://bens-1-pics.oss-cn-hongkong.aliyuncs.com/2025/09/29/17590491367939.jpg?x-oss-process=image/auto-orient,1/quality,q_90/watermark,text_YmVuc2Jsb2cudGVjaA,color_f5eded,size_15,x_10,y_10){:width="600px"}{: .align-center}


## Identities

### Vector-by-vector -> matrix

For numerator layout convention, the **Vector-by-vector** indentities **(bold terms** are vectors):

* a is not a function of x: ${\frac {\partial \mathbf {a} }{\partial \mathbf {x} }}=0$
* ${\displaystyle {\frac {\partial \mathbf {x} }{\partial \mathbf {x} }}=I}$
* **A** is not a function of **x**:
    * ${\frac {\partial \mathbf {A} \mathbf {x} }{\partial \mathbf {x} }}=\mathbf {A}$
    * ${\frac {\partial \mathbf {x} ^{\top }\mathbf {A} }{\partial \mathbf {x} }}={\frac {\partial \mathbf {A}^T \mathbf {x} }{\partial \mathbf {x} }}=\mathbf {A} ^{\top }$
    * with vector-by-vector, $y=x^TA=A^Tx$ is the same vector
* a is not a function of **x**, **u** = **u**(**x**):
$${\frac {\partial a\mathbf {u} }{\partial \,\mathbf {x} }}=a{\frac {\partial \mathbf {u} }{\partial \mathbf {x} }}$$
* v = v(**x**),a is not a function of **x**,
$${\frac {\partial v\mathbf {a} }{\partial \mathbf {x} }}=\mathbf {a} {\frac {\partial v}{\partial \mathbf {x} }}$$
* v=v(**x**), **u**=**u**(**x**),
$${\frac {\partial v\mathbf {u} }{\partial \mathbf {x} }}=v{\frac {\partial \mathbf {u} }{\partial \mathbf {x} }}+\mathbf {u} {\frac {\partial v}{\partial \mathbf {x} }}$$
* **A** is not a function of **x**,**u** = **u**(**x**)
$${\frac {\partial \mathbf {A} \mathbf {u} }{\partial \mathbf {x} }}=\mathbf {A} {\frac {\partial \mathbf {u} }{\partial \mathbf {x} }}$$

* **u** = **u**(**x**), **v** = **v**(**x**)
$${\frac {\partial (\mathbf {u} +\mathbf {v} )}{\partial \mathbf {x} }}={\frac {\partial \mathbf {u} }{\partial \mathbf {x} }}+{\frac {\partial \mathbf {v} }{\partial \mathbf {x} }}$$

* **u** = **u**(**x**),
$${\frac {\partial \mathbf {g} (\mathbf {u} )}{\partial \mathbf {x} }}={\frac {\partial \mathbf {g} (\mathbf {u} )}{\partial \mathbf {u} }}{\frac {\partial \mathbf {u} }{\partial \mathbf {x} }}$$

* **u** = **u**(**x**),
$${\frac {\partial \mathbf {f} (\mathbf {g} (\mathbf {u} ))}{\partial \mathbf {x} }}={\frac {\partial \mathbf {f} (\mathbf {g} )}{\partial \mathbf {g} }}{\frac {\partial \mathbf {g} (\mathbf {u} )}{\partial \mathbf {u} }}{\frac {\partial \mathbf {u} }{\partial \mathbf {x} }}$$

### scalar-by-vector -> vector

Most idendities can by inferred from vector by vector cases.
Special cases:

* **u** = **u**(**x**), **v** = **v**(**x**), **A** is not a function of **x**:
$${\frac {\partial (\mathbf {u} \cdot \mathbf {v} )}{\partial \mathbf {x} }}={\frac {\partial \mathbf {u} ^{\top }\mathbf {v} }{\partial \mathbf {x} }}=\mathbf {u} ^{\top }{\frac {\partial \mathbf {v} }{\partial \mathbf {x} }}+\mathbf {v} ^{\top }{\frac {\partial \mathbf {u} }{\partial \mathbf {x} }}$$

Trick: track the dimensions

$${\frac {\partial (\mathbf {u} \cdot \mathbf {A} \mathbf {v} )}{\partial \mathbf {x} }}={\frac {\partial \mathbf {u} ^{\top }\mathbf {A} \mathbf {v} }{\partial \mathbf {x} }}=\mathbf {u} ^{\top }\mathbf {A} {\frac {\partial \mathbf {v} }{\partial \mathbf {x} }}+\mathbf {v} ^{\top }\mathbf {A} ^{\top }{\frac {\partial \mathbf {u} }{\partial \mathbf {x} }}$$

Here: $${\frac {\partial \mathbf {u} }{\partial \mathbf {x} }},{\frac {\partial \mathbf {v} }{\partial \mathbf {x} }}$$ both in numerator layout

* **a** is not a function of **x**:
$${\frac {\partial (\mathbf {a} \cdot \mathbf {x} )}{\partial \mathbf {x} }}={\frac {\partial (\mathbf {x} \cdot \mathbf {a} )}{\partial \mathbf {x} }}={\frac {\partial \mathbf {a} ^{\top }\mathbf {x} }{\partial \mathbf {x} }}={\frac {\partial \mathbf {x} ^{\top }\mathbf {a} }{\partial \mathbf {x} }}=\mathbf {a} ^{\top }$$


* **A** is not a function of **x**
$${\frac {\partial \mathbf {x} ^{\top }\mathbf {A} \mathbf {x} }{\partial \mathbf {x} }}=\mathbf {x} ^{\top }\left(\mathbf {A} +\mathbf {A} ^{\top }\right)$$

$${\frac {\partial ^{2}\mathbf {x} ^{\top }\mathbf {A} \mathbf {x} }{\partial \mathbf {x} \partial \mathbf {x} ^{\top }}}=\mathbf {A} +\mathbf {A} ^{\top }$$

if **A** is also symmetric:

$${\frac {\partial \mathbf {x} ^{\top }\mathbf {A} \mathbf {x} }{\partial \mathbf {x} }}=\mathbf {x} ^{\top }\left(\mathbf {A} +\mathbf {A} ^{\top }\right)=2\mathbf {x} ^{\top }\mathbf {A}$$

$${\frac {\partial ^{2}\mathbf {x} ^{\top }\mathbf {A} \mathbf {x} }{\partial \mathbf {x} \partial \mathbf {x} ^{\top }}}=2\mathbf {A}$$

if **A**=**I**:

$${\frac {\partial (\mathbf {x} \cdot \mathbf {x} )}{\partial \mathbf {x} }}={\frac {\partial \mathbf {x} ^{\top }\mathbf {x} }{\partial \mathbf {x} }}={\frac {\partial \left\Vert \mathbf {x} \right\Vert ^{2}}{\partial \mathbf {x} }}=2\mathbf {x} ^{\top }$$

* ${\frac {\partial ^{2}f}{\partial \mathbf {x} \partial \mathbf {x} ^{\top }}}=\mathbf {H} ^{\top }$; $ \mathbf {H} $ is the Hessian matrix

$$(\mathbf {H} _{f})_{i,j}={\frac {\partial ^{2}f}{\partial x_{i}\,\partial x_{j}}}.$$


Notion: all the identities above works for vector times vector, no matrix!

* **a**, **b** are not functions of **x**

$${\frac {\partial \;{\textbf {a}}^{\top }{\textbf {x}}{\textbf {x}}^{\top }{\textbf {b}}}{\partial \;{\textbf {x}}}}={\textbf {x}}^{\top }\left({\textbf {a}}{\textbf {b}}^{\top }+{\textbf {b}}{\textbf {a}}^{\top }\right)$$

where $a\in R^{n\times 1},b\in R^{n\times 1}$

* **A, b, C, D, e** are not functions of **x**

$${\frac {\partial \;({\textbf {A}}{\textbf {x}}+{\textbf {b}})^{\top }{\textbf {C}}({\textbf {D}}{\textbf {x}}+{\textbf {e}})}{\partial \;{\textbf {x}}}}=({\textbf {A}}{\textbf {x}}+{\textbf {b}})^{\top }{\textbf {C}}{\textbf {D}}+({\textbf {D}}{\textbf {x}}+{\textbf {e}})^{\top }{\textbf {C}}^{\top }{\textbf {A}}$$


* **a** is not a function of **x**

$${\frac {\partial \;\|\mathbf {x} -\mathbf {a} \|}{\partial \;\mathbf {x} }}={\frac {(\mathbf {x} -\mathbf {a} )^{\top }}{\|\mathbf {x} -\mathbf {a} \|}}$$

### scalar-by-matrix -> matrix

$${\frac {\partial (u+v)}{\partial \mathbf {X} }}={\frac {\partial u}{\partial \mathbf {X} }}+{\frac {\partial v}{\partial \mathbf {X} }}$$

$${\frac {\partial uv}{\partial \mathbf {X} }}=u{\frac {\partial v}{\partial \mathbf {X} }}+v{\frac {\partial u}{\partial \mathbf {X} }}$$

$${\frac {\partial g(u)}{\partial \mathbf {X} }}={\frac {\partial g(u)}{\partial u}}{\frac {\partial u}{\partial \mathbf {X} }}$$

$${\frac {\partial f(g(u))}{\partial \mathbf {X} }}={\frac {\partial f(g)}{\partial g}}{\frac {\partial g(u)}{\partial u}}{\frac {\partial u}{\partial \mathbf {X} }}$$

$${\frac {\partial g(\mathbf {U} )}{\partial X_{ij}}}=\operatorname {tr} \left({\frac {\partial g(\mathbf {U} )}{\partial \mathbf {U} }}{\frac {\partial \mathbf {U} }{\partial X_{ij}}}\right)$$


$${\frac {\partial \mathbf {a} ^{\top }\mathbf {X} \mathbf {b} }{\partial \mathbf {X} }}=\mathbf {b} \mathbf {a} ^{\top }$$

Here $a\in R^{ m \times 1}, b\in R^{n \times 1}, X \in R^{m\times n}$

* **a** and **b** are not functions of X, f(**v**) is a real-valued differentiable function

$${\frac {\partial f(\mathbf {Xa+b} )}{\partial \mathbf {X} }}=\mathbf {a} {\frac {\partial f}{\partial \mathbf {v} }}$$


* a, b and C are not functions of X

$${\frac {\partial (\mathbf {X} \mathbf {a} )^{\top }\mathbf {C} (\mathbf {X} \mathbf {b} )}{\partial \mathbf {X} }}=\left(\mathbf {C} \mathbf {X} \mathbf {b} \mathbf {a} ^{\top }+\mathbf {C} ^{\top }\mathbf {X} \mathbf {a} \mathbf {b} ^{\top }\right)^{\top }$$


$${\frac {\partial (\mathbf {X} \mathbf {a} )^{\top } (\mathbf {X} \mathbf {a} )}{\partial \mathbf {X} }}=\left( \mathbf {X} \mathbf {a} \mathbf {a} ^{\top }+\mathbf {X} \mathbf {a} \mathbf {a} ^{\top }\right)^{\top }=2\mathbf {a}\mathbf {a} ^{\top }\mathbf {X}^{\top }$$
