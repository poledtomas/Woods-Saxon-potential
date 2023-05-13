This program is solving the Schrödinger equation with Wood-Saxon potencial numerically using the finite difference method. The finite difference method is an approach to solve differential equations numerically. The crux of the scheme lies in approximating the differential operator by simple differences. In principal, the code is calculating eigenvalues of matrix.  Time dependent Schrodinger equation

$-\frac{\hbar^2}{2m}\nabla^2\psi(x,t) + V \psi(x,t) = i\hbar\frac{\partial}{\partial t}\psi(x,t) $

But we solve only  the time-independent Schrödinger equation $t=t_0$
$-\frac{\hbar^2}{2m}\nabla^2\psi(x) + V \psi(x) = E\psi(x),$
where $\nabla^2=\frac{d^2}{d x^2}$.
