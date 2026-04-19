using Random
using LinearAlgebra
using GLMakie
using Statistics
using Distributions

dark_latexfonts = merge(theme_dark(), theme_latexfonts())
set_theme!(dark_latexfonts)

"""
Source: Mackay ITILA book (chapter 34)

Set of N observations D = { x^{(n) }_{n=1}^{N} 
where x is a J-dimensional vector and is a linear mixture of I underlying 
source signals s

x = Gs

G is the matrix of mixing coefficients and is not known. 

The simplest algorithm: I=J (same number of sources and observations)

Goal: recover s (within some multiplicative factors and possibly permuted) 

Assume that latent variables are independently distributed 
with marginal distributions P(s_i | H) = p_i (s_i) where H denotes the 
assumed form of this model and assumed probability distributions p_i of 
latent variables. 

The probability of observables and hidden variables given G and H:

P({x, s} | G, H) = ∏_{n=1}^N [ P(x^{(n)} | s^{(n)}, G, H) P( s^{(n)} | H) ]
= ∏_n [ ( ∏_j δ ( x_j^{(n)} - Σ_i G_{ji} s_i^{(n)} ) ( ∏_i p_i (s_i ^{(n)} ) ) ]

(going from the first equality to second equality: we plugged in x=Gs for the 
first term in the product, and used the independence assumption 
for the second product.)

Assume x is generated without noise (not really true but makes the inference 
easier). 

For learning G from D, the relevant quantity is the likelihood function 

P(D|G,H) = ∏_n P(x^(n) | G, H)

Each of the terms in the product, we can obtain by marginalizing over 
latent variables. Remember that 

∫ds δ(x-vs) f(s) = 1/v f(x/v)

For ease of notation, adopt hte summation convention: e.g. 

G_ji s_i^(n) = Σ_i G_{ji} s_i^(n)

a single factor in the likelihood is given by

P(x^n | G, H) = ∫d^I s^(n) P(x^(n) | s^(n), G, H) P(s^n | H)
              = ∫d^I s^(n) ∏_j δ( x_j - G_ji s_i^(n) ) ∏_i p_i (s_i^(n) )
              = \frac{1}{|det G|} ∏_i p_i ( G_{ij}^{-1} x_j )

              => ln P(x^(n) | G, H) = -ln |det G| + Σ_i ln p_i (G_{ij}^{-1} x_j)

(note that going from second equality to third, we used the delta function 
formula above, but instead of 1/v, we have 1/detG and instead of f(x/v), 
we have G_ji -> G_ij^{-1})

Figuring out the maximum likelihood: note the following identities from 
matrix calculus:

∂ / ∂G_{ji} ln(det G)   =   G_{ij}^{-1} = W_ij
∂ / ∂G_{ji} G_{lm}^{-1} =   -G_{lj}^{-1} G_{im}^{-1} = -W_{lj}W_{im}
∂ / ∂W_{ij} f           =   -G_{jm} ( ∂ / ∂ G_{lm} f ) G_{li}

Define a_i = W_{ij} x_j

ϕ_i (a_i) = d ( ln p_i (a_i) ) / d(a_i) = z_i (34.13)

indicates which direction a_i needs to change to make the probability of the 
data greater. 

Then we can obtain the gradient wrt G_{ji} using mat calc identities:

∂/∂G_{ji} ln P(x^(n) | G,H) = -W_ij -a_i z_i′ W_i′j

or alternatively derivative wrt W_{ij}:

∂/∂W_{ij} ln P(x^(n) | G,H) = G_ji + x_j z_i

if we choose to change W to ascend the gradient, we obtain the 
learning rule

ΔW ∝ [W^T]^{-1} + zx^T

---
The online steepest ascent algorithm (algorithm 34.2):

1) Put x through a linear mapping:
    a = Wx
2) Put a through a nonlinear map:
    z_i = ϕ_i (a_i)
where a popular choice for ϕ is ϕ = -tanh(a_i)
3) Adjust hte weights in accordance with
    ΔW ∝ [W^T]^{-1} + zx^T
---

Choices of ϕ:
Choice of ϕ defines the assumed prior for latent variable s. 

Consider the linear choice ϕ(a) = -ka. Due to 34.13, this implies 
we are dealing with a Gaussian distribution on the latent variables. 
( df/dx = d (ln p(x)) / dx = -kx => p(x) = e^{-kx^2 / 2} )

Gaussian distribution of latent variables is invariant under rotation of 
latent variables so there can be no evidence favouring any particular 
alignment of the latent variable space. Therefore it will never 
recover G. We need to hope that the sources are non Gaussian. Most real 
sources are non Gaussian and have thick tails. 

Consider now the tanh nonlinearity:

ϕ(a) = -tanh(a)

then we are implicitly assuming 
p(s) ∝ 1/cosh(s) ∝ 1 / ( e^{s_i} + e^{-s_i} )

This is a heavy tailed distribution. 

One can also use a tanh with gain β: 
ϕ(a) = -tanh(βa). The implicit probabilistic model is p(s)∝1/( cosh(βs) )^(1/β)

In the limit of large β the nonlinearity becomes a step function and 
the probability distribution becomes becomes biexponential (exp(-|s|)). 
In the limit of β -> 0, p(s) approaches a gaussian with mean zero and 
variance 1/β. 

A faster learning algorithm:

Finding the gradient of an objective function is a great idea. Ascending 
it directly is not so great. Above algorithm is slow and ill conditioned. 
It involves a matrix inverse which is a no-no. 

Covariant optimization:
Principle of covariance: a consistent algorithm should give the same 
results independent of the units in which quantities are measured. 

non covariant algo example: steepest descents rule. Consider a dimensionaless 
objective function L(w):

Δw_i = k ∂L / ∂w_i

The lhs has dimensions [w_i] whereas rhs has dimensions 1/[w_i]. The behavior 
of the algorithm is not covariant wrt linear rescaling of the vector w. 
Its not the end of the world, if k decreases with n (step) as 1/n, then 
the Munro-Robbins theorem shows that parameters will converge to the 
asymptotic parameters. But the non covariant algo will take a large number of 
iterations to converge. 

A covariant version:

Δw_i = k ∑_i′ M_ii′ ∂L/∂w_i

where M is a positive definite matrix whose i, i′ element has 
dimensions [w_i w_i′]. How can we figure out M? Two sources: 
metrics and curvatures. 

If a natural metric which can define distances in parameter space w, then 
M can be obtained from this metric. In the special case where 
there is a known quadratic metric can be obtained from the quadratic form, 
then the matrix can be obtained from the quadratic form. 

For example, if the length is w^2 then the natural matrix is M=I and 
steepest descents is appropriate. (look into covariant example, 
if M has units w^2 and k is dim.less, the rhs and lhs units match). 

Another way of finding a metric: look at the curvature of the objective function, 
defining A=-∇∇L (where ∇=∂/∂w). Then the matrix M=A⁻¹ will give a covariant 
algorithm. Furthermore, this is the Newton algorithm so we recognize that 
it will be fast (which is a problem for steepest descents algo). 

Sometimes A consits of both data dependent and independent terms, in this case, 
choose to define the metric using data independent terms only. The algo 
will still be covariant but not implement an exact Newton step. There is no 
unique choice in algorithm. 

For our MLE problem we have evalueated the gradient wrt G and the gradient 
wrt W=G⁻¹. Let's construct an alternative covariant algo. Take the 
second derivative of log likelihood wrt W and obtain two terms. 

(∂ / ∂W_{ij}) ln p(x^(n) | G, H) = G_{ji} + x_j z_i (34.15)

Data independent term:
∂G_{ji} / ∂W_{kl} = -G_{jk} G_{li}

data dependent:
∂(z_i x_j) / ∂W_{kl} = x_j x_l δ_{ik} z_i′ (no sum over i)

it is tempting to drop the data dependent term and define the matrix M 
by [M^{-1}]_{(ij)(kl)} = [G_{jk} G_{li}]. But this matrix is not 
positive definite. It is a poor approximation of the curvature which 
must be pos-def in the neighbourhood of a ML solution. We look into the data 
dependent term for inspiration. The aim is to find a convenient approximation 
to the curvature. What is the average value of x_j x_l δ_{ik} z_i′? 

If the true value of G is G*, then

⟨ x_j x_l δ_{ik} z_i′ ⟩ = ⟨ G_{jm}* s_m s_n G_{ln}* δ_ik z_i′ ⟩

Now we make severe assumptions: Replace G* with the present value of G, 
replace correlated average ⟨s_m s_n z_i′ ⟩ by ⟨s_m s_n⟩⟨z_i′⟩ = Σ_{mn}D_i. 

Σ is the variance-covariance matrix of hte latent variables. D_i is the 
typical value of the curvature d² ln p_i(a) / da². Given that sources 
are assumed to be independent, Σ and D are both diagonal matrices. These 
approximations motivate the matrix M given by

[M^{-1}]_{(ij)(kl)} = G_{jm} Σ_{mn} G_{ln} δ_{ik} D_i

=> 

M_{ij}{kl} = W_{mj} Σ⁻¹_{mn} W_{nl} δ_{ik} D⁻¹_i

for further simplicity, we assume that the sources are similar to each 
other so that Σ and D are both homogeneous and ΣD=1. This will lead to an 
algorithm that is covariant wrt linear rescaling of the data x, but 
not wrt linear rescaling of latent variables. We thus use 

M_{(ij)(kl)} = W_{mj} W_{ml} δ_{ik}

Multiplying this matrix by the gradient, we obtain the covariant learning 
algorithm

ΔW_{ij} = k ( W_{ij} + W_{i′j} a_{i′} z_i )

This expression doesn't require inversion of W. The only 
additional computation once z has been computed is a single 
backward pass through weights to compute the quantity

x′_j = W_{i′j} a_{i′}

=> ΔW_{ij} = k ( W_{ij} + x′_j z_i )

the quantity ( W_{ij} + x′_j z_i ) is also called the natural gradient. 

---
Algo 32.4
Repeat for each datapoint x⃗:

1) Put x through a linear mapping:
        a = Wx

2) Put a through a nonlinear map:
        z_i = ϕ_i (a_i)

3) Put a back through W:
        x′ = Wᵀa

4) Adjust weights in accordance with
        ΔW ∝ W + zx′ᵀ
---
"""

s = rand(2, 1000) # NOTE: memory layout should be column major, just being lazy now
G = [0.2 0.4; 0.3 0.5]
x = G * s

# Demean and whiten x
#
μ = mean(x, dims=1)
_x = x .- μ
whitening_matrix = sqrt(inv(cov(_x')))
inv_cov = inv(cov(_x'))
isapprox(whitening_matrix' * whitening_matrix, inv_cov)
__x = whitening_matrix * _x
cov(__x')

# Calculate the whitening filter.
E, D = eigen(cov(_x'));
# % Whiten the data
X_w = sqrt(pinv(D)) * E' * _x;

k = 0.0001
n_iterations = 1000

# density(x[1, :])
# density!(s[1, :])

# Initialize the algorithm variables
# k
W = [1.0 0.1; 0.1 1.0]
a = zeros(eltype(x), size(x, 1)) # NOTE: being a little wasteful with the memory here
z = zeros(eltype(x), size(x, 1))
x′ = zeros(eltype(x), size(x, 1))
ΔW = zeros(eltype(W[1, 1]), size(W))

for n in 1:n_iterations
    for x_i in eachslice(__x, dims=2)
        # x_i = eachslice(__x, dims=2)[3]
        a = W * x_i
        z = -tanh.(a)
        x′ = W' * a
        # ΔW = W + z * x′'
        # ΔW = (I - z * a') * W
        ΔW = z * x′'
        W += k * ΔW
    end
    println(n)
end

println(G)
println(inv(W))

y = inv(whitening_matrix) * W * (__x) .+ μ
# y = W * (_x .+ μ)
s

lines(y[1, :])
lines!(s[1, :])

# Try fastICA
N = 3
M = 1000
t = 1:M
sine = sin.(2 * pi * 0.01 * (1:M))
sine2 = sign.(sin.(2 * pi * 0.05 * (1:M)))
s3 = rand(Laplace(), M)
period = 50
s = [sine'; sine2'; s3']

s .+= 0.1 * randn(size(s))


G = [0.4 0.6 0.2; 0.2 0.9 0.3; 1 1 1]
x = G * s

density(x[1, :])
lines(x[2, :])

μ = mean(x, dims=2)
_x = x .- μ

eigvals, E = eigen(cov(_x'))
D = Diagonal(eigvals)
whitening_matrix = sqrt(pinv(D)) * E'
__x = whitening_matrix * _x
# cov(__x') (check)

f_method = :exp

if f_method == :logcosh
    f(u) = log(cosh(u))
    g(u) = tanh(u)
    g′(u) = 1 - tanh(u)^2
elseif f_method == :exp
    f(u) = -exp(-(u^2 / 2))
    g(u) = u * exp(-(u^2 / 2))
    g′(u) = (1 - u^2) * exp(-(u^2 / 2))
end

# Algorithm (from wikipedia)
m1 = ones(M)
C = N # number of desired components
W = randn(N, C) # unmixing matrix
S = randn(C, M)  # independent component matrix
X = __x

for p in 1:C
    w_p = randn(N)
    for i in 1:50000
        w_p = (1 / M) * (
            X * g.(w_p' * X)' .- g′.(w_p' * X) * m1 .* w_p
        )

        # if p != 1
        #     w_p .-= (w_p' * W[:, 1:(p-1)]) .* W[:, 1:(p-1)]
        # end

        for j in 1:(p-1)
            w_j = view(W, :, j)
            w_p .-= w_p' * w_j .* w_j
        end

        w_p = w_p ./ norm(w_p)
    end
    W[:, p] = w_p
end
S = W' * X

s_recons = pinv(whitening_matrix) * S .+ μ

s

fig = Figure()
for i in 1:N
    ax = Axis(fig[1, i])
    lines!(ax, S[i, :])
end
for i in 1:C
    ax = Axis(fig[2, i])
    lines!(ax, s[i, :])
end

