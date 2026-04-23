using LinearAlgebra

"""
x is a vector
"""
function naive_dft(x::AbstractVector{<:T}) where {T}
    N = length(x)
    X = zeros(Complex{T}, N)
    naive_dft!(X, x, N)
    return X
end

function naive_dft!(X::AbstractVector{T},
    x::AbstractVector{K},
    N::Integer) where {T,K}

    for k in eachindex(X)
        for n in eachindex(x)
            X[k] += x[n] * exp(-2pi * 1im * ((k - 1) / N) * (n - 1))
        end
    end
    return nothing
end


"""
Note that normalization is different between 
naive DFT (1/N for ifft) and vandermonde DFT (1/√N for both)
"""
function vandermonde(N::Integer, T1=ComplexF64)
    W = Matrix{T1}(undef, N, N)
    vandermonde!(W, N)
    return W
end

function vandermonde!(W::AbstractMatrix{<:T1}, N::Integer) where {T1}
    inv_sq_N = 1 / sqrt(N)
    ω = exp((-2 * pi * 1im) / N)
    for j in 1:N
        for k in j:N
            W[j, k] = ω^((j - 1) * (k - 1))
        end
    end
    LowerTriangular(W) .= LowerTriangular(transpose(W))
    W .*= inv_sq_N
    return nothing
end

@assert (issymmetric(vandermonde(5)))
@assert isapprox(
    vandermonde(2),
    ((1 / sqrt(2)) * [1 1; 1 -1] .+ 0im)
)
@assert isapprox(
    vandermonde(4),
    (1 / sqrt(4)) * [1 1 1 1;
        1 -1im -1 1im;
        1 -1 1 -1;
        1 1im -1 -1im]
)

function vandermonde_dft!(
    X::AbstractVector{<:T1},
    W::AbstractMatrix{<:T1},
    x::AbstractVector{<:T2}
) where {T1,T2}
    X .= W * x
    return nothing
end

function vandermonde_dft(x::AbstractVector{<:T}) where {T}
    N = length(x)
    W = vandermonde(N)
    X = Vector{Complex{T}}(undef, N)
    vandermonde_dft!(X, W, x)
    return X, W
end

