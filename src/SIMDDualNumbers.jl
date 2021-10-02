module SIMDDualNumbers

using VectorizationBase, SLEEFPirates, ForwardDiff
using VectorizationBase: AbstractSIMD
using IfElse: ifelse

@generated function Base.abs(x::ForwardDiff.Dual{TAG,S,N}) where {TAG,S<:AbstractSIMD,N}
  quote
    $(Expr(:meta,:inline))
    val = x.value
    p = x.partials
    cmp = val < zero($S)
    absx = $ifelse(cmp, -val, val)
    Base.Cartesian.@nexprs $N n -> p_n = p[n]
    ForwardDiff.Dual{$TAG}(absx, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> $ifelse(cmp, -p_n, p_n)))
  end
end
@inline function Base.max(
  x::ForwardDiff.Dual{TAG,<:AbstractSIMD,N},
  y::ForwardDiff.Dual{TAG,<:AbstractSIMD,N}
) where {TAG,N}
  vx = ForwardDiff.value(x)
  vy = ForwardDiff.value(y)
  xgy = vx > vy
  z = ifelse(xgy, vx, vy)
  p = VectorizationBase.fmap(ifelse, xgy, ForwardDiff.partials(x).values, ForwardDiff.partials(y).values)
  ForwardDiff.Dual{TAG}(z, ForwardDiff.Partials(p))
end

@inline Base.max(x::T, y::Real) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(y::Real, x::T) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(x::T, y::Int) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))
@inline Base.max(y::Int, x::T) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = max(x, T(y))

@inline function Base.min(
  x::ForwardDiff.Dual{TAG,<:AbstractSIMD,N},
  y::ForwardDiff.Dual{TAG,<:AbstractSIMD,N}
) where {TAG,N}
  vx = ForwardDiff.value(x)
  vy = ForwardDiff.value(y)
  xgy = vx < vy
  z = ifelse(xgy, vx, vy)
  p = VectorizationBase.fmap(ifelse, xgy, ForwardDiff.partials(x).values, ForwardDiff.partials(y).values)
  ForwardDiff.Dual{TAG}(z, ForwardDiff.Partials(p))
end
@inline Base.min(x::T, y::Real) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = min(x, T(y))
@inline Base.min(y::Real, x::T) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = min(x, T(y))
@inline Base.min(x::T, y::Int) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = min(x, T(y))
@inline Base.min(y::Int, x::T) where {N,T<:ForwardDiff.Dual{<:Any,<:AbstractSIMD,N}} = min(x, T(y))

@generated function SLEEFPirates.tanh_fast(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
  quote
    $(Expr(:meta,:inline))
    t = tanh_fast(x.value)
    ∂t = $(VectorizationBase.vfnmadd_fast)(t, t, one(S))
    p = x.partials
    ForwardDiff.Dual{T}(t, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> $(Base.FastMath.mul_fast)(∂t, p[n])))
  end
end
@generated function SLEEFPirates.sigmoid_fast(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
  quote
    $(Expr(:meta,:inline))
    s = sigmoid_fast(x.value)
    ∂s =  $(VectorizationBase.vfnmadd_fast)(s,s,s)
    p = x.partials
    ForwardDiff.Dual{T}(s, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> $(Base.FastMath.mul_fast)(∂s, p[n])))
  end
end
@generated function VectorizationBase.relu(x::ForwardDiff.Dual{T,S,N}) where {T,S,N}
  quote
    $(Expr(:meta,:inline))
    v = x.value
    z = zero(v)
    cmp = v < z
    r = ifelse(cmp, z, v)
    p = x.partials
    ForwardDiff.Dual{T}(r, ForwardDiff.Partials(Base.Cartesian.@ntuple $N n -> ifelse(cmp, z, p[n])))
  end
end


end
