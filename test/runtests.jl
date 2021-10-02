using SIMDDualNumbers, Aqua, ForwardDiff, VectorizationBase
using Test

function toaos(d::ForwardDiff.Dual{TAG,Vec{W,T},P}) where {TAG,W,T,P}
  let v = ForwardDiff.value(d),
    p = ForwardDiff.partials(d)
    ntuple(Val(W)) do w
      ForwardDiff.Dual{TAG}(v(w), ForwardDiff.Partials( ntuple(i -> p[i](w), Val(P)) ))
    end
  end
end

@testset "SIMDDualNumbers.jl" begin
  
  dx = ForwardDiff.Dual(
    Vec(ntuple(_ -> randn(), VectorizationBase.pick_vector_width(Float64))...),
    Vec(ntuple(_ -> randn(), VectorizationBase.pick_vector_width(Float64))...),
    Vec(ntuple(_ -> randn(), VectorizationBase.pick_vector_width(Float64))...)
  )
  dy = ForwardDiff.Dual(
    Vec(ntuple(_ -> rand(), VectorizationBase.pick_vector_width(Float64))...),
    Vec(ntuple(_ -> rand(), VectorizationBase.pick_vector_width(Float64))...),
    Vec(ntuple(_ -> rand(), VectorizationBase.pick_vector_width(Float64))...)
  )

  dxaos = toaos(dx)
  dyaos = toaos(dy)
  for uf ∈ [SIMDDualNumbers.tanh_fast, SIMDDualNumbers.sigmoid_fast, abs, VectorizationBase.relu]
    ref = map(uf, dxaos)
    answer = toaos(uf(dx))
    for i ∈ eachindex(ref)
      @test ref[i] ≈ answer[i]
    end
  end
  for bf ∈ [max, min]
    ref = map(bf, dxaos, dyaos)
    answer = toaos(bf(dx, dy))
    for i ∈ eachindex(ref)
      @test ref[i] ≈ answer[i]
    end
  end

  
  Aqua.test_all(SIMDDualNumbers, ambiguities=false) #TODO: test ambiguities once ForwardDiff fixes them, or once ForwardDiff is dropped
end
