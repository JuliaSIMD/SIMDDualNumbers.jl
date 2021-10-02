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

function test(ref, vanswer)
  answer = toaos(vanswer)
  for i ∈ eachindex(ref)
    @test ref[i] ≈ answer[i]
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
    test(map(uf, dxaos), uf(dx))
  end
  for bf ∈ [max, min]
    test(map(bf, dxaos, dyaos), bf(dx, dy))
  end

  vz = Vec(ntuple(_ -> rand(), VectorizationBase.pick_vector_width(Float64))...)
  tz = Tuple(vz)
  cmp = dx > dy
  cmpaos = dxaos .> dyaos
  test(map(ifelse, cmpaos, dxaos, dyaos), SIMDDualNumbers.ifelse(cmp, dx, dy))
  test(map(ifelse, cmpaos, tz, dyaos), SIMDDualNumbers.ifelse(cmp, vz, dy))
  test(map(ifelse, cmpaos, dxaos, tz), SIMDDualNumbers.ifelse(cmp, dx, vz))
  
  Aqua.test_all(SIMDDualNumbers, ambiguities=false) #TODO: test ambiguities once ForwardDiff fixes them, or once ForwardDiff is dropped
end
