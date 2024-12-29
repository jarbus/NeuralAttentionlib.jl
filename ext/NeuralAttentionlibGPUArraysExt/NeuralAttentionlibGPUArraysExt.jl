module NeuralAttentionlibGPUArraysExt

using NeuralAttentionlib
using NeuralAttentionlib: CollapsedDimsArray
using GPUArrays
using GPUArrays.GPUArraysCore
using KernelAbstractions

backend(::Type{<:CollapsedDimsArray{E, A}}) where {E, A} = KernelAbstractions.get_backend(A)

NeuralAttentionlib.ptrtypetag(arr::AnyGPUArray) = NeuralAttentionlib.ptrtypetag(backend(arr))

function NeuralAttentionlib.batched_transpose_f!(f, B::AnyGPUArray{T, 3}, A::AnyGPUArray{T, 3}) where T
    axes(B,1) == axes(A,2) && axes(B,2) == axes(A,1) && axes(A,3) == axes(B,3) || throw(DimensionMismatch(string(f)))
    GPUArrays.gpu_call(B, A) do ctx, B, A
        idx = @index(Global, Cartesian)
        @inbounds B[idx[2], idx[1], idx[3]] = f(A[idx[1], idx[2], idx[3]])
        return
    end
    return B
end

end
