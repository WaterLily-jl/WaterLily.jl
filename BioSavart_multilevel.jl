using WaterLily
function MLArray(x)
    levels = [x]
    N = size(x)
    while all(N .|> WaterLily.divisible)
        N = @. 1+N÷2
        y = similar(x,N); fill!(y,0)
        push!(levels,y)
    end
    return levels
end

function ml_ω!(ml,a::Flow)
    # vorticity on finest grid
    @inside ml[1][I] = WaterLily.curl(3,I,a.u)
    # pool values at each level
    for l ∈ 2:lastindex(ml)
        WaterLily.restrict!(ml[l],ml[l-1])
    end
end

@inline @fastmath function biotsavart(x,j,ω,I,dx,ϵ=1e-8)
    r = x-dx*(SA[Tuple(I)...] .-0.5); i=j%2+1
    # the 2π is for 2D flows! In 3D it should be 4π 
    sign(i-j)*ω[I]*r[j]/(2π*r'*r+ϵ^2) # the curl introduces a sign change
end
function u_ω(i,x,ml)
    # initialize at coarsest level
    ui = zero(eltype(x)); j = i%2+1
    l = lastindex(ml.levels)
    R = inside(ml[l])
    Imax,dx,ω = 0,0,0

    # loop levels
    while l>=1
        # set grid scale and index nearest to x
        ω = ml[l]
        dx = 2^(l-1)
        Imax = CI(round.(Int,x/dx .+0.5)...)

        # get contributions other than Imax
        for I in R
            I != Imax && (ui += biotsavart(x,j,ω,I,dx))
        end

        # move "up" one level near Imax
        l=l-1
        R = up(Imax)
    end

    # add Imax contribution
    return ui + biotsavart(x,j,ω,Imax,dx)
end