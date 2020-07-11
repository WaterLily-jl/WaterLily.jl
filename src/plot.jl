using Plots
Plots.GRBackend()

function flood(x,y,f::Array;cfill=:RdBu_11,clims=(),kv...)
    if length(clims)==2
        @. f=min(clims[2],max(clims[1],f))
    else
        clims = (minimum(f),maximum(f))
    end
    contour(x,y,f',
        linewidth=0,linecolor=:black,
        fill = (true,palette(cfill)),
        clims = clims,kv...)
end

function body(x,y)
    plot!(Shape(x,y),c=:black,legend = false)
end
