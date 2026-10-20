using Printf
files = filter(isfile, ARGS)
meta = [first(filter(startswith('#'), readlines(f))) for f in files]
data = [[split(l, ',') for l in readlines(f) if !startswith(l, '#')] for f in files]
labels = [split(m)[2][1:end-1] for m in meta]
foreach(println, meta); println()
print(rpad("case", 13), rpad("kind", 8), rpad("variant", 20))
foreach(l -> print(lpad("$l ms", 12), lpad("×", 7)), labels); println(lpad("max|Δ|", 10))
for (k, row) in enumerate(first(data))
    print(rpad(row[1], 13), rpad(row[2], 8), rpad(row[3], 20))
    for d in data; print(@sprintf("%12.3f%6.2f×", parse(Float64, d[k][4]), parse(Float64, d[k][5]))); end
    println(@sprintf("%10.1e", maximum(parse(Float64, d[k][6]) for d in data)))
end
