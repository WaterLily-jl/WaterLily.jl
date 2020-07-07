####
#### Coverage summary, printed as "(percentage) covered".
####
#### Useful for CI environments that just want a summary (eg a Gitlab setup).
####

using Coverage
cd(joinpath(@__DIR__, "..", "..")) do
    covered_lines, total_lines = get_summary(process_folder())
    percentage = covered_lines / total_lines * 100
    println("($(percentage)%) covered")
end
