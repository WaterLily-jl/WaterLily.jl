# Contributing

WaterLily.jl is an open-source project, and we are happy to accept contributions from the
community. Please [open an issue](https://github.com/WaterLily-jl/WaterLily.jl/issues/new/choose)
for bugs, ideas and questions, or submit a
[pull request](https://github.com/WaterLily-jl/WaterLily.jl/pulls). For larger contributions,
it helps to get in touch with the principal developers first (see [AUTHORS.md](AUTHORS.md)).

WaterLily.jl and its contributions are licensed under the MIT license (see
[LICENSE.md](LICENSE.md)).

## Pull requests

- Submit benchmark results with every pull request: WaterLily should always be fast. Run
  [WaterLily-Benchmarks](https://github.com/WaterLily-jl/WaterLily-Benchmarks) against master
  (see the README), or ask a maintainer to comment `/benchmark` on the pull request.
- Run the tests (`julia --project -e 'using Pkg; Pkg.test()'`; the header of
  `test/runtests.jl` explains how to include GPU backends) and the pre-commit checks
  (`pre-commit run -a`).
- Note user-visible changes in `NEWS.md`.
- A [review checklist](.github/review-checklist.md) is added to every new pull request. It
  lists what reviewers look for; add `[no checklist]` to the description to skip it.

## LLM/AI usage

Pull requests made with the help of LLMs and other AI tools are allowed. Because an
LLM-assisted pull request can have different problems than a human-only one, and may need a
different kind of review, please say in the description which parts an LLM contributed.

As for any pull request, we expect high quality, and the developers will work with you to get
there. You are responsible for the code you submit, whether you wrote it or an AI tool
generated it: you should understand and be able to explain it, and the existing code it
relates to. [AGENTS.md](AGENTS.md) describes the conventions of this repository for AI coding
agents.

This section is adapted from the [Trixi.jl contributing guide](https://github.com/trixi-framework/Trixi.jl/blob/main/CONTRIBUTING.md).
