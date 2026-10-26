# Research papers and official implementations

Original upstream repositories are collected at the top level under [`implementations/`](../implementations/README.md) as pinned Git submodules. No method code is authored or modified inside that folder.

Start here:

1. [Official upstream implementation manifest](../implementations/README.md)
2. [Initialize all upstream checkouts](../implementations/clone-all.sh)
3. [Paper and derivation index](papers-and-derivations.md)

From the repository root, run `git submodule update --init --recursive` to clone all pinned upstreams. Each source keeps its own license, dependencies and history. Where an official release could not be verified, the manifest notes the gap rather than substituting unofficial code under an “official” label.

The chapter's compact educational code remains under `worldmodels/`, separate from the upstream source shelf.
