# Third-party provenance

The minimal SEAL-compatible backend in `third_party/SEAL/` was copied from
[VITA-Group/SEAL](https://github.com/VITA-Group/SEAL) at commit
`b2ad6858fc48143021e5df5ce10396f5c693b1ec`.

Included files are limited to:

- the modified Qwen2 inference backend used for double-newline steering;
- the MATH answer parser and grader required by the evaluation protocol;
- `get_math_results.py` and the package initializers.

SEAL's bundled `latex2sympy2.py` and `PS.g4` are included at the backend root.
The files under `third_party/SEAL/gen/` were regenerated from that grammar with
SEAL's bundled `antlr-4.11.1-complete.jar`. This avoids the incompatible PyPI
`latex2sympy2` package, which pins ANTLR 4.7.2, while the SEAL/SymPy environment
uses `antlr4-python3-runtime==4.11.1`.

`modeling_qwen2.py` retains its upstream Apache-2.0 copyright header. The math
grading code retains the provenance comments from the SEAL distribution.
