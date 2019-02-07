# Salient Isosurface Detection with Local Higher Order Moments
Implementation of:

"Salient Iso-Surface Detection with Model-Independent Statistical Signatures"
by Shivaraj Tenginakai, Jinho Lee, Raghu Machiraju

<https://ieeexplore.ieee.org/abstract/document/964516>

Computes local higher order moments (m2, m3 and m4), skew and curtosis for a
given volume (json+raw format).

## Included Libraries
- <https://github.com/nlohmann/json>
- <https://github.com/dstahlke/gnuplot-iostream>

## Other Dependencies
- Boost (tested with 1.64.0)
- C11 and C++14 compatible compilers (tested with cc and g++ version 8.2.1)
- CUDA runtime (tested with V10.0.130)

