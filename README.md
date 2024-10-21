# Solving 3D Schrödinger Equations

## English Instruction Document

This program utilizes the Lanczos algorithm to solve the three-dimensional Schrödinger equation. By discretizing on the lattice in a finite volume, the solving of partial differential equations is transformed into an eigenvalue problem. Two types of Laplacian operators, the 7-point stencil and the 27-point stencil, are defined on the lattice. Some codes are customizable; for instance, the program provides two types of test potentials: harmonic oscillator potential and finite-depth spherical well potential, and one can also define a different potential. When making one- or two-dimensional slices of the wave function, one can also choose the position of the slice. Additionally, parameters can be adjusted, such as particle mass (the program uses two-nucleon system as an example) and lattice spacing (which should be adjusted according to particle mass). Note: This program utilizes natural units, with all units in $\text{MeV}$.

For detailed instructions on the Lanczos algorithm, please refer to this [repository](https://github.com/FROZEN-nyako/Lanczos_Algorithm).

## 中文说明文档

本程序利用 Lanczos 算法求解三维薛定谔方程。通过在有限体积格点上进行离散化将偏微分方程的求解转化为本征值问题，并提供了 7-point stencil 和 27-point stencil 两种拉普拉斯算子在格点上的的定义。部分代码可以自行修改，例如，本程序提供了两种测试势：谐振子势和有限深球势阱，也定义其他的势；做波函数的一维和二维切片时也可以选择切片的位置。另外，参数也可以自行设置，例如，可以修改粒子的质量（本程序以双核子系统为例），格点的间距（应该根据粒子质量调整）等。注意：本程序使用自然单位制，所有单位均取 $\text{MeV}$。

关于 Lanczos 算法的详细说明请参见[此链接](https://github.com/FROZEN-nyako/Lanczos_Algorithm)。
