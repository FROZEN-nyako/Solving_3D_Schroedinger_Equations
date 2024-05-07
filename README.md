# Solving 3D Schrödinger Equations

English Instruction Document

This program utilizes the Lanczos algorithm to solve the three-dimensional Schrödinger equation. By discretizing on the lattice, the solving of partial differential equations is transformed into an eigenvalue problem. Two types of Laplacian operators, the 7-point stencil and the 27-point stencil, are defined on the lattice. The Lanczos algorithm is used to reduce the complexity of matrix diagonalization, significantly reducing the time required for diagonalization while maintaining accuracy. The code sections marked with TODO highlight are customizable; for instance, the program provides definitions for three types of potentials: harmonic oscillator potential, Coulomb potential, and finite-depth spherical well potential, from which one can select for experimentation. When making two-dimensional slices of the wave function, one can also choose the position of the slice. Additionally, parameters can be adjusted, such as particle mass (the program uses a two-nucleon system as an example), lattice spacing (which should be adjusted according to particle mass), and the spring constant of the harmonic oscillator. Note: This program utilizes natural units, with all units in MeV.

中文说明文档

本程序利用 Lanczos 算法求解三维薛定谔方程。通过在格点上进行离散化将偏微分方程的求解转化为本征值问题，并提供了 7-point stencil 和 27-point stencil 两种拉普拉斯算子在格点上的的定义。Lanczos 算法用于降低矩阵对角化的复杂度，在保证精度的情况下大幅减少了对角化所需的时间。TODO 高亮标记的代码是可以自行修改的部分，例如，本程序提供了三种势的定义：谐振子势、库伦势和有限深球势阱，可以选择其中一种进行试验；做波函数的二维切片时也可以选择切片的位置。另外，参数也可以自行设置，例如，可以修改粒子的质量（本程序以双核子系统为例），格点的间距（应该根据粒子质量调整），谐振子的弹簧常数等。注意：本程序使用自然单位制，所有单位均取 MeV。
