
#ifdef IGEMM_1
typedef cutlass::gemm::IgemmTraits<cutlass::MatrixLayout::kColumnMajor,
        cutlass::MatrixLayout::kRowMajor,
        cutlass::Shape<32, 128, 128> > IGemmTraits1;

run_gemm<IGemmTraits1>(512, 512, 64);
#endif


