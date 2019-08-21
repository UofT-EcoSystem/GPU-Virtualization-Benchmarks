
#ifdef HGEMM_1
typedef cutlass::gemm::HgemmTraits<cutlass::MatrixLayout::kColumnMajor,
        cutlass::MatrixLayout::kRowMajor,
        cutlass::Shape<8, 128, 128> > HGemmTraits1;

run_gemm<HGemmTraits1>(2048, 2048, 1024);
#endif


