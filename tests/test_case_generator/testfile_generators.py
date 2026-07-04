import os
import sys
from pathlib import Path
from scipy.sparse import csr_array
from scipy.io import mmread
import numpy as np
from scs_matrix import scsMatrix
from itertools import count

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / ".." / "data" / "expected"
DEFAULT_MATRIX = SCRIPT_DIR / ".." / "data" / "testMatrices" / "test0.mtx"


def file_gen(filename: str, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = mmread(filename).tocsr(copy=True)

    of_base = Path(filename).stem

    # # for SELL-C-SIGMA conversion test
    for sigma in range(1, 11):
        for c in range(1, 11):
            outfile_convert = out_dir / f"{of_base}_C_{c}_sigma_{sigma}.in"
            mat = scsMatrix(matrix, c, sigma)
            # mat.print_dense()
            mat.print_test(str(outfile_convert))
            print(f"{outfile_convert.name} done")

    repcount = (1, 2, 3)

    # # for SPMV test
    for i in repcount:
        outfile_spmv = out_dir / f"{of_base}_spmv_x_{i}.in"
        v = np.ones(matrix.shape[0])
        res_mult = v.copy()
        for _ in range(i):
            res_mult = matrix.toarray() @ res_mult
            # print(res_mult)
        with open(outfile_spmv, "w") as fp:
            line = "vec = " + ", ".join("%f" % val for val in res_mult) + ", "
            fp.write(line)
        print(f"{outfile_spmv.name} done")

    # # for SPMMV test
    for i in repcount:
        outfile_spmmv = out_dir / f"{of_base}_spmmv_x_{i}.in"
        shape = (matrix.shape[0], 3)
        v = np.arange(np.prod(shape)).reshape(shape)
        res_mult = v.copy()
        for _ in range(i):
            res_mult = matrix.toarray() @ res_mult
            # print(res_mult)
        with open(outfile_spmmv, "w") as fp:
            line = "row order matrix = " + ", ".join(
                "%f" % val for val in res_mult.ravel()) + ", "
            fp.write(line)
        print(f"{outfile_spmmv.name} done")


def main():
    if len(sys.argv) > 2:
        filename = sys.argv[1]
        out_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        out_dir = str(DEFAULT_OUT_DIR)
    else:
        print(f"Usage: {sys.argv[0]} <matrix_file.mtx> [output_dir]")
        filename = str(DEFAULT_MATRIX)
        out_dir = str(DEFAULT_OUT_DIR)

    file_gen(filename, out_dir)


if __name__ == "__main__":
    main()
