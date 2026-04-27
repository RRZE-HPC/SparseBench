import sys
from scipy.sparse import csr_array
from scipy.io import mmread
import numpy as np
from scs_matrix import scsMatrix
from itertools import count

def file_gen(filename:str):
    matrix = mmread(filename).tocsr(copy=True)
    
    of_base = filename.split('/')[-1]
    of_base = of_base.split('.')[0]
     
    # # for SELL-C-SIGMA convertion test 
    for sigma in range(1,11):
        for c in range(1,11):
            outfile_convert = f"{of_base}_C_{c}_sigma_{sigma}.in"
            mat = scsMatrix(matrix,c,sigma)
            # mat.print_dense()
            mat.print_test(outfile_convert)
            print(f"{outfile_convert} done")
    
    repcount = (1,2,3)
    
    # # for SPMV test
    for i in repcount:
        outfile_spmv = f"{of_base}_spmv_x_{i}.in"
        v = np.ones(matrix.shape[0])
        res_mult = v.copy()
        for _ in range(i):
            res_mult = matrix.toarray() @ res_mult
            # print(res_mult)
        fp = open(outfile_spmv, "w")
        line = "vec = " + ", ".join("%f" % val for val in res_mult) + ", "
        fp.write(line)
        print(f"{outfile_spmv} done")
    
    # # for SPMMV test
    for i in repcount:
        outfile_spmv = f"{of_base}_spmmv_x_{i}.in"
        shape = (matrix.shape[0], 3)
        v = np.arange(np.prod(shape)).reshape(shape)
        res_mult = v.copy()
        for _ in range(i):
            res_mult = matrix.toarray() @ res_mult
            # print(res_mult)
        fp = open(outfile_spmv, "w")
        line = "row order matrix = " + ", ".join("%f" % val for val in res_mult.ravel()) + ", "
        fp.write(line)
        print(f"{outfile_spmv} done")
    
    

def main():
    filename:str
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <matrix_file.mtx>")
        # sys.exit(1)
        filename = "./data/testMatrices/test0.mtx"
    else:
        filename = sys.argv[1]
        
    file_gen(filename)

    
    

if __name__ == "__main__":
    main()
