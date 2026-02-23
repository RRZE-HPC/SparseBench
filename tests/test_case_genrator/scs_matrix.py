import numpy as np
import sys
from scipy.sparse import csr_array

class scsMatrix:
    # Basic dimensions
    nr: int = 0                  # number of rows
    nc: int = 0                 # number of columns
    nnz: int = 0                 # number of nonzeros

    totalNr: int = 0             # total rows (global)
    totalNnz: int = 0            # total nnz (global)

    # Ownership range
    startRow: int = None
    stopRow: int = None

    # Sparse storage
    colInd: np.ndarray = None   # dtype=int
    val: np.ndarray = None      # dtype=float

    # SCS / chunking parameters
    C: int = 0
    sigma: int = 0

    nrPadded: int = 0
    nChunks: int = 0
    nElems: int = 0

    # Chunk metadata
    chunkPtr: np.ndarray = None
    chunkLens: np.ndarray = None

    # Permutations
    oldToNewPerm: np.ndarray = None
    newToOldPerm: np.ndarray = None

    def local_rows(self) -> int:
        """Number of rows owned by this rank."""
        return self.stopRow - self.startRow

    def density(self) -> float:
        """Matrix density (nnz / (nr * nc))."""
        if self.nr == 0 or self.nc == 0:
            return 0.0
        return self.nnz / (self.nr * self.nc)

    def validate(self) -> None:
        """Basic consistency checks."""
        assert self.nr >= 0
        assert self.nc >= 0
        assert self.nnz >= 0

        if self.colInd is not None:
            assert len(self.colInd) == self.nnz

        if self.val is not None:
            assert len(self.val) == self.nnz

        if self.chunkPtr is not None and self.nChunks > 0:
            assert len(self.chunkPtr) == self.nChunks + 1

    def print_test(self, filename: str = None):
        fp = open(filename, "w") if filename else sys.stdout

        def print_array(arr, name: str, fmt: str = "d"):
            spec = "%d" if fmt == "d" else "%f"
            line = name + ": " + ", ".join(spec % v for v in arr) + ", "
            fp.write(line + "\n")

        try:
            fp.write(f"m->startRow = {self.startRow}\n")
            fp.write(f"m->stopRow = {self.stopRow}\n")
            fp.write(f"m->totalNr = {self.totalNr}\n")
            fp.write(f"m->totalNnz = {self.totalNnz}\n")
            fp.write(f"m->nr = {self.nr}\n")
            fp.write(f"m->nc = {self.nc}\n")
            fp.write(f"m->nnz = {self.nnz}\n")
            fp.write(f"m->C = {self.C}\n")
            fp.write(f"m->sigma = {self.sigma}\n")
            fp.write(f"m->nChunks = {self.nChunks}\n")
            fp.write(f"m->nrPadded = {self.nrPadded}\n")
            fp.write(f"m->nElems = {self.nElems}\n")

            print_array(self.oldToNewPerm, "oldToNewPerm")
            print_array(self.newToOldPerm, "newToOldPerm")
            print_array(self.chunkLens,    "chunkLens")
            print_array(self.chunkPtr,     "chunkPtr")
            print_array(self.colInd,       "colInd")
            print_array(self.val,          "val",    fmt="f")
        finally:
            if filename:
                fp.close()

    def print_dense(self):
        outmatrix = np.zeros((self.nr, self.nc))
    

    def __init__(self, csr_mat: csr_array, C: int, sigma: int):
        self.C = C
        self.sigma = sigma
        self.nr, self.nc = csr_mat.shape
        self.nChunks = (self.nr + self.C - 1) // self.C
        self.nrPadded = self.nChunks * self.C
        self.nnz = csr_mat.nnz
        self.totalNnz = self.nnz
        self.startRow = 0
        self.stopRow = self.nr
        self.totalNr = self.nr
        
        row_lens = np.diff(csr_mat.indptr)
        
        elems_per_row: list[tuple] = []
                         
        for idx in range(self.nrPadded):
            rowlen = 0
            if idx < row_lens.size :
                rowlen= row_lens[idx]
            elems_per_row.append((idx, int(rowlen)))
                
        def sections_then_sort_inplace(row_count_pair : list, sigma):
            n = len(row_count_pair)
            
            for i in range(0, n, sigma):
                # Define section slice
                section_slice = slice(i, min(i + sigma, n))
                # Sort the section by count
                row_count_pair[section_slice] = sorted(row_count_pair[section_slice], key=lambda x: x[1], reverse=True)

        sections_then_sort_inplace(elems_per_row, self.sigma)
        
        self.chunkLens = np.zeros(self.nChunks, dtype=int)
        self.chunkPtr = np.zeros(self.nChunks + 1, dtype=int)
        
        currentChunkPtr:int = 0
        
        for i in range(self.nChunks):
            start_idx = i * self.C
            stop_idx = min((i+1)* self.C, self.nrPadded)
            maxlengths = max(elems_per_row[start_idx : stop_idx], key= lambda x:x[1])[1]
            self.chunkLens[i] = int(maxlengths)
            self.chunkPtr[i] = currentChunkPtr
            currentChunkPtr += self.chunkLens[i] * self.C
            
        self.chunkPtr[self.nChunks] = currentChunkPtr
        self.nElems = currentChunkPtr
        
        self.oldToNewPerm = np.empty(self.nr, dtype=int)
        for i in range(self.nrPadded):
            old_row = elems_per_row[i][0]  # index is first element of tuple
            if old_row < self.nr:
                self.oldToNewPerm[old_row] = i
                
        self.newToOldPerm = np.empty(self.nr, dtype=int)
        valid = self.oldToNewPerm[0:self.nr]
        self.newToOldPerm[valid] = np.arange(self.nr)
        
        self.colInd = np.zeros(self.nElems, dtype=int)
        self.val = np.zeros(self.nElems, dtype=float)
        
        # Track how many elements we've seen in each row
        rowLocalElemCount = np.zeros(self.nrPadded, dtype=int)
        
        for i in range(self.nr):
            rowOld = i
            for j in range(csr_mat.indptr[i], csr_mat.indptr[i + 1]):
                col = csr_mat.indices[j]
                value = csr_mat.data[j]
                
                row = self.oldToNewPerm[rowOld]
                chunkIdx = row // self.C
                chunkStart = int(self.chunkPtr[chunkIdx])
                chunkRow = row % self.C
                idx = chunkStart + rowLocalElemCount[row] * self.C + chunkRow
                
                self.colInd[idx] = self.oldToNewPerm[col]
                self.val[idx] = value
                rowLocalElemCount[row] += 1
        
        
        