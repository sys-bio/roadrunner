#ifndef RR_SPARSITY_PATTERN_H
#define RR_SPARSITY_PATTERN_H

#include <set>
#include <vector>

namespace rr {

    /** A sparsity pattern in compressed sparse column format, row indices sorted in each column. */
    struct CscPattern {
        int nRows = 0;
        int nCols = 0;
        std::vector<int> colPtr;
        std::vector<int> rowIdx;

        /** Number of stored entries. */
        int nnz() const { return colPtr.empty() ? 0 : colPtr.back(); }

        /** Build a pattern from the set of column indices of every row. */
        static CscPattern fromRows(int nCols, const std::vector<std::set<int>> &rows) {
            CscPattern p;
            p.nRows = (int) rows.size();
            p.nCols = nCols;
            std::vector<std::vector<int>> columns(nCols);
            for (int i = 0; i < p.nRows; ++i) {
                for (int j: rows[i]) {
                    columns[j].push_back(i);
                }
            }
            p.colPtr.assign(nCols + 1, 0);
            for (int j = 0; j < nCols; ++j) {
                p.colPtr[j + 1] = p.colPtr[j] + (int) columns[j].size();
                p.rowIdx.insert(p.rowIdx.end(), columns[j].begin(), columns[j].end());
            }
            return p;
        }
    };

}

#endif // RR_SPARSITY_PATTERN_H
