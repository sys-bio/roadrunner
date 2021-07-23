//
// Created by Ciaran on 08/07/2021.
//

#ifndef ROADRUNNER_MATRIX3D_H
#define ROADRUNNER_MATRIX3D_H

#include "Matrix.h"

namespace rr {

    template<typename IndexType, typename DataType>
    class Matrix3D {
        using Matrix3DInitializer = std::initializer_list<std::initializer_list<std::initializer_list<DataType>>>;
    public:

        /**
         * @brief default construct a 3D matrix
         */
        Matrix3D() = default;

        /**
         * @brief construct an empty 3D matrix with numRows x numCols x numZ dimensions.
         */
        Matrix3D(int numRows, int numCols, int numZ)
                : index_(std::vector<IndexType>(numZ)),
                  data_(std::vector<Matrix < DataType>>

        (numZ)) {
            for (int i = 0; i < numZ; i++) {
                data_[i].resize(numRows, numCols);
            }
        }

        Matrix3D(std::initializer_list<IndexType> idx, Matrix3DInitializer data)
                : index_(idx.begin(), idx.end()), data_(data.begin(), data.end()) {
            if (index_.size() != data.size()) {
                throw std::logic_error("The size of index != size of 3D data in Matrix3D initializer list. ");
            }
        }

        void insert(IndexType idx, Matrix <DataType> mat) {
            // enforce unique indexes by override existing idx
            auto it = std::find(index_.begin(), index_.end(), idx);
            if (it != index_.end()) {
                auto pos = std::distance(index_.begin(), it);
                data_[pos] = mat;
                return;
            }
            if (!index_.empty()) {
                // All sizes are compared to index 0, so when initializing
                // from empty, we do not check integrity of user input
                // need to check that all Z have dimensions x and y
                if (mat.numRows() != numRows()) {
                    std::ostringstream err;
                    err << "Cannot insert a matrix with " << mat.numRows();
                    err << " rows into a Matrix3D that has " << numRows() << "rows";
                    throw std::invalid_argument(err.str());
                }
                if (mat.numCols() != numCols()) {
                    std::ostringstream err;
                    err << "Cannot insert a matrix with " << mat.numCols();
                    err << " rows into a Matrix3D that has " << numCols() << "rows";
                    throw std::invalid_argument(err.str());
                }
            }
            index_.push_back(idx);
            data_.push_back(mat);
        }

        void pushBack(IndexType idx, Matrix <DataType> mat) {
            if (numRows() != mat.numRows() || numCols() != mat.numCols()) {
                std::ostringstream err;
                err << "Number of rows and columns in mat are invalid for this Matrix3D (";
                err << numRows() << ", " << numCols() << ")" << std::endl;
                throw std::invalid_argument(err.str());
            }
            index_.push_back(idx);
            data_.push_back(mat);
        }

        /**
         * @brief Indexer to slice a Matrix3D and index value and data at idx
         * @returns the k'th 2D matrix, indexed by Z dimension
         */
        Matrix <DataType> &operator[](int idx) {
            if (idx > numZ()) {
                std::ostringstream err;
                err << "requested idx " << idx << " from a Matrix3D with " << numZ() << " elements";
                throw std::invalid_argument(err.str());
            }
            return data_[idx];
        }

        /**
         * @brief 1D Matrix slicer.
         * @details Convert a 3D matrix to a 2D rr::Matrix<DataType> by slicing the Z direction using numerical indices.
         *  Equivalent to Matrix3D<DataType>::operator[]
         * @param k corresponds to the number of sub matrices, indexed by k in the set 0, 1, ..., zMax.
         * @return A single 2D rr::Matrix<DataType> object at index k of this Matrix3D<DataType>
         *  with dimensions (xMax, yMax) (width by height)
         */
        rr::Matrix<DataType> &slice(int k) {
            if (k > numZ()) {
                std::ostringstream err;
                err << "requested kth index " << k << " from a Matrix3D with " << numZ()
                    << " elements in the depth direction";
                throw std::invalid_argument(err.str());
            }
            return data_[k];
        }

        /**
         * @brief 2D Matrix slicer.
         * @details Convert a 3D matrix to a 1D std::vector<DataType> by slicing the Z (depth)
         *  and then the y (height) direction using numerical indices.
         * @param k indexes the number of sub matrices, k \in 0, 1, ..., zMax.
         * @param j indexes the number of row vectors in submatrix k for j \in 0, 1, ..., yMax
         * @return A single 1D std::vector<DataType> object at index k, j of this Matrix3D<DataType>
         *  with dimensions (xMax) (width). This is a row vector at submatrix k, row j of this rr::Matrix3D<DataType>
         */
        std::vector<DataType> slice(int k, int j) {
            if (k > numZ()) {
                std::ostringstream err;
                err << "requested kth index " << k << " from a Matrix3D with " << numZ()
                    << " elements in the depth direction";
                throw std::invalid_argument(err.str());
            }
            if (j > numRows()) {
                std::ostringstream err;
                err << "requested jth index " << j << " from a Matrix3D with " << numRows()
                    << " elements in the height (y) direction";
                throw std::invalid_argument(err.str());
            }
            auto submatrix = data_[k].getValues();
            return submatrix[j];
        }

        /**
         * @brief 3D Matrix slicer.
         * @details Convert a 3D matrix to a scalar DataType by slicing the (k, j, i) (depth, height, width)
         *  direction using numerical indices.
         * @param k corresponds to the number of sub matrices, indexed by k in the set 0, 1, ..., zMax.
         * @param j indexes the number of row vectors in submatrix k for j \in 0, 1, ..., yMax
         * @param i indexes the number of elements in each row vector j; i \in 0, 1, ..., xMax
         * @return A scalar DataType object at index k, j, i of this Matrix3D<DataType>.
         */
        DataType slice(int k, int j, int i) {
            if (k > numZ()) {
                std::ostringstream err;
                err << "requested kth index " << k << " from a Matrix3D with " << numZ()
                    << " elements in the depth direction";
                throw std::invalid_argument(err.str());
            }
            if (j > numRows()) {
                std::ostringstream err;
                err << "requested jth index " << j << " from a Matrix3D with " << numRows()
                    << " elements in the width (x) direction";
                throw std::invalid_argument(err.str());
            }
            if (i > numCols()) {
                std::ostringstream err;
                err << "requested ith index " << i << " from a Matrix3D with " << numCols()
                    << " elements in the hight (y) direction";
                throw std::invalid_argument(err.str());
            }
            std::vector<std::vector<DataType>> submatrix = data_[k].getValues();
            // j indicates which row, i indicates which column
            return submatrix[j][i];
        }

        /**
         * @brief slicing operator that uses the user provided index.
         * @param idx the value of the element in this Matrix3D that has the unique index idx.
         * @return the Matrix<DataType> indexed by the IndexType idx.
         */
        Matrix <DataType> &getItem(IndexType idx) {
            // first check if idx in index
            if (std::find(index_.begin(), index_.end(), idx) == index_.end()) {
                std::ostringstream err;
                err << "Index \"" << idx << "\" has been requested but is not present in this Matrix3D.";
                throw std::invalid_argument(err.str());
            }
            // get index of idx
            int pos = std::distance(index_.begin(), std::find(index_.begin(), index_.end(), idx));
            return data_[pos];
        }

        /**
         * @brief getter for the index data field of the Matrix3D.
         * @returns a std::vector<IndexType>
         */
        const std::vector<IndexType> &getIndex() const {
            return index_;
        }

        void setKthMatrix(int k, IndexType idx, Matrix <DataType> data) {
            if (k > numZ()) {
                throw std::invalid_argument("k is too big");
            }
            if (numRows() != data.numRows() || numCols() != data.numCols()) {
                throw std::invalid_argument("wrong dimensions");
            }
            index_[k] = idx;
            data_[k] = data;

        }

        /**
         * @brief get number of rows in this 3D matrix
         * @details x is rows, y is columns, z is depth.
         */
        int numRows() {
            if (data_.empty())
                return 0;
            return data_[0].numRows();
        }

        /**
         * @brief get number of columns in this 3D matrix
         * @details if x is rows, y is columns, z is depth.
         */
        int numCols() {
            if (data_.empty())
                return 0;
            return data_[0].numCols();
        }

        /**
         * @brief get number of matrices in this 3D matrix
         * @details if x is rows, y is columns, z is depth.
         */
        int numZ() {
            if (index_.empty())
                return 0;
            return index_.size();
        }

        /**
         * @brief set row names for each of the z matrices
         */
        void setRowNames(const std::vector<std::string> &rowNames) {
            for (int i = 0; i < numZ(); i++) {
                data_[i].setRowNames(rowNames);
            }
        }

        /**
         * @brief set col names for each of the z matrices
         */
        void setColNames(const std::vector<std::string> &colNames) {
            for (int i = 0; i < numZ(); i++) {
                data_[i].setColNames(colNames);
            }

        }

        /**
         * @brief equality operator
         */
        bool operator==(Matrix3D<IndexType, DataType> &other) {
            if (numRows() != other.numRows() || numCols() != other.numCols() || numZ() != other.numZ()) {
                return false;
            }
            bool equal = true;
            for (int i = 0; i < numZ(); i++) {
                if (index_[i] != other.index_[i]) {
                    equal = false;
                    break;
                }
                if (data_[i] != other.data_[i]) {
                    equal = false;
                    break;
                }
            }
            return equal;
        }

        /**
         * @brief inequality operator
         */
        bool operator!=(Matrix3D<IndexType, DataType> &other) {
            return !(*this == other);
        }

        /**
         * @brief equality operator for double IndexType and DataType types only
         */
        bool almostEquals(Matrix3D<double, double> &other, double tol) {
            if (numRows() != other.numRows() || numCols() != other.numCols() || numZ() != other.numZ()) {
                return false;
            }
            bool equal = true;
            for (int i = 0; i < numZ(); i++) {
                if ((index_[i] - other.index_[i]) > tol) {
                    equal = false;
                    break;
                }
                if (!data_[i].almostEquals(other.data_[i], tol)) {
                    equal = false;
                    break;
                }
            }
            return equal;
        }

        template<typename IndexType_, typename DataType_>
        friend std::ostream &operator<<(std::ostream &os, Matrix3D<IndexType_, DataType_> &matrix3D);

    private:
        std::vector<IndexType> index_;
        std::vector<Matrix < DataType>> data_;
    };

    template<typename IndexType_, typename DataType_>
    std::ostream &operator<<(std::ostream &os, Matrix3D<IndexType_, DataType_> &matrix3D) {
        for (int i = 0; i < matrix3D.numZ(); i++) {
            os << "\t\t" << matrix3D.index_[i] << std::endl;
            os << matrix3D[i] << std::endl;
        }
        return os;
    }


}

#endif //ROADRUNNER_MATRIX3D_H
