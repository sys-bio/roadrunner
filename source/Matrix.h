//
// Created by Ciaran on 08/07/2021.
//

#ifndef ROADRUNNER_MATRIX_H
#define ROADRUNNER_MATRIX_H

#include "rr-libstruct/lsMatrix.h"
#include <algorithm>
#include <cmath>

namespace rr {

    /**
     * @class Matrix
     * A basic local matrix class, based on the libstruct version.
     */
    template<typename T>
    class Matrix : public ls::Matrix<T> {
    public:
        using ls::Matrix<T>::Matrix;
        using ls::Matrix<T>::numCols;
        using ls::Matrix<T>::numRows;
        using ls::Matrix<T>::operator();
        using ls::Matrix<T>::operator[];
        using ls::Matrix<T>::resize;
        using ls::Matrix<T>::getArray;
        using ls::Matrix<T>::getValues;
        using ls::Matrix<T>::setColNames;
        using ls::Matrix<T>::setRowNames;
        using ls::Matrix<T>::rowNames;
        using ls::Matrix<T>::colNames;
        using ls::Matrix<T>::swapCols;
        using ls::Matrix<T>::swapRows;

        /**
         * @brief Constructor for creating a Matrix<T> from a
         * ls::Matrix<T>
         */
        explicit Matrix<T>(ls::Matrix<T> &matrix);

        /**
         * @brief Constructor for creating a Matrix<T> from a
         * ls::Matrix<T>
         */
        explicit Matrix<T>(ls::Matrix<T> *matrix);

        /**
         * @brief Element-wise equality operator to compare a Matrix<T> with
         * another Matrix<T>
         * @note to compare Matrix<double> use Matrix<double>::almostEqual
         */
        bool operator==(Matrix<T> &other);

        /**
         * @brief inequality operators
         */
        bool operator!=(Matrix<T> &other);

        /**
         * @brief Element-wise comparison between this Matrix<double> with another @param other
         * Matrix<double>
         */
        bool almostEquals(Matrix<double> &other, const double &tolerance);

        /**
         * @brief matrix multiplication operator
         */
        Matrix<T> mult(Matrix<T> matrix);

        /**
         * @brief indicator method for empty Matrix.
         * @return true if matrix is empty, false otherwise
         */
        bool empty();

        /**
         * @brief Reorder Matrix rows such that row names
         * are in alphabetical order.
         */
        void sortRowsByLabels();

        /**
         * @brief Reorder Matrix cols such that col names
         * are in alphabetical order.
         */
        void sortColsByLabels();

        /**
        * @brief delete row @param which from the matrix.
        * @details memory isn't actually reclaimed nor the
        * array resized. Instead the elements of the array
        * to be removed are shifted one by one to the far right
        * of the array that underlies the matrix. The _Row
        * member variable is decremented so the additional
        * elements are hidden from user.
        */
        void deleteRow(int which);

        /**
         * @brief delete a row with the label @param which .
         * @details interally uses deleteRow(int) overload
         * after locating the index of the correct row to
         * remove.
         * @note This method assumes that row names are unique.
         * If they are not unique then only the first
         * found row labelled @param which will be removed.
         */
        void deleteRow(const std::string& which);

        /**
         * @brief delete col @param which from the matrix.
         * @details memory isn't actually reclaimed nor the
         * array resized. Instead the elements of the array
         * to be removed are shifted one by one to the far right
         * of the array that underlies the matrix. The _Col
         * member variable is decremented so the additional
         * elements are hidden from user.
         */
        void deleteCol(int which);

        /**
         * @brief delete a col with the label @param which .
         * @details interally uses deleteCol(int) overload
         * after locating the index of the correct col to
         * remove.
         * @note This method assumes that col names are unique.
         * If they are not unique then only the first
         * found col labelled @param which will be removed.
         */
        void deleteCol(const std::string& which);
    };

    /**
     * @brief Constructor for creating a Matrix<T> from a
     * ls::Matrix<T>
     */
    template<typename T>
    Matrix<T>::Matrix(ls::Matrix<T> &matrix) {
        resize(matrix.numRows(), matrix.numCols());
        for (int i = 0; i < numRows(); i++) {
            for (int j = 0; j < numCols(); j++) {
                operator()(i, j) = matrix(i, j);
            }
        }
        // row and col names are references, so we make a copy
        colNames = std::vector<std::string>(matrix.getColNames().begin(), matrix.getColNames().end());
        rowNames = std::vector<std::string>(matrix.getRowNames().begin(), matrix.getRowNames().end());
    }

    template<typename T>
    Matrix<T>::Matrix(ls::Matrix<T> *matrix)
            : Matrix<T>(*matrix) {}

    /**
     * @brief Element-wise equality operator to compare a Matrix<T> with
     * another Matrix<T>
     * @note to compare Matrix<double> use Matrix<double>::almostEqual
     */
    template<typename T>
    bool Matrix<T>::operator==(Matrix<T> &other) {
        // if both are empty, result is true
        if (empty() && other.empty()) {
            return true;
        }
        if (numRows() != other.numRows()) {
            return false;
        }
        if (numCols() != other.numCols()) {
            return false;
        }

        bool equals = true;
        for (int i = 0; i < numRows(); i++) {
            for (int j = 0; j < numCols(); j++) {
                if (this->operator()(i, j) != other(i, j)) {
                    equals = false;
                    break;
                }
            }
        }
        return equals;
    }

    /**
     * @brief inequality operators
     */
    template<typename T>
    bool Matrix<T>::operator!=(Matrix<T> &other) {
        return !(*this == other);
    }

    /**
     * @brief Element-wise compareison between this Matrix<double> with another @param other
     * Matrix<double>
     * @param tolerance the tolerance to use between the real and expected values.
     */
    template<typename T>
    bool Matrix<T>::almostEquals(Matrix<double> &other, const double &tolerance) {
        // when both matrices are empty, they are equal
        if (empty() && other.empty()) {
            return true;
        }
        if (numRows() != other.numRows()) {
            return false;
        }
        if (numCols() != other.numCols()) {
            return false;
        }
        bool equals = true;
        for (int i = 0; i < numRows(); i++) {
            for (int j = 0; j < numCols(); j++) {
                if (!(std::abs(this->operator()(i, j) - other(i, j)) <= tolerance)) {
                    equals = false;
                    break;
                }
            }
        }
        return equals;
    }

    template<typename T>
    Matrix<T> Matrix<T>::mult(Matrix<T> matrix) {
        ls::Matrix<T> mat = ls::mult(*this, matrix);
        return Matrix<T>(mat);
    }

    /**
     * @brief indicator method for empty Matrix.
     * @return true if matrix is empty, false otherwise
     */
    template<typename T>
    bool Matrix<T>::empty() {
        if (getArray() == nullptr)
            return true;
        return false;
    }

    /**
     * @brief Reorder Matrix rows such that row names
     * are in alphabetical order.
     */
    template<typename T>
    void Matrix<T>::sortRowsByLabels() {
        if (rowNames.empty()) {
            // don't sort unlabelled matrices
            return;
        }

        // dont sort 0 and 1 length row labels
        if (numRows() < 2) {
            return;
        }

        std::vector<std::string> rowLabels = rowNames; // make copy

        // now a cheeky bubble sort algorithm
        // poor performance, but easy to implement
        bool swapped = false;
        for (int i = 0; i < rowLabels.size() - 1; i++) {
            for (int j = 0; j < rowLabels.size() - 1; j++) {
                if (rowLabels[j] > rowLabels[j + 1]) {
                    swapRows(j, j + 1);
                    std::swap(rowLabels[j], rowLabels[j + 1]);
                    swapped = true;
                }
            }
            // little optimization
            if (!swapped) {
                break;
            }
        }
        setRowNames(rowLabels);
    }

    /**
     * @brief Reorder Matrix cols such that col names
     * are in alphabetical order.
     */
    template<typename T>
    void Matrix<T>::sortColsByLabels() {
        if (colNames.empty()) {
            // don't sort unlabelled matrices
            return;
        }

        // dont sort 0 and 1 length col labels
        if (numCols() < 2) {
            return;
        }

        std::vector<std::string> colLabels = colNames; // make copy

        // now a cheeky bubble sort algorithm
        // poor performance, but easy to implement
        bool swapped = false;
        for (int i = 0; i < colLabels.size() - 1; i++) {
            for (int j = 0; j < colLabels.size() - 1; j++) {
                if (colLabels[j] > colLabels[j + 1]) {
                    swapCols(j, j + 1);
                    std::swap(colLabels[j], colLabels[j + 1]);
                    swapped = true;
                }
            }
            // little optimization
            if (!swapped) {
                break;
            }
        }
        setColNames(colLabels);
    }


    template<typename T>
    void Matrix<T>::deleteRow(int which) {
        if (which > numRows() - 1) {
            throw std::invalid_argument("Cannot delete row " + std::to_string(which));
        }
        // get the underlying array
        auto *arr = getArray();
        // store original size for later
        int originalSize = this->_Rows * this->_Cols;

        // reduce the number of rows
        this->_Rows--;
        rowNames.erase(rowNames.begin() + which);

        if (which == this->_Rows) {
            /**
             * If row is the last row,
             * there's no need to actually do any
             * deleting. Instead we just keep
             * the items there, but reduce the size of the
             * row. Users are only exposed to the
             * correct set of rows.
             */
            return;
        }

        // figure out which elements
        // in the underlying array to remove
        int firstIdx = which * this->_Cols;
        int secondIdx = which * this->_Cols + this->_Cols;
        while (secondIdx != originalSize) {
            // shift all items right of removed elements
            // to the left. We don't actually delete or
            // deallocate, but hide
            std::swap(arr[firstIdx], arr[secondIdx]);
            secondIdx++;
            firstIdx++;
        }
    }

    template<typename T>
    void Matrix<T>::deleteRow(const std::string& which){
        auto it = std::find(rowNames.begin(), rowNames.end(), which);
        if (it != rowNames.end()){
            // item has been found. Find its index
            int idx = std::distance(rowNames.begin(), it);
            deleteRow(idx);
        }
    }

    template<typename T>
    void Matrix<T>::deleteCol(int which) {
        if (which > numCols() - 1) {
            throw std::invalid_argument("Cannot delete col " + std::to_string(which));
        }
        // get the underlying array
        auto *arr = getArray();
        // store original size for later
        int originalSize = this->_Rows * this->_Cols;

        // start at the last index that we need to move
        // logic of this computation:
        //  start on the last row at "(numRows - 1) * numCols"
        //  then add the idx of column we want (which).
        int idx = which + ((this->_Rows - 1) * this->_Cols);

        while (idx >= which) {
            // if we are deleting the last column, then the item at
            // the last index (Row*Col) should not be swapped.
            // In this case, we just leave the item inplace. Reducing
            // number of columns hides the extra data from the user.
            if (idx == originalSize - 1) {
                // aaand remember to increment the idx
                idx -= this->_Rows + (this->_Cols - this->_Rows);
                continue;
            }

            for (int i = idx; i < originalSize - 1; i++) {
                std::swap(arr[i], arr[i + 1]);
            }

            // when num cols > num rows
            idx -= this->_Rows + (this->_Cols - this->_Rows);
        }

        // reduce the number of columns
        this->_Cols--;
        colNames.erase(colNames.begin() + which);
    }


    template<typename T>
    void Matrix<T>::deleteCol(const std::string& which){
        auto it = std::find(colNames.begin(), colNames.end(), which);
        if (it != colNames.end()){
            // item has been found. Find its index
            int idx = std::distance(colNames.begin(), it);
            deleteCol(idx);
        }
    }

} // namespace rr


#endif //ROADRUNNER_MATRIX_H
