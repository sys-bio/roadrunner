//
// Created by Ciaran on 08/07/2021.
//

#ifndef ROADRUNNER_MATRIX_H
#define ROADRUNNER_MATRIX_H

#include "rr-libstruct/lsMatrix.h"
#include <algorithm>

namespace rr {

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
        explicit Matrix<T>(ls::Matrix<T> &matrix) {
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

        explicit Matrix<T>(ls::Matrix<T> *matrix)
                : Matrix<T>(*matrix) {}

        /**
         * @brief Element-wise equality operator to compare a Matrix<T> with
         * another Matrix<T>
         * @note to compare Matrix<double> use Matrix<double>::almostEqual
         */
        bool operator==(Matrix<T> &other) {
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
        bool operator!=(Matrix<T> &other) {
            return !(*this == other);
        }

        /**
         * @brief Element-wise compareison between this Matrix<double> with another @param other
         * Matrix<double>
         */
        bool almostEquals(Matrix<double> &other, const double &tolerance) {
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
                    if ((this->operator()(i, j) - other(i, j)) > tolerance) {
                        equals = false;
                        break;
                    }
                }
            }
            return equals;
        }

        Matrix<T> mult(Matrix<T> matrix) {
            ls::Matrix<T> mat = ls::mult(*this, matrix);
            return Matrix < T > (mat);
        }

        /**
         * @brief indicator method for empty Matrix.
         * @return true if matrix is empty, false otherwise
         */
        bool empty() {
            if (getArray() == nullptr)
                return true;
            return false;
        }

        /**
         * @brief Reorder Matrix rows such that row names
         * are in alphabetical order.
         */
        void sortRowsByLabels() {
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
            for (int i = 0; i < rowLabels.size() - 1 ; i++) {
                for (int j = 0; j < rowLabels.size() - 1; j++) {
                    if (rowLabels[j] > rowLabels[j+1]) {
                        swapRows(j, j+1);
                        std::swap(rowLabels[j], rowLabels[j+1]);
                        swapped = true;
                    }
                }
                // little optimization
                if (!swapped){
                    break;
                }
            }
            setRowNames(rowLabels);
        }
        /**
         * @brief Reorder Matrix cols such that col names
         * are in alphabetical order.
         */
        void sortColsByLabels() {
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
            for (int i = 0; i < colLabels.size() - 1 ; i++) {
                for (int j = 0; j < colLabels.size() - 1; j++) {
                    if (colLabels[j] > colLabels[j+1]) {
                        swapCols(j, j+1);
                        std::swap(colLabels[j], colLabels[j+1]);
                        swapped = true;
                    }
                }
                // little optimization
                if (!swapped){
                    break;
                }
            }
            setColNames(colLabels);
        }
    };

}
#endif //ROADRUNNER_MATRIX_H
