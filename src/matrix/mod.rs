use crate::prelude::*;
use std::ops::{Index, IndexMut};

pub mod ops;

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix2<T> {
    data: Vec<T>,
    dim: (usize, usize),
}

impl<T: Clone> Matrix2<T> {
    pub fn clone_row_to_vec(&self, row: usize) -> Vec<T> {
        (0..self.cols())
            .map(|col| self[(row, col)].clone())
            .collect()
    }

    pub fn clone_row(&self, row: usize) -> Matrix2<T> {
        Matrix2::from_row(
            (0..self.cols())
                .map(|col| self[(row, col)].clone())
                .collect(),
        )
    }
}

impl<T: Default + Clone> Matrix2<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            dim: (rows, cols),
        }
    }

    pub fn zero(&mut self) {
        for row in &mut self.data {
            *row = T::default();
        }
    }
}

impl<T> Matrix2<T> {
    pub fn from_array<const R: usize, const C: usize>(arr: [[T; C]; R]) -> Self {
        let mut data = Vec::with_capacity(R * C);

        for row in arr {
            for x in row {
                data.push(x);
            }
        }

        Self { data, dim: (R, C) }
    }

    pub fn dim(&self) -> (usize, usize) {
        self.dim
    }

    pub fn rows(&self) -> usize {
        self.dim.0
    }

    pub fn cols(&self) -> usize {
        self.dim.1
    }

    pub fn row_as_vec(&self, row: usize) -> Vec<&T> {
        (0..self.cols()).map(|col| &self[(row, col)]).collect()
    }

    pub fn from_row(row_vec: Vec<T>) -> Self {
        Self {
            dim: (1, row_vec.len()),
            data: row_vec,
        }
    }
    pub fn from_vec(vec: Vec<Vec<T>>) -> Result<Self> {
        let rows = vec.len();
        let cols = vec.get(0).map(|row| row.len()).unwrap_or(0);

        let mut data = Vec::new();
        for row in vec {
            if cols != row.len() {
                return Err(Error::DimensionErr);
            }

            for x in row {
                data.push(x);
            }
        }

        Ok(Self {
            data,
            dim: (rows, cols),
        })
    }
    pub fn to_vec(mut self) -> Vec<Vec<T>> {
        let mut res = Vec::with_capacity(self.rows());
        for _ in 0..self.rows() {
            let mut r = Vec::with_capacity(self.cols());
            for _ in 0..self.cols() {
                r.push(self.data.remove(0))
            }
            res.push(r);
        }
        res
    }

    pub fn as_vec(&self) -> Vec<Vec<&T>> {
        let mut res = Vec::with_capacity(self.rows());
        for row in 0..self.rows() {
            let mut r = Vec::with_capacity(self.cols());
            for col in 0..self.cols() {
                r.push(&self[(row, col)])
            }
            res.push(r)
        }
        res
    }
}

impl<T: Clone> Matrix2<&T> {
    pub fn clone_inner(&self) -> Matrix2<T> {
        let mut data_clone = Vec::with_capacity(self.rows() * self.cols());
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                data_clone.push(self[(row, col)].clone())
            }
        }
        Matrix2 {
            data: data_clone,
            dim: (self.rows(), self.cols()),
        }
    }
}

impl<T> Matrix2<T>
where
    T: Default,
{
    /// Applies a function to every element of the matrix
    pub fn apply<F: Fn(T) -> T>(&mut self, f: F) {
        for x in &mut self.data {
            let old = std::mem::take(x);
            let _ = std::mem::replace(x, f(old));
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix2<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[i * self.cols() + j]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix2<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        let idx = i * self.cols() + j;
        &mut self.data[idx]
    }
}

impl From<Matrix2<u32>> for Matrix2<f64> {
    fn from(value: Matrix2<u32>) -> Self {
        Self {
            dim: value.dim(),
            data: value.data.into_iter().map(|x| x as f64).collect(),
        }
    }
}

impl From<Matrix2<i32>> for Matrix2<f64> {
    fn from(value: Matrix2<i32>) -> Self {
        Self {
            dim: value.dim(),
            data: value.data.into_iter().map(|x| x as f64).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn access_matrix2_from_array() {
        let matrix = Matrix2::from_array([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 2)], 6);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 1)], 5);
    }

    #[test]
    fn matrix2_from_vec() {
        let vec = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let matrix = Matrix2::from_vec(vec).unwrap();

        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 2)], 6);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 1)], 5);
    }

    #[test]
    fn matrix2_from_vec_err() {
        let vec = vec![vec![1, 2, 3], vec![4, 5, 9], vec![1, 2]];
        let matrix = Matrix2::from_vec(vec);

        assert_eq!(matrix, Err(Error::DimensionErr));

        let vec = vec![vec![1, 2], vec![4, 5, 9], vec![1, 2, 2]];
        let matrix = Matrix2::from_vec(vec);

        assert_eq!(matrix, Err(Error::DimensionErr));
    }

    #[test]
    fn matrix2_apply() {
        let mut matrix = Matrix2::from_array([[1, 2], [2, 2], [4, 8]]);

        matrix.apply(|x| x / 2);

        assert_eq!(matrix.to_vec(), [[0, 1], [1, 1], [2, 4]]);
    }
}
