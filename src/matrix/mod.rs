use rand::Rng;

use crate::prelude::*;
use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

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

    pub fn concat_rows(&mut self, mut other: Matrix2<T>) -> Result<()> {
        if self.cols() != other.cols() {
            return Err(Error::DimensionErr);
        }

        self.data.append(&mut other.data);
        self.dim.0 += other.rows();
        Ok(())
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

    pub fn as_row_major(&self) -> &Vec<T> {
        &self.data
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

    /// Shuffles the rows of two matrices with the same amount of rows
    /// as if their rows were concatenated
    pub fn shuffle_rows_synced(m1: &mut Matrix2<T>, m2: &mut Matrix2<T>) -> Result<()> {
        if m1.rows() != m2.rows() {
            return Err(Error::DimensionErr);
        }

        let mut rng = rand::thread_rng();
        let rows = m1.rows();
        let cols_m1 = m1.cols();
        let cols_m2 = m2.cols();
        for i in 0..rows {
            let rand_row = rng.gen_range(i..rows);
            for col in 0..cols_m1 {
                m1.data.swap(i * cols_m1 + col, rand_row * cols_m1 + col);
            }
            for col in 0..cols_m2 {
                m2.data.swap(i * cols_m2 + col, rand_row * cols_m2 + col);
            }
        }

        Ok(())
    }
}

impl<T: Display> Display for Matrix2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                write!(f, "{} ", self[(row, col)])?;
            }
            writeln!(f)?;
        }

        Ok(())
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

impl<T: Copy> Matrix2<T> {
    pub fn copy_rows(&self, from: usize, n: usize) -> Self {
        let end_row = (from + n).min(self.rows());
        let data = &self.data[from * self.cols()..end_row * self.cols()];
        Self {
            data: data.to_vec(),
            dim: (end_row - from, self.cols()),
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
    use std::collections::HashMap;

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

    #[test]
    fn shuffle_rows() {
        let mut relation = HashMap::new();
        relation.insert([1, 2], [9]);
        relation.insert([2, 2], [7]);
        relation.insert([4, 8], [1]);
        let mut m1 = Matrix2::from_array([[1, 2], [2, 2], [4, 8]]);
        let mut m2 = Matrix2::from_array([[9], [7], [1]]);

        assert_eq!(Ok(()), Matrix2::shuffle_rows_synced(&mut m1, &mut m2));

        println!("m1 = {m1}\nm2 = {m2}");
        assert!(m1
            .to_vec()
            .into_iter()
            .zip(m2.to_vec())
            .all(|(v1, v2)| relation[v1.as_slice()] == v2.as_slice()));
    }
}
