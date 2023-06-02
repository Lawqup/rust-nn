use super::Matrix2;
use crate::prelude::*;
use std::ops::{Add, AddAssign, Mul, Sub};

pub trait Dot<I> {
    type Output;
    fn dot(self, rhs: I) -> Result<Self::Output>;
}

pub trait Transpose {
    fn transpose(&self) -> Self;
}

impl<T: Default + Copy> Transpose for Matrix2<T> {
    fn transpose(&self) -> Self {
        let mut transposed = Matrix2::new(self.cols(), self.rows());

        for row in 0..self.rows() {
            for col in 0..self.cols() {
                transposed[(col, row)] = self[(row, col)];
            }
        }
        transposed
    }
}

impl<'a, T> Dot<&Matrix2<T>> for &'a Matrix2<T>
where
    T: Mul<Output = T> + Default + AddAssign + Copy,
{
    type Output = Matrix2<T>;
    fn dot(self, rhs: &Matrix2<T>) -> Result<Self::Output> {
        if self.cols() != rhs.rows() {
            return Err(Error::DimensionErr);
        }

        let mut data = Vec::with_capacity(self.rows() * rhs.cols());

        for lhs_row in 0..self.rows() {
            for rhs_col in 0..rhs.cols() {
                let mut sum = T::default();
                for n in 0..self.cols() {
                    sum += self[(lhs_row, n)] * rhs[(n, rhs_col)]
                }
                data.push(sum);
            }
        }

        Ok(Matrix2 {
            data,
            dim: (self.rows(), rhs.cols()),
        })
    }
}

/// Adds two Matrix2s element-wise.
impl<'a, T> Add for &'a Matrix2<T>
where
    &'a T: Add<Output = T>,
{
    type Output = Result<Matrix2<T>>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.dim != rhs.dim {
            return Err(Error::DimensionErr);
        }

        let mut data = Vec::with_capacity(self.rows() * self.cols());
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                data.push(&self[(row, col)] + &rhs[(row, col)])
            }
        }

        Ok(Matrix2 {
            data,
            dim: (self.rows(), rhs.cols()),
        })
    }
}

/// Subs two Matrix2s element-wise.
impl<'a, T> Sub for &'a Matrix2<T>
where
    &'a T: Sub<Output = T>,
{
    type Output = Result<Matrix2<T>>;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.dim != rhs.dim {
            return Err(Error::DimensionErr);
        }

        let mut data = Vec::with_capacity(self.rows() * self.cols());
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                data.push(&self[(row, col)] - &rhs[(row, col)])
            }
        }

        Ok(Matrix2 {
            data,
            dim: (self.rows(), rhs.cols()),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::{
        ops::{Dot, Transpose},
        Matrix2,
    };
    use crate::prelude::*;

    #[test]
    fn matrix2_transpose() {
        let matrix = Matrix2::from_array([[1, 2, 3], [4, 5, 6]]).transpose();

        assert_eq!(matrix.clone().to_vec(), [[1, 4], [2, 5], [3, 6]]);
        assert_eq!(matrix.dim, (3, 2));
    }

    #[test]
    fn square_matrix_multiplication() {
        let m1 = Matrix2::from_array([[1, 2], [3, 4]]);
        let m2 = Matrix2::from_array([[3, 2], [1, 3]]);

        let m3 = m1.dot(&m2).unwrap();

        assert_eq!(m3.clone().to_vec(), [[5, 8], [13, 18]]);

        assert_eq!(m3.dim, (2, 2));
    }

    #[test]
    fn non_square_matrix_multiplication() {
        let m1 = Matrix2::from_array([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix2::from_array([[1, 2], [3, 4]]);

        let m3 = m1.dot(&m2).unwrap();
        assert_eq!(m3.clone().to_vec(), [[7, 10], [15, 22], [23, 34]]);
        assert_eq!(m3.dim, (3, 2));
    }

    #[test]
    fn matrix_multiplication_error() {
        let m1 = Matrix2::from_array([[1, 2], [3, 4], [5, 6]]).transpose();
        let m2 = Matrix2::from_array([[1, 2], [3, 4]]);

        let m3 = m1.dot(&m2);
        assert_eq!(m3, Err(Error::DimensionErr));
    }

    #[test]
    fn matrix2_addition() {
        let m1 = Matrix2::from_array([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix2::from_array([[1, 2], [3, 4], [2, 1]]);

        let m3 = &m1 + &m2;
        assert_eq!(m3.unwrap().to_vec(), [[2, 4], [6, 8], [7, 7]]);
    }

    #[test]
    fn matrix2_addition_err() {
        // unequal rows
        let m1 = Matrix2::from_array([[1, 2], [3, 4], [5, 6]]).transpose();
        let m2 = Matrix2::from_array([[1, 2], [3, 4]]);

        let m3 = &m1 + &m2;
        assert_eq!(m3, Err(Error::DimensionErr));

        // unequal cols
        let m1 = Matrix2::from_array([[1, 2], [3, 4], [5, 6]]).transpose();
        let m2 = Matrix2::from_array([[1, 2, 1], [3, 4, 1], [1, 2, 3]]);

        let m3 = &m1 + &m2;
        assert_eq!(m3, Err(Error::DimensionErr));
    }
}
