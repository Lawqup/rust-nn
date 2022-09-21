use std::ops::{AddAssign, Index, Mul};

#[derive(Debug, PartialEq)]
pub struct Matrix2<T> {
    data: Vec<Matrix1<T>>,
    dim: (usize, usize),
}

#[derive(Debug, PartialEq)]
pub struct Matrix1<T> {
    data: Vec<T>,
    dim: usize,
}

#[derive(Debug, PartialEq)]
pub enum MatrixError {
    /// Indicates some dimension is incorrect in a Matrix operation.
    DimensionErr,
}

pub trait Dot<I> {
    type Output;
    fn dot(&self, rhs: I) -> Result<Self::Output, MatrixError>;
}

impl<T> Dot<&Matrix1<T>> for Matrix2<T>
where
    T: Mul<Output = T> + Default + AddAssign + Copy,
{
    type Output = Matrix1<T>;
    fn dot(&self, rhs: &Matrix1<T>) -> Result<Self::Output, MatrixError> {
        // columns of LHS == rows of RHS
        if self.dim.1 != rhs.rows() {
            return Err(MatrixError::DimensionErr);
        }

        let mut data = Vec::new();

        for ri in 0..self.dim.0 {
            data.push(self[ri].dot(rhs).unwrap());
        }

        Ok(Matrix1::from_vec(&mut data))
    }
}

impl<T> Dot<&Matrix1<T>> for Matrix1<T>
where
    T: Mul<Output = T> + Default + AddAssign + Copy,
{
    type Output = T;
    fn dot(&self, rhs: &Matrix1<T>) -> Result<Self::Output, MatrixError> {
        // columns of LHS == rows of RHS
        if self.dim != rhs.rows() {
            return Err(MatrixError::DimensionErr);
        }

        let mut sum = T::default();
        for ri in 0..self.dim {
            sum += self[ri] * rhs[ri];
        }

        Ok(sum)
    }
}

impl<T> Matrix2<T> {
    pub fn from_array<const R: usize, const C: usize>(arr: [[T; C]; R]) -> Self {
        let mut data = Vec::new();

        for row in arr {
            data.push(Matrix1::from_array(row));
        }

        Self { data, dim: (R, C) }
    }
}

impl<T> Index<usize> for Matrix2<T> {
    type Output = Matrix1<T>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> Index<usize> for Matrix1<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> Matrix1<T> {
    pub fn from_array<const R: usize>(arr: [T; R]) -> Self {
        let mut data = Vec::new();

        for e in arr {
            data.push(e);
        }

        Self { data, dim: R }
    }

    /// Initializes matrix from a vector with the
    /// row size dimension of the vector's len.
    pub fn from_vec(vec: &mut Vec<T>) -> Self {
        let dim = vec.len();
        Self {
            data: std::mem::take(vec),
            dim,
        }
    }

    /// Returs the rows of this 1d Matrix
    pub fn rows(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_matrix1_from_array() {
        let matrix = Matrix1::from_array([3, 4, 7, 9]);
        assert_eq!(matrix[0], 3);
        assert_eq!(matrix[1], 4);
        assert_eq!(matrix[2], 7);
        assert_eq!(matrix[3], 9);
    }

    #[test]
    fn access_matrix1_from_vec() {
        let matrix = Matrix1::from_vec(&mut vec![3, 4, 7, 9]);
        assert_eq!(matrix[0], 3);
        assert_eq!(matrix[1], 4);
        assert_eq!(matrix[2], 7);
        assert_eq!(matrix[3], 9);
    }

    #[test]
    fn access_matrix2_from_vec() {
        let matrix = Matrix2::from_array([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(matrix[0][1], 2);
        assert_eq!(matrix[1][2], 6);
        assert_eq!(matrix[0][0], 1);
        assert_eq!(matrix[1][1], 5);
    }

    #[test]
    fn matrix1_dot_product() {
        let vec1 = Matrix1::from_array([3.0, 4.5, 1.2, 2.0]);
        let vec2 = Matrix1::from_array([1.0, 2.0, 4.0, 0.0]);
        assert_eq!(vec1.dot(&vec2), Ok(16.8));
    }

    #[test]
    fn matrix2_dot_product() {
        let matrix = Matrix2::from_array([[1, 2], [2, 2], [4, 8]]);
        let vec = Matrix1::from_array([1, 2]);
        assert_eq!(matrix.dot(&vec), Ok(Matrix1::from_array([5, 6, 20])));
    }

    #[test]
    fn matrix1_dot_err() {
        let vec1 = Matrix1::from_array([3.0, 4.5, 1.2, 2.0]);
        let vec2 = Matrix1::from_array([1.0, 2.0, 4.0]);
        assert_eq!(vec1.dot(&vec2), Err(MatrixError::DimensionErr));
    }

    #[test]
    fn matrix2_dot_err() {
        let matrix = Matrix2::from_array([[1, 2], [2, 2], [4, 8]]);
        let vec = Matrix1::from_array([1]);
        assert_eq!(matrix.dot(&vec), Err(MatrixError::DimensionErr));
    }
}
