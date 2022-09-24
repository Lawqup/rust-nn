use std::ops::{Add, AddAssign, Index, IndexMut, Mul};

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
    /// Indicates a matrix made from a vec has an unsuitable shape.
    InitErr,
}

pub trait Dot<I> {
    type Output;
    fn dot(self, rhs: I) -> Result<Self::Output, MatrixError>;
}

pub trait Transpose {
    fn transpose(&self) -> Self;
}

impl<'a, T> Dot<&Matrix1<T>> for &'a Matrix2<T>
where
    T: Mul<Output = T> + Default + AddAssign + Copy,
{
    type Output = Matrix1<T>;
    fn dot(self, rhs: &Matrix1<T>) -> Result<Self::Output, MatrixError> {
        // columns of LHS == rows of RHS
        if self.dim.1 != rhs.size() {
            return Err(MatrixError::DimensionErr);
        }

        let mut data = Vec::new();

        for ri in 0..self.dim.0 {
            data.push(self[ri].dot(rhs).unwrap());
        }

        Ok(Matrix1::from_vec(data))
    }
}

impl<'a, T> Dot<&Matrix2<T>> for &'a Matrix2<T>
where
    T: Mul<Output = T> + Default + AddAssign + Copy,
{
    type Output = Matrix2<T>;
    fn dot(self, rhs: &Matrix2<T>) -> Result<Self::Output, MatrixError> {
        // columns of LHS == rows of RHS
        if self.dim.1 != rhs.dim.0 {
            return Err(MatrixError::DimensionErr);
        }

        let mut data = Vec::new();

        for ci in 0..rhs.dim.1 {
            let mut col = Vec::new();
            for ri in 0..rhs.dim.0 {
                col.push(rhs[ri][ci]);
            }
            data.push(self.dot(&Matrix1::from_vec(col)).unwrap().to_vec());
        }

        Ok(Matrix2::from_vec(data).unwrap().transpose())
    }
}

impl<'a, T> Dot<&Matrix1<T>> for &'a Matrix1<T>
where
    T: Mul<Output = T> + Default + AddAssign + Copy,
{
    type Output = T;
    fn dot(self, rhs: &Matrix1<T>) -> Result<Self::Output, MatrixError> {
        // columns of LHS == rows of RHS
        if self.dim != rhs.size() {
            return Err(MatrixError::DimensionErr);
        }

        let mut sum = T::default();
        for ri in 0..self.dim {
            sum += self[ri] * rhs[ri];
        }

        Ok(sum)
    }
}

/// Adds two Matrix1s element-wise.
impl<'a, T> Add for &'a Matrix1<T>
where
    &'a T: Add<Output = T>,
{
    type Output = Result<Matrix1<T>, MatrixError>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.dim != rhs.dim {
            return Err(MatrixError::DimensionErr);
        }

        let mut res = Vec::new();
        for (l, r) in self.into_iter().zip(rhs) {
            res.push(l + r);
        }

        Ok(Matrix1::from_vec(res))
    }
}

/// Adds two Matrix2s element-wise.
impl<'a, T> Add for &'a Matrix2<T>
where
    &'a T: Add<Output = T>,
{
    type Output = Result<Matrix2<T>, MatrixError>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.dim != rhs.dim {
            return Err(MatrixError::DimensionErr);
        }

        let mut res = Vec::new();
        for (l, r) in self.into_iter().zip(rhs) {
            res.push((l + r).unwrap().to_vec());
        }

        Ok(Matrix2::from_vec(res).unwrap())
    }
}

/// Adds two Matrix2s element-wise.
impl<'a, T> Add<&'a Matrix1<T>> for &'a Matrix2<T>
where
    &'a T: Add<Output = T>,
{
    type Output = Result<Matrix2<T>, MatrixError>;
    fn add(self, rhs: &'a Matrix1<T>) -> Self::Output {
        // Row lengths must match
        if self.dim.1 != rhs.dim {
            return Err(MatrixError::DimensionErr);
        }

        let mut res = Vec::new();
        for lrow in self {
            res.push((lrow + rhs).unwrap().to_vec());
        }

        Ok(Matrix2::from_vec(res).unwrap())
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

    pub fn dim(&self) -> (usize, usize) {
        self.dim
    }

    pub fn column_size(&self) -> usize {
        self.dim.0
    }

    pub fn row_size(&self) -> usize {
        self.dim.1
    }

    pub fn from_vec(vec: Vec<Vec<T>>) -> Result<Self, MatrixError> {
        let mut data = Vec::new();

        let mut cols = None;
        let rows = vec.len();

        for row in vec {
            if cols.is_some() && cols.unwrap() != row.len() {
                return Err(MatrixError::InitErr);
            }

            cols = Some(row.len());
            let col = Matrix1::from_vec(row);

            data.push(col);
        }

        Ok(Self {
            data,
            dim: (rows, cols.unwrap_or(0)),
        })
    }
    pub fn to_vec(self) -> Vec<Vec<T>> {
        self.into_iter().map(|m| m.to_vec()).collect()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Matrix1<T>> {
        self.into_iter()
    }

    /// Applies a function to every element of the matrix
    pub fn apply<F: Fn(T) -> T>(&mut self, f: F) {
        for row in self.data.iter_mut() {
            row.apply(&f)
        }
    }
}

impl<T: Default + Copy> Transpose for Matrix2<T> {
    fn transpose(&self) -> Self {
        let mut transposed = (0..self.dim.1)
            .map(|_| (0..self.dim.0).map(|_| T::default()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        for ri in 0..self.dim.0 {
            for ci in 0..self.dim.1 {
                transposed[ci][ri] = self[ri][ci];
            }
        }

        Self::from_vec(transposed).unwrap()
    }
}

impl<T> Index<usize> for Matrix2<T> {
    type Output = Matrix1<T>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Matrix2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> Index<usize> for Matrix1<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Matrix1<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
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
    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let dim = vec.len();
        Self {
            data: std::mem::take(&mut vec),
            dim,
        }
    }

    pub fn to_vec(self) -> Vec<T> {
        self.data
    }

    pub fn as_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns the length of this 1-dimensional Matrix
    pub fn size(&self) -> usize {
        self.dim
    }

    /// Applies a function to every element of the matrix
    pub fn apply<F: Fn(T) -> T>(&mut self, f: F) {
        let data = std::mem::take(&mut self.data);
        self.data = data.into_iter().map(|x| f(x)).collect();
    }
}

impl<T> IntoIterator for Matrix2<T> {
    type Item = Matrix1<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Matrix2<T> {
    type Item = &'a Matrix1<T>;
    type IntoIter = std::slice::Iter<'a, Matrix1<T>>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for &'a Matrix1<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
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
        let matrix = Matrix1::from_vec(vec![3, 4, 7, 9]);
        assert_eq!(matrix[0], 3);
        assert_eq!(matrix[1], 4);
        assert_eq!(matrix[2], 7);
        assert_eq!(matrix[3], 9);
    }

    #[test]
    fn access_matrix2_from_array() {
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

    #[test]
    fn matrix1_into_iter() {
        let data = [3.0, 4.5, 1.2, 2.0];
        let matrix = Matrix1::from_array(data);

        for (e, data) in matrix.into_iter().zip(data) {
            assert_eq!(e, &data);
        }

        // test if moved
        assert_eq!(matrix.dim, 4);
    }

    #[test]
    fn matrix2_into_iter() {
        let data = [[1, 2], [2, 2], [4, 8]];
        let matrix = Matrix2::from_array(data);
        for (row, data) in matrix.iter().zip(data) {
            assert_eq!(row.data, data);
        }

        // test if moved
        assert_eq!(matrix.dim, (3, 2));
    }

    #[test]
    fn vector_addition() {
        let vec1 = Matrix1::from_array([3.0, 4.5, 1.2, 2.0]);
        let vec2 = Matrix1::from_array([1.0, 2.0, 4.0, 1.0]);
        let vec3 = &vec1 + &vec2;
        assert_eq!(vec3.unwrap().data, [4.0, 6.5, 5.2, 3.0]);
    }

    #[test]
    fn vector_addition_err() {
        let vec1 = Matrix1::from_array([3.0, 4.5, 1.2, 2.0]);
        let vec2 = Matrix1::from_array([1.0, 2.0, 4.0]);
        let vec3 = &vec1 + &vec2;
        assert_eq!(vec3, Err(MatrixError::DimensionErr));
    }

    #[test]
    fn matrix2_from_vec() {
        let vec = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let matrix = Matrix2::from_vec(vec).unwrap();

        assert_eq!(matrix[0][1], 2);
        assert_eq!(matrix[1][2], 6);
        assert_eq!(matrix[0][0], 1);
        assert_eq!(matrix[1][1], 5);
    }

    #[test]
    fn matrix2_from_vec_err() {
        let vec = vec![vec![1, 2, 3], vec![4, 5, 9], vec![1, 2]];
        let matrix = Matrix2::from_vec(vec);

        assert_eq!(matrix, Err(MatrixError::InitErr));

        let vec = vec![vec![1, 2], vec![4, 5, 9], vec![1, 2, 2]];
        let matrix = Matrix2::from_vec(vec);

        assert_eq!(matrix, Err(MatrixError::InitErr));
    }

    #[test]
    fn matrix1_to_vec() {
        let vec = vec![5, 6, 5, 6, 1];
        let matrix = Matrix1::from_vec(vec.clone());

        assert_eq!(matrix.to_vec(), vec);
    }

    #[test]
    fn matrix2_transpose() {
        let matrix = Matrix2::from_array([[1, 2, 3], [4, 5, 6]]).transpose();

        assert_eq!(
            matrix
                .data
                .into_iter()
                .map(|m| m.to_vec())
                .collect::<Vec<_>>(),
            [[1, 4], [2, 5], [3, 6]]
        );
        assert_eq!(matrix.dim, (3, 2));
    }

    #[test]
    fn square_matrix_multiplication() {
        let m1 = Matrix2::from_array([[1, 2], [3, 4]]);
        let m2 = Matrix2::from_array([[3, 2], [1, 3]]);

        let m3 = m1.dot(&m2).unwrap();

        assert_eq!(
            m3.data.into_iter().map(|m| m.to_vec()).collect::<Vec<_>>(),
            [[5, 8], [13, 18]]
        );

        assert_eq!(m3.dim, (2, 2));
    }

    #[test]
    fn non_square_matrix_multiplication() {
        let m1 = Matrix2::from_array([[1, 2], [3, 4], [5, 6]]);
        let m2 = Matrix2::from_array([[1, 2], [3, 4]]);

        let m3 = m1.dot(&m2).unwrap();
        assert_eq!(
            m3.data.into_iter().map(|m| m.to_vec()).collect::<Vec<_>>(),
            [[7, 10], [15, 22], [23, 34]]
        );
        assert_eq!(m3.dim, (3, 2));
    }

    #[test]
    fn matrix_multiplication_error() {
        let m1 = Matrix2::from_array([[1, 2], [3, 4], [5, 6]]).transpose();
        let m2 = Matrix2::from_array([[1, 2], [3, 4]]);

        let m3 = m1.dot(&m2);
        assert_eq!(m3, Err(MatrixError::DimensionErr));
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
        assert_eq!(m3, Err(MatrixError::DimensionErr));

        // unequal cols
        let m1 = Matrix2::from_array([[1, 2], [3, 4], [5, 6]]).transpose();
        let m2 = Matrix2::from_array([[1, 2, 1], [3, 4, 1], [1, 2, 3]]);

        let m3 = &m1 + &m2;
        assert_eq!(m3, Err(MatrixError::DimensionErr));
    }

    #[test]
    fn matrix2_plus_matrix1() {
        let matrix = Matrix2::from_array([[1, 2], [2, 2], [4, 8]]);
        let vec = Matrix1::from_array([1, 2]);

        let m3 = &matrix + &vec;
        assert_eq!(m3.unwrap().to_vec(), [[2, 4], [3, 4], [5, 10]]);

        let vec = Matrix1::from_array([1]);

        let m3 = &matrix + &vec;
        assert_eq!(m3, Err(MatrixError::DimensionErr));
    }

    #[test]
    fn matrix1_apply() {
        let mut matrix = Matrix1::from_array([1, 2, 3, 4, 5]);

        matrix.apply(|x| x * x);

        assert_eq!(matrix.to_vec(), [1, 4, 9, 16, 25]);
    }

    #[test]
    fn matrix2_apply() {
        let mut matrix = Matrix2::from_array([[1, 2], [2, 2], [4, 8]]);

        matrix.apply(|x| x / 2);

        assert_eq!(matrix.to_vec(), [[0, 1], [1, 1], [2, 4]]);
    }
}
