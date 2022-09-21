use std::ops::Index;

pub struct Matrix2<T> {
    data: Vec<Matrix1<T>>,
    dim: (usize, usize),
}

pub struct Matrix1<T> {
    data: Vec<T>,
    dim: usize,
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
        Self {
            data: std::mem::take(vec),
            dim: vec.len(),
        }
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
}
