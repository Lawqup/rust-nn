/// Error type for RustNN
#[derive(Debug, PartialEq)]
pub enum Error {
    /// Indicates some dimension is incorrect in a Matrix operation.
    DimensionErr,
}

pub type Result<T> = std::result::Result<T, Error>;
