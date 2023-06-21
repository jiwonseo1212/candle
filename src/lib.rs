mod cpu_backend;
mod device;
mod dtype;
mod error;
mod op;
mod shape;
mod storage;
mod strided_index;
mod tensor;

pub use cpu_backend::CpuStorage;
pub use device::Device;
pub use dtype::{DType, WithDType};
pub use error::{Error, Result};
pub use shape::Shape;
pub use storage::Storage;
use strided_index::StridedIndex;
pub use tensor::{Tensor, TensorId};