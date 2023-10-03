mod io;
mod analytic;

pub(crate) type FloatType = f32;

pub use io::{ read_b8_file, read_01_file };

pub use analytic::cal_2nd_order_correlation;
