extern crate nalgebra;
extern crate num_traits;

#[cfg(test)]
#[macro_use]
extern crate proptest;

#[cfg(test)]
extern crate approx;

pub mod derive;
pub mod eval;
pub mod integrate;
pub mod poly;
