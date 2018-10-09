use ::num_traits::{Num, NumCast, cast};
use std::ops::{AddAssign, MulAssign};

#[derive(Debug)]
pub struct Poly<T: Num + NumCast> {
    coeffs: Vec<T>
}

impl<T> Poly<T>
    where T: Num + NumCast + AddAssign + MulAssign + Copy
{
    pub fn new(array: &[T]) -> Poly<T> {
        let vec = array.to_vec();
        Poly { coeffs: vec }
    }

    pub fn eval(&self, x: &T) -> T {
        let mut pow: T = cast(1.).unwrap();
        let mut result: T = cast(0.).unwrap();
        for coeff in &self.coeffs {
            result += pow * *coeff;
            pow *= *x;
        }
        result
    }

    pub fn derive(&self) -> Poly<T> {
        let mut derive_coeffs = self.coeffs[1..].to_vec();
        for (i, coeff) in derive_coeffs.iter_mut().enumerate() {
            let power: T = cast(i + 1).unwrap();
            *coeff *= power;
        }
        Poly::new(&derive_coeffs)
    }
}

use std::cmp::PartialEq;

impl<T: Num + NumCast + PartialEq> PartialEq for Poly<T> {
    fn eq(&self, rhs: &Poly<T>) -> bool {
        self.coeffs == rhs.coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval() {
        let poly = Poly::new(&[1., 2., 3.]);    // 1 + 2x + 3x^2
        assert_eq!(poly.eval(&-2.), 9.);
        assert_eq!(poly.eval(&-1.), 2.);
        assert_eq!(poly.eval(&0.), 1.);
        assert_eq!(poly.eval(&1.), 6.);
        assert_eq!(poly.eval(&2.), 17.);
    }

    #[test]
    fn test_derive() {
        let poly = Poly::new(&[1., 2., 3.]);    // 1 + 2x + 3x^2
        let poly_prime = poly.derive();
        let expected = Poly::new(&[2., 6.]);    // 1 + 4x
        assert_eq!(poly_prime, expected);
    }
}
