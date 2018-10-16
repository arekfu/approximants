use ::num_traits::{Num, NumCast, Signed, cast};
use std::cmp::PartialOrd;
use std::fmt::Display;

#[derive(Debug)]
pub struct Poly<T: Num + NumCast> {
    coeffs: Vec<T>
}

impl<T> Poly<T>
    where T: Num + NumCast + Copy + PartialOrd + Display + Signed
{
    pub fn new(array: &[T]) -> Poly<T> {
        let vec = array.to_vec();
        Poly { coeffs: vec }
    }

    pub fn eval(&self, x: &T) -> T {
        let mut pow: T = cast(1.).unwrap();
        let mut result: T = cast(0.).unwrap();
        for coeff in &self.coeffs {
            result = result + pow * *coeff;
            pow = pow * *x;
        }
        result
    }

    pub fn derive(&self) -> Poly<T> {
        let mut derive_coeffs = self.coeffs[1..].to_vec();
        for (i, coeff) in derive_coeffs.iter_mut().enumerate() {
            let power: T = cast(i + 1).unwrap();
            *coeff = *coeff * power;
        }
        Poly::new(&derive_coeffs)
    }

    pub fn newton_raphson(&self, start: &T, tolerance: &T) -> (T, T) {
        let mut cur_x = *start;
        let mut cur_y = self.eval(&cur_x);
        let derive = self.derive();
        println!("x={}, y={}, tol={}", cur_x, cur_y, *tolerance);
        while cur_y.abs() >= *tolerance {
            cur_x = cur_x - cur_y / derive.eval(&cur_x);
            cur_y = self.eval(&cur_x);
            println!("x={}, y={}", cur_x, cur_y);
        }
        (cur_x, cur_y)
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
        let expected = Poly::new(&[2., 6.]);    // 2 + 6x
        assert_eq!(poly_prime, expected);
    }

    #[test]
    fn test_newton_raphson_1() {
        let poly = Poly::new(&[0., 1.]);    // x
        let tolerance = 1e-10;
        let (x0, y0) = poly.newton_raphson(&0.5, &tolerance);
        assert_eq!(x0, 0.);
        assert_eq!(y0, 0.);
    }

    #[test]
    fn test_newton_raphson_2() {
        {
            let poly = Poly::new(&[-2., 0., 1.]);    // x^2 - 2
            let tolerance = 1e-10;
            let (x0, y0) = poly.newton_raphson(&1.0, &tolerance);
            assert!((x0 - 2.0_f64.sqrt()).abs() < tolerance);
            assert!(y0.abs() < tolerance);
        }
        {
            let poly = Poly::new(&[-4., 0., 0., 0., 1.]);    // x^4 - 4
            let tolerance = 1e-10;
            let (x0, y0) = poly.newton_raphson(&1.0, &tolerance);
            assert!((x0 - 2.0_f64.sqrt()).abs() < tolerance);
            assert!(y0.abs() < tolerance);
        }
    }
}
