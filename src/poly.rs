use num_traits::{cast, Float};
use std::cmp::PartialEq;

use super::derive::Derive;
use super::eval::Eval;

#[derive(Debug)]
pub struct Poly<T> {
    coeffs: Vec<T>
}

impl<T: Copy> Poly<T> {
    pub fn new(array: &[T]) -> Poly<T> {
        Poly { coeffs: array.to_vec() }
    }
}

impl<T: PartialEq> PartialEq for Poly<T> {
    fn eq(&self, rhs: &Poly<T>) -> bool {
        self.coeffs == rhs.coeffs
    }
}

impl<T: Float> Derive for Poly<T> {
    fn derive(&self) -> Poly<T> {
        let mut derive_coeffs = Vec::with_capacity(self.coeffs.len() - 1);
        derive_coeffs.extend(&self.coeffs[1..]);
        derive_coeffs.iter_mut()
            .enumerate()
            .for_each(|(i, coeff)| *coeff = *coeff * cast(i + 1).unwrap());
        Poly::new(&derive_coeffs)
    }
}

impl<T: Float> Eval for Poly<T> {
    type Var = T;
    fn eval(&self, x: &T) -> T {
        let mut pow: T = cast(1.).unwrap();
        let mut result: T = cast(0.).unwrap();
        for coeff in &self.coeffs {
            result = result + pow * *coeff;
            pow = pow * *x;
        }
        result
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use super::super::derive::newton_raphson;

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
        let (x0, y0) = newton_raphson(poly, &0.5, &tolerance);
        assert_eq!(x0, 0.);
        assert_eq!(y0, 0.);
    }

    #[test]
    fn test_newton_raphson_2() {
        {
            let poly = Poly::new(&[-2., 0., 1.]);    // x^2 - 2
            let tolerance = 1e-10;
            let (x0, y0) = newton_raphson(poly, &1.0, &tolerance);
            assert!((x0 - 2.0_f64.sqrt()).abs() < tolerance);
            assert!(y0.abs() < tolerance);
        }
        {
            let poly = Poly::new(&[-4., 0., 0., 0., 1.]);    // x^4 - 4
            let tolerance = 1e-10;
            let (x0, y0) = newton_raphson(poly, &1.0, &tolerance);
            assert!((x0 - 2.0_f64.sqrt()).abs() < tolerance);
            assert!(y0.abs() < tolerance);
        }
    }
}
