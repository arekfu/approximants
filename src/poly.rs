
use nalgebra::{DMatrix, DVector, Real};
use num_traits::{cast, Float};

use super::derive::Derive;
use super::eval::Eval;
use super::integrate::Integrate;

#[derive(Debug, PartialEq)]
pub struct Poly<T> {
    coeffs: Vec<T>,
}

impl<T: Copy> Poly<T> {
    pub fn new(array: &[T]) -> Poly<T> {
        Poly {
            coeffs: array.to_vec(),
        }
    }
}

impl<T: Float> Poly<T> {
    /// Returns the powers of `x` from x^1 up to x^(the degree of the polynomial)
    fn powers(&self, x: T) -> Vec<T> {
        Poly::powers_n(x, self.coeffs.len() - 1)
    }

    /// Returns the powers of `x` from x^1 to x^n
    fn powers_n(x: T, n: usize) -> Vec<T> {
        let mut pow = x;
        (1..n + 1)
            .map(|_i| {
                let this = pow;
                pow = pow * x;
                this
            }).collect()
    }
}

impl<T: Float> Eval for Poly<T> {
    type Var = T;
    fn eval(&self, x: T) -> T {
        self.coeffs[0] + self.coeffs[1..]
            .iter()
            .zip(self.powers(x).iter())
            .fold(cast(0.).unwrap(), |acc, (&coeff, &pow)| acc + coeff * pow)
    }
}

impl<T: Float + Real> Poly<T> {
    pub fn interpolate(xs: &[T], ys: &[T]) -> Option<Poly<T>> {
        let degree = xs.len() - 1;
        let rows: Vec<Vec<T>> = xs
            .iter()
            .map(|x| {
                let mut vec = Vec::new();
                vec.push(cast(1.).unwrap());
                vec.extend(Poly::powers_n(*x, degree).iter());
                vec
            }).collect();
        let matrix = DMatrix::from_fn(degree + 1, degree + 1, |i, j| rows[i][j]);
        let b: DVector<T> = DVector::from_fn(degree + 1, |i, _j| ys[i]);
        eprintln!("rows: {:?}", rows);
        eprintln!("matrix: {}\nb: {}", matrix, b);
        matrix
            .lu()
            .solve(&b)
            .map(|solution| Poly::new(solution.as_slice()))
    }
}

impl<T: Float> Integrate for Poly<T> {
    fn primitive(&self) -> Poly<T> {
        let int_size = self.coeffs.len() + 1;
        let mut int_coeffs = Vec::with_capacity(int_size);
        int_coeffs.push(cast(0.).unwrap());
        int_coeffs.extend(&self.coeffs[..]);
        int_coeffs[1..]
            .iter_mut()
            .enumerate()
            .for_each(|(i, coeff)| *coeff = *coeff / cast(i + 1).unwrap());
        Poly::new(&int_coeffs)
    }
}

impl<T: Float> Derive for Poly<T> {
    fn derive(&self) -> Poly<T> {
        let mut derive_coeffs = Vec::with_capacity(self.coeffs.len() - 1);
        derive_coeffs.extend(&self.coeffs[1..]);
        derive_coeffs
            .iter_mut()
            .enumerate()
            .for_each(|(i, coeff)| *coeff = *coeff * cast(i + 1).unwrap());
        Poly::new(&derive_coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::super::derive::newton_raphson;
    use super::super::integrate::integrate;
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_eval() {
        let poly = Poly::new(&[1., 2., 3.]); // 1 + 2x + 3x^2
        assert_eq!(poly.eval(-2.), 9.);
        assert_eq!(poly.eval(-1.), 2.);
        assert_eq!(poly.eval(0.), 1.);
        assert_eq!(poly.eval(1.), 6.);
        assert_eq!(poly.eval(2.), 17.);
    }

    #[test]
    fn test_derive() {
        let poly = Poly::new(&[1., 2., 3.]); // 1 + 2x + 3x^2
        let poly_prime = poly.derive();
        let expected = Poly::new(&[2., 6.]); // 2 + 6x
        assert_eq!(poly_prime, expected);
    }

    #[test]
    fn test_newton_raphson_1() {
        let poly = Poly::new(&[0., 1.]); // x
        let tolerance = 1e-10;
        let (x0, y0) = newton_raphson(&poly, 0.5, tolerance);
        assert_eq!(x0, 0.);
        assert_eq!(y0, 0.);
    }

    #[test]
    fn test_newton_raphson_2() {
        {
            let poly = Poly::new(&[-2., 0., 1.]); // x^2 - 2
            let tolerance = 1e-10;
            let (x0, y0) = newton_raphson(&poly, 1.0, tolerance);
            assert!((x0 - 2.0_f64.sqrt()).abs() < tolerance);
            assert!(y0.abs() < tolerance);
        }
        {
            let poly = Poly::new(&[-4., 0., 0., 0., 1.]); // x^4 - 4
            let tolerance = 1e-10;
            let (x0, y0) = newton_raphson(&poly, 1.0, tolerance);
            assert!((x0 - 2.0_f64.sqrt()).abs() < tolerance);
            assert!(y0.abs() < tolerance);
        }
    }

    #[test]
    fn test_primitive() {
        let poly = Poly::new(&[-2., 0., 1.]); // x^2 - 2
        let poly_int = poly.primitive();
        assert!(poly_int == Poly::new(&[0., -2., 0., 1. / 3.]));
    }

    #[test]
    fn test_derive_primitive() {
        let poly = Poly::new(&[-2., 0., 1.]); // x^2 - 2
        let poly_int = poly.primitive();
        let poly_int_prime = poly_int.derive();
        assert!(poly == poly_int_prime);
    }

    #[test]
    fn test_powers_n() {
        assert_eq!(Poly::powers_n(0., 3), vec![0., 0., 0.]);
        assert_eq!(Poly::powers_n(1., 3), vec![1., 1., 1.]);
        assert_eq!(Poly::powers_n(2., 3), vec![2., 4., 8.]);
    }

    #[test]
    fn test_integrate() {
        let tolerance = 1e-10;
        let poly = Poly::new(&[-2., 0., 1.]); // x^2 - 2
        let integral = integrate(&poly, 0., 1.);
        assert!(Float::abs(integral - 1. / 3. + 2.) < tolerance);
    }

    prop_compose! {
        fn points()(xs in prop::collection::vec(-1e1f64..1e1f64, 1..11)
                    .prop_map(|mut xs| {
                        xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                        xs.dedup();
                        xs
                    }))
            (ys in prop::collection::vec(-1e1f64..1e1f64, xs.len()), xs in Just(xs))
                -> (Vec<f64>, Vec<f64>) {
                    (xs, ys)
                }
    }

    proptest! {
        #[test]
        fn test_interpolate(pts in points()) {
            let tolerance = 1e-5;
            let (xs, ys): (Vec<f64>, _) = pts;
            let poly = match Poly::interpolate(&xs, &ys) {
                Some(pol) => pol,
                None => Poly::new(&[0.0])
            };
            for (x, y) in xs.iter().zip(ys.iter()) {
                let eval = poly.eval(*x);
                prop_assert!((eval- *y).abs() < tolerance,
                "poly: {:?}\neval: {}\nexpected: {}", poly, eval, *y);
            }
        }
    }

}
