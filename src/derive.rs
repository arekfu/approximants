use super::eval::Eval;

pub trait Derive: Eval {
    fn derive(&self) -> Self;
}

pub fn newton_raphson<T: Derive>(function: T, start: &T::Var, tolerance: &T::Var) -> (T::Var, T::Var) {
    let mut cur_x = *start;
    let mut cur_y = function.eval(&cur_x);
    let derive = function.derive();
    while cur_y >= *tolerance || cur_y <= -*tolerance {
        cur_x = cur_x - cur_y / derive.eval(&cur_x);
        cur_y = function.eval(&cur_x);
    }
    (cur_x, cur_y)
}
