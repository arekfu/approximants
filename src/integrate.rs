use super::eval::Eval;

pub trait Integrate: Eval {
    fn primitive(&self) -> Self;
}

pub fn integrate<T: Integrate>(function: &T, from: T::Var, to: T::Var) -> T::Var {
    let prim = function.primitive();
    prim.eval(to) - prim.eval(from)
}
