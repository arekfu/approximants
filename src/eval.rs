use num_traits::Float;

pub trait Eval {
    type Var: Float;
    fn eval(&self, x: Self::Var) -> Self::Var;
}
