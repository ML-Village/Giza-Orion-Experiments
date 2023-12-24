use array::{SpanTrait, ArrayTrait};
use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::numbers::{FixedTrait, FP16x16};

pub fn tensor() -> Tensor<FP16x16> {
    Tensor::<FP16x16>::new(array![10].span(), array![FixedTrait::<FP16x16>::new(62309, false), FixedTrait::<FP16x16>::new(36266, false), FixedTrait::<FP16x16>::new(181790, true), FixedTrait::<FP16x16>::new(108210, false), FixedTrait::<FP16x16>::new(185100, true), FixedTrait::<FP16x16>::new(212755, false), FixedTrait::<FP16x16>::new(50096, true), FixedTrait::<FP16x16>::new(125312, true), FixedTrait::<FP16x16>::new(95748, false), FixedTrait::<FP16x16>::new(68370, false)].span())
}
