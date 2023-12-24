use array::{SpanTrait, ArrayTrait};
use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor};
use orion::numbers::{FixedTrait, FP16x16};

pub fn tensor() -> Tensor<FP16x16> {
    Tensor::<FP16x16>::new(array![10].span(), array![FixedTrait::<FP16x16>::new(47202, true), FixedTrait::<FP16x16>::new(52416, false), FixedTrait::<FP16x16>::new(5222, false), FixedTrait::<FP16x16>::new(59319, false), FixedTrait::<FP16x16>::new(195875, false), FixedTrait::<FP16x16>::new(41060, true), FixedTrait::<FP16x16>::new(157397, false), FixedTrait::<FP16x16>::new(85242, false), FixedTrait::<FP16x16>::new(3234, true), FixedTrait::<FP16x16>::new(103464, true)].span())
}
