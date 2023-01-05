#![feature(test)]

use rust_perceptron::perceptron::Perceptron;
extern crate test;

#[test]
fn test_perceptron() {
    let mut p = Perceptron::new(2, 0.1);
    let training_data = [
        ([0.0, 0.0], -1.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 1.0),
    ];

    for (inputs, target) in training_data.iter() {
        p.train(inputs, *target);
    }

    let truncated = format!("{:.5}", p.bias);
    let truncated_bias: f64 = truncated.parse().unwrap();

    let weight_0 = format!("{:.5}", p.weights[0]);
    let truncated_weight_0: f64 = weight_0.parse().unwrap();

    let weight_1 = format!("{:.5}", p.weights[1]);
    let truncated_weight_1: f64 = weight_1.parse().unwrap();

    assert_eq!(truncated_bias, 0.17720);
    assert_eq!(truncated_weight_0, 0.16720);
    assert_eq!(truncated_weight_1, 0.17820);
    assert_eq!(p.predict(&[0.0, 0.0]), 1.0);
    assert_eq!(p.predict(&[0.0, 1.0]), 1.0);
    assert_eq!(p.predict(&[1.0, 0.0]), 1.0);
    assert_eq!(p.predict(&[1.0, 1.0]), 1.0);
}
