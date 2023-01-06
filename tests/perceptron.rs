#![feature(test)]

use rust_perceptron::perceptron::Perceptron;

fn truncate(number: f64) -> f64 {
    let truncated = format!("{:.5}", number);
    let truncated_number: f64 = truncated.parse().unwrap();
    truncated_number
}

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
        p.train(inputs, *target, Some(6));
    }

    assert_eq!(truncate(p.bias), 0.17720);
    assert_eq!(truncate(p.weights[0]), 0.16720);
    assert_eq!(truncate(p.weights[1]), 0.17820);
    assert_eq!(p.predict(&[0.0, 0.0]), 1.0);
    assert_eq!(p.predict(&[0.0, 1.0]), 1.0);
    assert_eq!(p.predict(&[1.0, 0.0]), 1.0);
    assert_eq!(p.predict(&[1.0, 1.0]), 1.0);
}
