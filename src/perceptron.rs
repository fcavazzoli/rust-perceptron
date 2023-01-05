use rayon::prelude::*;
pub struct Perceptron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
}

impl Perceptron {
    pub fn new(n: usize, learning_rate: f64) -> Self {
        Perceptron {
            weights: vec![0.0; n],
            bias: 0.0,
            learning_rate: learning_rate,
        }
    }

    fn feedforward(&self, inputs: &[f64], num_cores: Option<usize>) -> f64 {
        let mut sum = 0.0;
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cores.unwrap_or(1))
            .build()
            .unwrap();

        thread_pool.install(|| {
            sum = (0..self.weights.len())
                .into_par_iter()
                .map(|i| inputs[i] * self.weights[i])
                .sum::<f64>();
        });

        // let sum = (0..self.weights.len()).fold(0.0, |acc, i| acc + inputs[i] * self.weights[i]);
        sum + self.bias
    }

    pub fn train(&mut self, inputs: &[f64], target: f64, num_cores: Option<usize>) {
        let guess = self.feedforward(inputs, num_cores);
        let error = target - guess;

        (0..self.weights.len()).for_each(|i| {
            self.weights[i] += error * inputs[i] * self.learning_rate;
        });

        self.bias += error * self.learning_rate;
    }

    pub fn predict(&self, inputs: &[f64]) -> f64 {
        self.feedforward(inputs, None).signum()
    }
}
