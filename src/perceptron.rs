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

        sum + self.bias
    }

    pub fn train(&mut self, inputs: &[f64], target: f64, num_cores: Option<usize>) {
        let guess = self.feedforward(inputs, num_cores);
        let error = target - guess;

        let num_cores = num_cores.unwrap_or(rayon::max_num_threads());
        let mut chunk_size = self.weights.len() / num_cores;
        if chunk_size < 1 {
            chunk_size = 1;
        }

        self.weights
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * chunk_size;
                let end = start + chunk_size;
                for i in start..end {
                    chunk[i - start] += error * inputs[i] * self.learning_rate;
                }
            });

        self.bias += error * self.learning_rate;
    }

    pub fn predict(&self, inputs: &[f64]) -> f64 {
        self.feedforward(inputs, None).signum()
    }
}
