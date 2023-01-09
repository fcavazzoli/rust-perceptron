use rayon::prelude::*;

pub enum OptionalParams {
    NumCores(usize),
}

struct DefaultParams {
    num_cores: usize,
}

impl DefaultParams {
    fn new() -> Self {
        DefaultParams { num_cores: 1 }
    }

    fn unpack(optional_args: &[OptionalParams]) -> Self {
        let mut default_params = DefaultParams::new();
        for opt in optional_args {
            match opt {
                OptionalParams::NumCores(num_cores) => default_params.num_cores = *num_cores,
            }
        }
        default_params
    }
}
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

    fn feedforward(&self, inputs: &[f64], optional_args: &[OptionalParams]) -> f64 {
        let default_params = DefaultParams::unpack(optional_args);
        let mut sum = 0.0;

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(default_params.num_cores)
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

    pub fn train(&mut self, inputs: &[f64], target: f64, optional_args: &[OptionalParams]) {
        let default_params = DefaultParams::unpack(optional_args);

        let guess = self.feedforward(inputs, optional_args);
        let error = target - guess;

        let num_cores = default_params.num_cores;
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

    pub fn predict(&self, inputs: &[f64], optional_args: &[OptionalParams]) -> f64 {
        self.feedforward(inputs, optional_args).signum()
    }
}
